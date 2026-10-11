"""
chain.py : La chaîne RAG complète, question -> réponse citée.

    question -> retrieve (Chroma) -> prompt (règles + extraits numérotés) -> LLM (Ollama)
             -> le code traduit les [n] cités en sources (URL ou page du PDF)

Le LLM n'écrit jamais d'URL lui-même : il ne cite que des numéros, et c'est le code qui
retrouve la source dans les métadonnées. Un numéro inexistant est retiré de la réponse.

Frais de livraison : une question de frais de livraison suit un autre chemin (routage) :
    question -> LLM + outil -> calculer_frais_livraison(lieu, date) exécuté par le code
             -> LLM rédige la réponse -> le code ajoute le détail exact du calcul

Conversation : une question de suivi ("Et samedi ?") est d'abord reformulée en
question complète à partir des derniers échanges, puis suit le chemin normal.
Les informations ajoutées par le code (allergènes, avertissement, calcul) sont des
"encadrés" séparés du texte du LLM, pour que l'interface les distingue.

Lancement (depuis la racine du projet, Ollama démarré) :
    python -m src.rag.chain "Les mini macarons contiennent-ils du gluten ?"
"""

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import ollama
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import ResponseError

from src.ingestion.chunking import carte_citation
from src.rag.prompt import (
    BRAND_NAME,
    TEAM_CONTACT,
    build_condense_messages,
    build_delivery_messages,
    build_messages,
)
from src.rag.retriever import DEFAULT_STRATEGY, retrieve
from src.tools.dates import parse_delivery_date
from src.tools.delivery import GRID_SOURCE, DeliveryQuote, ask_team, calculer_frais_livraison

# Version "instruct" : répond directement, sans phase de réflexion (la version "thinking"
# prenait ~50 s par réponse). 4 Go de VRAM : contexte limité à 4096 tokens.
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
NUM_CTX = 4096
KEEP_ALIVE = "30m"  # garde le modèle en mémoire : moins de rechargements (lents, parfois en échec)
HISTORY_TURNS = 2   # échanges gardés pour reformuler une question de suivi

# Groupe de références : [2], [1, 3], [1][4] -> on capture "2", "1, 3"...
CITATION_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
ALLERGEN_WORDS = re.compile(r"allerg|gluten|lactose|arachide|fruits? à coque|sésame|œuf|oeuf|soja"
                            r"|crustacé|poisson|moutarde|céleri|lupin|mollusque|sulfite",
                            re.IGNORECASE)
ALLERGEN_LINE = re.compile(r"^Allergènes : (.+)$", re.MULTILINE)  # format produit par load_pdf

# Question de frais de livraison : "livr..." + un mot de prix, dans n'importe quel ordre
DELIVERY_WORD = re.compile(r"\blivr", re.IGNORECASE)
PRICE_WORD = re.compile(r"combien|co[uû]t|prix|tarif|frais|€|euros?\b", re.IGNORECASE)

# Avertissement de la carte (p. 2), ajouté par le code dès qu'on parle d'allergènes
CROSS_CONTAMINATION = (f"{BRAND_NAME} ne peut garantir l'absence totale d'allergènes "
                       "(contaminations croisées possibles en production).")
CROSS_CONTAMINATION_SOURCE = carte_citation({"printed_page": None, "pdf_page": 2})


@dataclass
class Notice:
    """Encadré ajouté par le code (jamais généré par le LLM)."""
    kind: str  # "allergens" (liste officielle), "warning" (avertissement), "delivery" (calcul)
    text: str


@dataclass
class Answer:
    """Réponse du bot : texte du LLM, encadrés du code, sources citées, extraits utilisés."""
    text: str
    sources: list[tuple[list[int], str]] = field(default_factory=list)  # [([1, 6], citation), ...]
    contexts: list[Document] = field(default_factory=list)              # utiles pour RAGAS
    quotes: list[DeliveryQuote] = field(default_factory=list)          # calculs de l'outil
    notices: list[Notice] = field(default_factory=list)
    question: str = ""  # question réellement traitée (reformulée si question de suivi)


# ---------------------------------------------------------------- modèle
@lru_cache(maxsize=2)
def get_llm(with_tools: bool = False) -> Runnable:
    """Client du modèle local (température 0 : réponses stables, moins d'invention).

    with_tools : le modèle reçoit la description de l'outil de livraison et peut demander
    à l'appeler (function calling) au lieu de répondre directement.
    Sur un GPU de portable, le chargement du modèle échoue parfois (erreur CUDA au réveil
    de la carte) : on retente alors une fois.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=NUM_CTX, keep_alive=KEEP_ALIVE)
    runnable = llm.bind_tools([calculer_frais_livraison]) if with_tools else llm
    return runnable.with_retry(retry_if_exception_type=(ResponseError,), stop_after_attempt=2)


def ollama_status() -> str | None:
    """None si Ollama répond et que le modèle est installé, sinon un message pour l'utilisateur."""
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception:
        return "Ollama ne répond pas. Lancez l'application Ollama, puis rechargez la page."
    if LLM_MODEL not in installed:
        return f"Le modèle n'est pas installé. Dans un terminal : ollama pull {LLM_MODEL}"
    return None


def warm_up(strategy: str = DEFAULT_STRATEGY) -> None:
    """Charge le modèle d'embeddings, la base Chroma et le LLM avant la première question."""
    retrieve("bonjour", strategy=strategy, k=1)
    get_llm().invoke("Réponds juste : ok")


# ---------------------------------------------------------------- citations
def cited_numbers(text: str) -> list[int]:
    """Numéros cités dans le texte, dans l'ordre d'apparition, sans doublon."""
    numbers: list[int] = []
    for group in CITATION_GROUP.findall(text):
        for n in re.split(r"\s*[,;]\s*", group):
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


def remove_invalid_citations(text: str, n_docs: int) -> str:
    """Retire les références vers un extrait qui n'existe pas (ex. [9] avec 6 extraits)."""
    def keep_valid(match: re.Match) -> str:
        valid = [n for n in re.split(r"\s*[,;]\s*", match.group(1)) if 1 <= int(n) <= n_docs]
        return f"[{', '.join(valid)}]" if valid else ""
    return re.sub(r"\s+([.,;:!?])", r"\1", CITATION_GROUP.sub(keep_valid, text))


def build_sources(text: str, docs: list[Document]) -> list[tuple[list[int], str]]:
    """Traduit les numéros cités en citations lisibles ; deux extraits de la même source
    (ex. deux chunks de la page 31) sont regroupés : ([1, 6], 'Carte ..., page PDF 31')."""
    by_citation: dict[str, list[int]] = {}
    for n in cited_numbers(text):
        by_citation.setdefault(docs[n - 1].metadata["citation"], []).append(n)
    return [(numbers, citation) for citation, numbers in by_citation.items()]


# ---------------------------------------------------------------- encadrés ajoutés par le code
def official_allergens(text: str, docs: list[Document]) -> list[str]:
    """Lignes "Allergènes" recopiées telles quelles de la carte, pour chaque produit cité.

    Le petit modèle a tendance à résumer la liste ("contient des fruits à coque") : la liste
    complète est donc ajoutée par le code, mot pour mot depuis la source, jamais générée.
    """
    lines = []
    for n in cited_numbers(text):
        doc = docs[n - 1]
        match = ALLERGEN_LINE.search(doc.page_content)
        if "product" in doc.metadata and match:
            lines.append(f"{doc.metadata['product']} [{n}] — Allergènes : {match.group(1)}")
    return lines


def allergen_notices(question: str, text: str, docs: list[Document]) -> list[Notice]:
    """Pour une question d'allergènes : listes officielles + avertissement de contamination croisée."""
    if not ALLERGEN_WORDS.search(question):
        return []
    notices = []
    if allergens := official_allergens(text, docs):
        notices.append(Notice("allergens", "Liste officielle (carte) :\n"
                              + "\n".join(f"- {line}" for line in allergens)))
    notices.append(Notice("warning", f"{CROSS_CONTAMINATION} Source : {CROSS_CONTAMINATION_SOURCE}. "
                                     f"En cas d'allergie, confirmez auprès de l'équipe : {TEAM_CONTACT}."))
    return notices


def delivery_notices(quotes: list[DeliveryQuote]) -> list[Notice]:
    """Détail du calcul tel que l'outil l'a produit : un prix ne passe jamais par le texte
    généré sans être affiché aussi tel que le code l'a calculé."""
    return [Notice("delivery", f"{quote.message} Source : {GRID_SOURCE}.") for quote in quotes if quote.ok]


# ---------------------------------------------------------------- conversation
def standalone_question(question: str, history: list[tuple[str, str]] | None) -> str:
    """Reformule une question de suivi en question complète grâce aux derniers échanges.
    Sans historique, la question est gardée telle quelle (pas d'appel au LLM)."""
    if not history:
        return question
    messages = build_condense_messages(history[-HISTORY_TURNS:], question)
    rewritten = str(get_llm().invoke(messages).content).strip().strip('"«» ')
    return rewritten if 0 < len(rewritten) < 4 * len(question) + 200 else question


# ---------------------------------------------------------------- outil de livraison
def run_tool_call(call: dict, question: str) -> DeliveryQuote:
    """Exécute un appel d'outil demandé par le LLM (arguments invalides -> renvoi vers l'équipe).

    Le LLM fournit le lieu ; la date est lue dans la question par le code (parse_delivery_date),
    jamais prise au LLM : il recopiait des dates fausses (bug trouvé par l'évaluation).
    """
    if call["name"] != calculer_frais_livraison.__name__:
        return ask_team(f"Outil inconnu : {call['name']}.")
    delivery_day = parse_delivery_date(question)
    args = {**call["args"], "date_livraison": delivery_day.isoformat() if delivery_day else None}
    try:
        return calculer_frais_livraison(**args)
    except TypeError:
        return ask_team("Je n'ai pas compris le lieu ou la date de livraison.")


def is_delivery_cost_question(question: str) -> bool:
    """Routage : la question porte-t-elle sur le coût d'une livraison ?"""
    return bool(DELIVERY_WORD.search(question) and PRICE_WORD.search(question))


def answer_delivery(question: str) -> Answer:
    """Function calling : le LLM extrait lieu et date et demande l'outil, le code calcule,
    puis le LLM rédige la réponse à partir du résultat."""
    messages = build_delivery_messages(question)
    reply: AIMessage = get_llm(with_tools=True).invoke(messages)
    if not reply.tool_calls:  # pas de lieu dans la question : le LLM demande le code postal
        return Answer(text=str(reply.content).strip(), question=question)
    quotes = [run_tool_call(call, question) for call in reply.tool_calls]
    if not any(quote.ok for quote in quotes):  # aucun prix : le message de l'outil suffit
        return Answer(text="\n\n".join(q.message for q in quotes), quotes=quotes, question=question)
    messages += [reply] + [ToolMessage(content=quote.message, tool_call_id=call["id"])
                           for call, quote in zip(reply.tool_calls, quotes)]
    text = str(get_llm().invoke(messages).content).strip()
    return Answer(text=text, quotes=quotes, notices=delivery_notices(quotes), question=question)


# ---------------------------------------------------------------- point d'entrée
def answer(question: str, strategy: str = DEFAULT_STRATEGY,
           history: list[tuple[str, str]] | None = None) -> Answer:
    """Répond à une question : outil de livraison si elle porte sur un coût de livraison,
    sinon RAG avec citation des sources récupérées.

    history : derniers échanges [(question, réponse), ...] pour comprendre une question de suivi.
    """
    question = standalone_question(question, history)
    if is_delivery_cost_question(question):
        return answer_delivery(question)
    docs = retrieve(question, strategy=strategy)
    raw = str(get_llm().invoke(build_messages(question, docs)).content)
    text = remove_invalid_citations(raw.strip(), len(docs))
    return Answer(text=text, sources=build_sources(text, docs), contexts=docs,
                  notices=allergen_notices(question, text, docs), question=question)


def format_answer(result: Answer) -> str:
    """Réponse + encadrés + liste des sources, pour l'affichage dans le terminal."""
    icons = {"allergens": "📋", "warning": "⚠️", "delivery": "🚚"}
    lines = [result.text] + [f"\n{icons[n.kind]} {n.text}" for n in result.notices]
    if result.sources:
        lines += ["", "Sources :"] + [f"  [{', '.join(map(str, numbers))}] {citation}"
                                      for numbers, citation in result.sources]
    return "\n".join(lines)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Quels sont les délais pour commander un buffet ?"
    print(format_answer(answer(question)))


if __name__ == "__main__":
    main()
