"""
chain.py — Étape 3c : la chaîne RAG complète, question -> réponse citée.

    question -> retrieve (Chroma) -> prompt (règles + extraits numérotés) -> LLM (Ollama)
             -> le code traduit les [n] cités en sources (URL ou page du PDF)

Le LLM n'écrit jamais d'URL lui-même : il ne cite que des numéros, et c'est le code qui
retrouve la source dans les métadonnées. Un numéro inexistant est retiré de la réponse.

Étape 4 — une question de frais de livraison suit un autre chemin (routage) :
    question -> LLM + outil -> calculer_frais_livraison(lieu, date) exécuté par le code
             -> LLM rédige la réponse -> le code ajoute le détail exact du calcul

Lancement (depuis la racine du projet, Ollama démarré) :
    python -m src.rag.chain "Les mini macarons contiennent-ils du gluten ?"
"""

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import ResponseError

from src.ingestion.chunking import carte_citation
from src.rag.prompt import TEAM_CONTACT, build_delivery_messages, build_messages
from src.rag.retriever import DEFAULT_STRATEGY, retrieve
from src.tools.delivery import GRID_SOURCE, DeliveryQuote, ask_team, calculer_frais_livraison

# Version "instruct" : répond directement, sans phase de réflexion (la version "thinking"
# prenait ~50 s par réponse). 4 Go de VRAM : contexte limité à 4096 tokens.
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
NUM_CTX = 4096
KEEP_ALIVE = "30m"  # garde le modèle en mémoire : moins de rechargements (lents, parfois en échec)

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
CROSS_CONTAMINATION = ("Traiteur démo ne peut garantir l'absence totale d'allergènes "
                       "(contaminations croisées possibles en production).")
CROSS_CONTAMINATION_SOURCE = carte_citation({"printed_page": None, "pdf_page": 2})


@dataclass
class Answer:
    """Réponse du bot : texte, sources citées, et extraits utilisés (utiles pour RAGAS)."""
    text: str
    sources: list[tuple[list[int], str]] = field(default_factory=list)  # [([1, 6], citation), ...]
    contexts: list[Document] = field(default_factory=list)
    quotes: list[DeliveryQuote] = field(default_factory=list)  # calculs de l'outil de livraison


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


def add_allergen_notice(question: str, text: str, docs: list[Document]) -> str:
    """Pour une question d'allergènes : listes officielles + avertissement de contamination croisée."""
    if not ALLERGEN_WORDS.search(question):
        return text
    parts = [text]
    if allergens := official_allergens(text, docs):
        parts.append("Liste officielle (carte) :\n" + "\n".join(f"- {line}" for line in allergens))
    parts.append(f"⚠️ {CROSS_CONTAMINATION} Source : {CROSS_CONTAMINATION_SOURCE}. "
                 f"En cas d'allergie, confirmez auprès de l'équipe : {TEAM_CONTACT}.")
    return "\n\n".join(parts)


def run_tool_call(call: dict) -> DeliveryQuote:
    """Exécute un appel d'outil demandé par le LLM (arguments invalides -> renvoi vers l'équipe)."""
    if call["name"] != calculer_frais_livraison.__name__:
        return ask_team(f"Outil inconnu : {call['name']}.")
    try:
        return calculer_frais_livraison(**call["args"])
    except TypeError:
        return ask_team("Je n'ai pas compris le lieu ou la date de livraison.")


def add_delivery_quotes(text: str, quotes: list[DeliveryQuote]) -> str:
    """Ajoute le détail du calcul tel que l'outil l'a produit : un prix ne passe jamais
    par le texte généré sans être affiché aussi tel que le code l'a calculé."""
    for quote in quotes:
        source = f" Source : {GRID_SOURCE}." if quote.ok else ""
        text += f"\n\n🚚 {quote.message}{source}"
    return text


def is_delivery_cost_question(question: str) -> bool:
    """Routage : la question porte-t-elle sur le coût d'une livraison ?"""
    return bool(DELIVERY_WORD.search(question) and PRICE_WORD.search(question))


def answer_delivery(question: str) -> Answer:
    """Function calling : le LLM extrait lieu et date et demande l'outil, le code calcule,
    puis le LLM rédige la réponse à partir du résultat."""
    messages = build_delivery_messages(question)
    reply: AIMessage = get_llm(with_tools=True).invoke(messages)
    if not reply.tool_calls:  # pas de lieu dans la question : le LLM demande le code postal
        return Answer(text=str(reply.content).strip())
    quotes = [run_tool_call(call) for call in reply.tool_calls]
    if not any(quote.ok for quote in quotes):  # aucun prix : le message de l'outil suffit
        return Answer(text="\n\n".join(quote.message for quote in quotes), quotes=quotes)
    messages += [reply] + [ToolMessage(content=quote.message, tool_call_id=call["id"])
                           for call, quote in zip(reply.tool_calls, quotes)]
    text = str(get_llm().invoke(messages).content).strip()
    return Answer(text=add_delivery_quotes(text, quotes), quotes=quotes)


def answer(question: str, strategy: str = DEFAULT_STRATEGY) -> Answer:
    """Répond à une question : outil de livraison si elle porte sur un coût de livraison,
    sinon RAG avec citation des sources récupérées."""
    if is_delivery_cost_question(question):
        return answer_delivery(question)
    docs = retrieve(question, strategy=strategy)
    raw = str(get_llm().invoke(build_messages(question, docs)).content)
    text = remove_invalid_citations(raw.strip(), len(docs))
    return Answer(text=add_allergen_notice(question, text, docs),
                  sources=build_sources(text, docs), contexts=docs)


def format_answer(result: Answer) -> str:
    """Réponse + liste des sources, pour l'affichage dans le terminal."""
    lines = [result.text]
    if result.sources:
        lines += ["", "Sources :"] + [f"  [{', '.join(map(str, numbers))}] {citation}"
                                      for numbers, citation in result.sources]
    return "\n".join(lines)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Quels sont les délais pour commander un buffet ?"
    print(format_answer(answer(question)))


if __name__ == "__main__":
    main()
