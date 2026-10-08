"""
chain.py — Étape 3c : la chaîne RAG complète, question -> réponse citée.

    question -> retrieve (Chroma) -> prompt (règles + extraits numérotés) -> LLM (Ollama)
             -> le code traduit les [n] cités en sources (URL ou page du PDF)

Le LLM n'écrit jamais d'URL lui-même : il ne cite que des numéros, et c'est le code qui
retrouve la source dans les métadonnées. Un numéro inexistant est retiré de la réponse.

Lancement (depuis la racine du projet, Ollama démarré) :
    python -m src.rag.chain "Les mini macarons contiennent-ils du gluten ?"
"""

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import ResponseError

from src.ingestion.chunking import carte_citation
from src.rag.prompt import TEAM_CONTACT, build_messages
from src.rag.retriever import DEFAULT_STRATEGY, retrieve

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


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    """Client du modèle local (température 0 : réponses stables, moins d'invention).

    Sur un GPU de portable, le chargement du modèle échoue parfois (erreur CUDA au réveil
    de la carte) : on retente alors une fois.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=NUM_CTX, keep_alive=KEEP_ALIVE)
    return llm.with_retry(retry_if_exception_type=(ResponseError,), stop_after_attempt=2)


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


def answer(question: str, strategy: str = DEFAULT_STRATEGY) -> Answer:
    """Répond à une question en citant les sources récupérées."""
    docs = retrieve(question, strategy=strategy)
    raw = get_llm().invoke(build_messages(question, docs)).content
    text = remove_invalid_citations(str(raw).strip(), len(docs))
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
