"""
retriever.py — Étape 3a : retrouver les extraits les plus proches d'une question.

La question est encodée avec le même modèle e5 que les chunks (préfixe "query: "),
puis Chroma renvoie les k chunks les plus proches au sens de la similarité cosinus.
"""

from langchain_core.documents import Document

from src.ingestion.vectorstore import get_vectorstore

# 6 et non 3 : certaines bonnes sources (ex. délais, carte p. 2) n'arrivent qu'en 4e-5e position
TOP_K = 6
DEFAULT_STRATEGY = "structure"  # l'étape 6 comparera avec "recursive"


def retrieve(question: str, strategy: str = DEFAULT_STRATEGY, k: int = TOP_K) -> list[Document]:
    """Renvoie les k chunks les plus pertinents pour la question, du plus proche au moins proche."""
    return get_vectorstore(strategy).similarity_search(question, k=k)
