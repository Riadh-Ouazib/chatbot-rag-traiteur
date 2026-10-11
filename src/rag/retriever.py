"""
retriever.py : Retrouver les extraits les plus proches d'une question.

La question est encodée avec le même modèle e5 que les chunks (préfixe "query: "),
puis Chroma renvoie les k chunks les plus proches au sens de la similarité cosinus.
"""

from langchain_core.documents import Document

from src.ingestion.vectorstore import get_vectorstore

# 6 et non 3 : certaines bonnes sources (ex. délais, carte p. 2) n'arrivent qu'en 4e-5e position
TOP_K = 6
DEFAULT_STRATEGY = "structure"  # meilleur score que "recursive" (voir eval/)
# Budget de contexte : ~3 caractères par token en français, le modèle a 4096 tokens en tout
# (règles + extraits + réponse). Certains chunks (listes de codes postaux) sont très longs.
MAX_CONTEXT_CHARS = 7000


def fit_budget(docs: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> list[Document]:
    """Garde les extraits dans l'ordre de pertinence tant qu'ils tiennent dans le budget
    (le premier est toujours gardé)."""
    kept, total = [], 0
    for doc in docs:
        if kept and total + len(doc.page_content) > max_chars:
            continue
        kept.append(doc)
        total += len(doc.page_content)
    return kept


def retrieve(question: str, strategy: str = DEFAULT_STRATEGY, k: int = TOP_K) -> list[Document]:
    """Renvoie jusqu'à k chunks pertinents pour la question, du plus proche au moins proche."""
    return fit_budget(get_vectorstore(strategy).similarity_search(question, k=k))
