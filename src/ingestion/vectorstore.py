"""
vectorstore.py : Modèle d'embeddings et accès à la base vectorielle ChromaDB.

Partagé entre l'indexation et la recherche : les questions doivent être
encodées par le même modèle, avec les mêmes réglages, que les documents.
"""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "data" / "chroma"

# Modèle multilingue (le contenu est en français). Les modèles e5 attendent un préfixe :
# "passage: " devant les documents, "query: " devant les questions.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Charge le modèle une seule fois (≈1 Go, téléchargé au premier lancement)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
        query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
    )


def collection_name(strategy: str) -> str:
    """Une collection Chroma par stratégie de chunking : 'traiteur_structure', 'traiteur_recursive'."""
    return f"traiteur_{strategy}"


def get_vectorstore(strategy: str) -> Chroma:
    """Ouvre (ou crée) la collection d'une stratégie, enregistrée dans data/chroma/."""
    return Chroma(
        collection_name=collection_name(strategy),
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # similarité cosinus
    )
