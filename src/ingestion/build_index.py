"""
build_index.py — Étape 2d : calculer les embeddings des chunks et les indexer dans ChromaDB.

Lancement (depuis la racine du projet), après clean_pages et load_pdf :
    python -m src.ingestion.build_index                 # les deux stratégies
    python -m src.ingestion.build_index structure       # une seule

La collection est vidée puis reconstruite : relancer le script ne crée pas de doublons.
"""

import sys

from src.ingestion.chunking import STRATEGIES, build_chunks
from src.ingestion.vectorstore import get_vectorstore

BATCH_SIZE = 64


def build_index(strategy: str) -> int:
    """Reconstruit la collection d'une stratégie ; renvoie le nombre de chunks indexés."""
    docs = build_chunks(strategy)
    store = get_vectorstore(strategy)
    store.reset_collection()  # repart de zéro : pas de chunks obsolètes ni de doublons
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
    return len(docs)


def main() -> None:
    strategies = sys.argv[1:] or list(STRATEGIES)
    for strategy in strategies:
        count = build_index(strategy)
        print(f"{strategy:10} : {count} chunks indexés")


if __name__ == "__main__":
    main()
