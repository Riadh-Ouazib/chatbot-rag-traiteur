"""Tests de recherche dans l'index Chroma.

Ils chargent le modèle d'embeddings (~10 s) et nécessitent d'avoir lancé build_index.
"""

import pytest

from src.ingestion.vectorstore import CHROMA_DIR

pytestmark = pytest.mark.skipif(not CHROMA_DIR.exists(), reason="lancer d'abord build_index")


@pytest.fixture(scope="module")
def structure_store():
    from src.ingestion.vectorstore import get_vectorstore
    return get_vectorstore("structure")


def top_citations(store, question: str, k: int = 3) -> list[str]:
    return [d.metadata["citation"] for d in store.similarity_search(question, k=k)]


def test_index_rempli(structure_store):
    assert structure_store._collection.count() > 300


def test_question_allergenes_trouve_le_bon_produit(structure_store):
    doc = structure_store.similarity_search("Les mini macarons contiennent-ils des arachides ?", k=1)[0]
    assert doc.metadata["product"] == "MINI MACARONS"
    assert "arachides" in doc.page_content


def test_question_delai_buffet_trouve_les_regles_de_commande(structure_store):
    citations = " ".join(top_citations(structure_store, "Quel est le délai pour commander un buffet ?"))
    assert "conditions-generales-de-vente" in citations or "traiteur/livraison" in citations


def test_question_frais_de_livraison_trouve_les_tarifs(structure_store):
    docs = structure_store.similarity_search("Combien coûte la livraison à Boulogne ?", k=3)
    assert any("€" in d.page_content and "livraison" in d.page_content.lower() for d in docs)
