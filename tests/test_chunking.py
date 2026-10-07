"""Tests du découpage en chunks (étape 2c)."""

import pytest

from src.ingestion.chunking import (
    CARTE_FILE,
    MAX_SECTION_SIZE,
    PAGES_FILE,
    STRATEGIES,
    build_chunks,
    split_sections,
)


def test_split_sections_titres_et_articles():
    text = "Intro\n## Paiement\nCarte ou chèque\nArticle 2 ANNULATION\n48h avant"
    sections = split_sections(text)
    assert [h for h, _ in sections] == ["", "Paiement", "Article 2 ANNULATION"]
    assert sections[2][1] == "Article 2 ANNULATION\n48h avant"


def test_split_sections_sans_titre():
    assert split_sections("Juste un paragraphe.") == [("", "Juste un paragraphe.")]


# ---------------------------------------------------------------- tests sur les vraies données
has_data = pytest.mark.skipif(not (PAGES_FILE.exists() and CARTE_FILE.exists()),
                              reason="lancer d'abord clean_pages et load_pdf")


@has_data
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chaque_chunk_a_une_source_citable(strategy):
    for doc in build_chunks(strategy):
        assert doc.metadata["citation"], doc.page_content[:80]
        assert doc.metadata["source_type"] in ("site", "carte")
        assert all(v is not None for v in doc.metadata.values())  # Chroma refuse None


@has_data
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ids_uniques_et_taille_bornee(strategy):
    docs = build_chunks(strategy)
    assert len({d.metadata["chunk_id"] for d in docs}) == len(docs)
    for doc in docs:
        assert len(doc.page_content) <= MAX_SECTION_SIZE + 200, doc.metadata["citation"]


@has_data
def test_structure_un_chunk_par_produit_avec_ses_allergenes():
    products = [d for d in build_chunks("structure") if "product" in d.metadata]
    assert len(products) >= 70
    for doc in products:
        assert doc.page_content.count("Allergènes") == 1, doc.metadata["product"]


@has_data
def test_structure_article_cgv_isole():
    docs = build_chunks("structure")
    article_2 = [d for d in docs if "> Article 2 CONDITIONS D’ANNULATION" in d.page_content]
    assert len(article_2) == 1
    assert "48h avant la prestation" in article_2[0].page_content
