"""Tests de la chaîne RAG (étape 3).

Les tests unitaires n'appellent pas le LLM. Les tests "bout en bout" sont sautés si Ollama
ou l'index Chroma ne sont pas disponibles (~5-15 s par question sur un GPU de portable).
"""

import pytest
from langchain_core.documents import Document

from src.ingestion.vectorstore import CHROMA_DIR
from src.rag.chain import (
    LLM_MODEL,
    allergen_notices,
    build_sources,
    cited_numbers,
    format_answer,
    remove_invalid_citations,
)
from src.rag.prompt import TEAM_CONTACT, build_messages, format_context

SITE_DOC = Document(page_content="Livraison 7j/7", metadata={"citation": "Livraison (https://x.com/livraison)"})
CARTE_DOC = Document(page_content="Produit : MINI MACARONS\nAllergènes : gluten, œufs.\n24,00 € HT",
                     metadata={"citation": "Carte, p. 24", "product": "MINI MACARONS"})
PAGE_31 = Document(page_content="Annulation 48h avant", metadata={"citation": "Carte, page PDF 31"})


# ---------------------------------------------------------------- sans LLM
def test_format_context_numerote_avec_la_source():
    context = format_context([SITE_DOC, CARTE_DOC])
    assert context.startswith("[1] (source : Livraison (https://x.com/livraison))\nLivraison 7j/7")
    assert "[2] (source : Carte, p. 24)" in context


def test_build_messages_regles_puis_question():
    (role_sys, system), (role_user, user) = build_messages("Délais ?", [SITE_DOC])
    assert (role_sys, role_user) == ("system", "human")
    assert TEAM_CONTACT in system
    assert user.endswith("Question : Délais ?")


@pytest.mark.parametrize("text, expected", [
    ("Oui [2]. Et aussi [1].", [2, 1]),
    ("Voir [1, 3] puis [3][2].", [1, 3, 2]),
    ("Aucune source.", []),
])
def test_cited_numbers(text, expected):
    assert cited_numbers(text) == expected


def test_numero_invente_retire():
    assert remove_invalid_citations("Vrai [1]. Inventé [9].", n_docs=6) == "Vrai [1]. Inventé."
    assert remove_invalid_citations("Mixte [2, 8].", n_docs=6) == "Mixte [2]."


def test_sources_traduites_et_regroupees():
    docs = [PAGE_31, SITE_DOC, PAGE_31]
    assert build_sources("A [3]. B [2]. C [1].", docs) == [
        ([3, 1], "Carte, page PDF 31"),
        ([2], "Livraison (https://x.com/livraison)"),
    ]


def test_allergenes_liste_officielle_recopiee_et_avertissement():
    allergens, warning = allergen_notices("Les macarons contiennent-ils du gluten ?", "Oui [1].", [CARTE_DOC])
    assert allergens.kind == "allergens" and "MINI MACARONS [1] — Allergènes : gluten, œufs." in allergens.text
    assert warning.kind == "warning"
    assert "contaminations croisées" in warning.text and TEAM_CONTACT in warning.text


def test_pas_d_avertissement_hors_allergenes():
    assert allergen_notices("Quels délais ?", "48h [1].", [CARTE_DOC]) == []


# ---------------------------------------------------------------- bout en bout (LLM réel)
def ollama_ready() -> bool:
    """Vrai si le serveur Ollama répond et que le modèle est installé."""
    try:
        import ollama
        return any(m.model == LLM_MODEL for m in ollama.list().models)
    except Exception:
        return False


llm = pytest.mark.skipif(not (CHROMA_DIR.exists() and ollama_ready()),
                         reason="Ollama, le modèle ou l'index Chroma ne sont pas disponibles")


@pytest.fixture(scope="module")
def ask():
    from src.rag.chain import answer
    return answer


@llm
def test_delai_plateaux_repas_16h_cite_la_carte(ask):
    result = ask("Jusqu'à quelle heure puis-je commander des plateaux repas pour demain ?")
    assert "16" in result.text and "17h" not in result.text
    assert result.sources


@llm
def test_question_hors_sujet_renvoie_vers_l_equipe(ask):
    result = ask("Quelle est la capitale de l'Australie ?")
    assert "Canberra" not in result.text
    assert "01 00 00 00 00" in format_answer(result)


@llm
def test_produit_sans_allergenes_indiques_renvoie_vers_l_equipe(ask):
    result = ask("Quels allergènes dans les brochettes de fruits ?")
    assert "non indiqués" in result.text
    assert "gluten" not in result.text.lower()
    assert "01 00 00 00 00" in format_answer(result)


@llm
def test_allergenes_macarons_liste_complete(ask):
    result = ask("Les mini macarons contiennent-ils des fruits à coque ?")
    assert "arachides, fruits à coque, gluten, lactose, lupin, œufs, soja, sésame, sulfites" in format_answer(result)
    assert any("p. 24" in citation for _, citation in result.sources)


# ---------------------------------------------------------------- étape 4 : routage et outil
@pytest.mark.parametrize("question, expected", [
    ("Combien coûte la livraison à Boulogne ?", True),
    ("Quel est le tarif pour livrer au 75015 samedi ?", True),
    ("Frais de livraison à Paris ?", True),
    ("Livrez-vous le dimanche ?", False),           # question de règle : RAG
    ("Combien coûte le plateau Tablée automnale ?", False),
])
def test_routage_question_de_frais_de_livraison(question, expected):
    from src.rag.chain import is_delivery_cost_question
    assert is_delivery_cost_question(question) is expected


def test_calendrier_donne_les_dates_avec_leur_nom():
    from datetime import date
    from src.rag.prompt import calendar
    text = calendar(date(2026, 10, 11))
    assert "- dimanche 2026-10-11 (aujourd'hui)" in text
    assert "- samedi 2026-10-17" in text


def test_budget_de_contexte_garde_le_premier_et_saute_les_trop_longs():
    from src.rag.retriever import fit_budget
    short, long_ = Document(page_content="a" * 100), Document(page_content="b" * 5000)
    assert fit_budget([long_, long_, short], max_chars=5200) == [long_, short]


@llm
def test_frais_de_livraison_calcules_par_l_outil(ask):
    from datetime import date, timedelta
    from decimal import Decimal
    from src.tools.delivery import day_multiplier
    saturday = date.today() + timedelta(days=(5 - date.today().weekday()) % 7 or 7)
    expected = (Decimal("39.70") * day_multiplier(saturday)[0]).quantize(Decimal("0.01"))
    result = ask("Combien coûte la livraison à Boulogne-Billancourt samedi ?")
    assert result.quotes and result.quotes[0].ok
    assert result.quotes[0].total_ht == expected
    assert f"{expected}".replace(".", ",") in format_answer(result)


@llm
def test_question_de_regle_de_livraison_sans_outil(ask):
    result = ask("Livrez-vous le dimanche ?")
    assert not result.quotes and result.sources


# ---------------------------------------------------------------- étape 5 : questions de suivi
@llm
def test_question_de_suivi_reformulee_puis_calculee_par_l_outil(ask):
    first = ask("Combien coûte la livraison au 92100 ?")
    follow_up = ask("Et si c'est un samedi ?", history=[("Combien coûte la livraison au 92100 ?", first.text)])
    assert "92100" in follow_up.question and "samedi" in follow_up.question.lower()
    assert follow_up.quotes and follow_up.quotes[0].ok
    assert "samedi" in follow_up.quotes[0].message


@llm
def test_sans_historique_pas_de_reformulation(ask):
    question = "Quels sont les délais pour un buffet ?"
    assert ask(question).question == question
