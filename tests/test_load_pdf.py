"""Tests de l'extraction de la carte PDF (étape 2b).

Les allergènes de référence ont été relus dans la carte : si un test échoue,
c'est que l'extraction a mélangé deux produits (risque pour la santé des clients).
"""

import pytest

from src.ingestion.load_pdf import (
    NO_ALLERGENS,
    RAW_DIR,
    Line,
    extract_carte,
    find_pdf,
    fix_text,
    group_titles,
    owner_title,
)


def title(text: str, x: float, y: float) -> Line:
    return Line(text, x, y, "DINCondensed-Bold", 22.0)


def body(text: str, x: float, y: float) -> Line:
    return Line(text, x, y, "FuturaT-Book", 10.0)


# ---------------------------------------------------------------- tests unitaires
@pytest.mark.parametrize("raw, expected", [
    ("30,50 $ HT", "30,50 € HT"),
    ("33,55 e TTC", "33,55 € TTC"),
    ("facturée 50,00\x1de HT", "facturée 50,00 € HT"),
    ("Plateau de fromages afﬁnés", "Plateau de fromages affinés"),
    ("une entrée et le dessert", "une entrée et le dessert"),  # 'e' normal non modifié
])
def test_fix_text(raw, expected):
    assert fix_text(raw) == expected


def test_titres_sur_plusieurs_lignes_fusionnes():
    lines = [title("BROCHETTES", 330, 263), title("DE FRUITS FRAIS", 330, 286),
             title("COFFRET", 330, 575)]
    assert [t.text for t in group_titles(lines)] == ["BROCHETTES DE FRUITS FRAIS", "COFFRET"]


def test_ligne_rattachee_au_titre_de_sa_colonne_malgre_decalage():
    titles = [title("EN-TÊTE", 50, 38), title("WRAPS", 49, 156), title("AUTRE COLONNE", 305, 100)]
    assert owner_title(body("• Wrap à la dinde", 49, 200), titles).text == "WRAPS"
    assert owner_title(body("35,00 € HT", 213, 157), titles).text == "WRAPS"
    assert owner_title(body("texte d'intro", 10, 300), titles) is None


# ---------------------------------------------------------------- tests sur la vraie carte
has_pdf = pytest.mark.skipif(not list(RAW_DIR.glob("Carte_*.pdf")), reason="carte PDF absente")


@pytest.fixture(scope="module")
def products() -> dict[str, dict]:
    records = extract_carte(find_pdf())
    return {r["name"]: r for r in records if r["kind"] == "product"}


REFERENCE_ALLERGENS = {
    "MIGNARDISES": "fruits à coque, gluten, lactose, lupin, œufs, soja, sulfites",
    "MINI MACARONS": "arachides, fruits à coque, gluten, lactose, lupin, œufs, soja, sésame, sulfites",
    "COFFRET DE CHOUX": "lactose, gluten, œufs, fruits à coque, soja, sulfites",
    "TABLÉE AUTOMNALE": "gluten, lactose, œufs, soja, fruits à coque, poissons, moutarde, sulfites, sésame, céleri",
    "RENDEZ-VOUS AU BISTROT": "lactose, poissons, soja, gluten, œufs, fruits à coque, sésame, céleri, "
                              "sulfites, moutarde, arachides",
    "PLATEAU DE WRAPS": "poissons, gluten, sulfites, lactose, soja, sésame",
}


@has_pdf
@pytest.mark.parametrize("name, allergens", REFERENCE_ALLERGENS.items())
def test_allergenes_de_reference(products, name, allergens):
    assert products[name]["allergens"] == f"Allergènes : {allergens}"


@has_pdf
def test_produit_sans_allergenes_renvoie_vers_equipe(products):
    brochettes = products["BROCHETTES DE FRUITS FRAIS DE SAISON"]
    assert brochettes["allergens"] is None
    assert NO_ALLERGENS in brochettes["text"]
    assert "gluten" not in brochettes["text"]  # ne doit pas hériter des choux voisins


@has_pdf
def test_chaque_produit_a_une_seule_ligne_allergenes(products):
    for name, product in products.items():
        assert product["text"].count("Allergènes") == 1, name


@has_pdf
def test_chaque_produit_a_un_prix_et_une_page(products):
    """Prix sur la fiche, ou 'PRIX UNIQUE' de la gamme dans la rubrique (lunch box, bowls)."""
    for name, product in products.items():
        assert "€ HT" in product["text"] or "€ HT" in product["section"], name
        assert product["pdf_page"] >= 1, name


@has_pdf
def test_prix_unique_de_gamme_sur_tous_les_produits(products):
    for name in ("DINDE À LA SAUGE", "REBLOCHON", "RILLETTES DE THON", "SAUMON"):
        assert "PRIX UNIQUE 19,50 € HT par personne" in products[name]["section"], name
        assert "19,50" not in products[name]["text"], name
    assert "PRIX UNIQUE 20,90 € HT par personne" in products["NOUILLES AU BŒUF"]["section"]


@has_pdf
def test_prix_a_droite_du_titre_bien_attribue(products):
    """Le prix de la tarte est à droite de deux titres : il doit aller au plus proche."""
    assert "40,00 € HT" in products["TARTE FEUILLETÉE"]["text"]
    assert "40,00 € HT" not in products["BOX PAIN PITA ET HOUMOUS"]["text"]
