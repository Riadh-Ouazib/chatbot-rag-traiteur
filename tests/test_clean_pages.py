"""Tests du nettoyage des pages scrapées."""

from urllib.parse import urlparse

import pytest

from src.config import BRAND_NAME
from src.ingestion.clean_pages import (
    CLEAN_FILE,
    clean_pages,
    clean_text,
    clean_title,
    find_boilerplate,
    load_pages,
)

MENU = "Mon panier\nCarte traiteur\nBuffet\n"


def make_page(url: str, body: str) -> dict:
    return {"url": url, "title": f"Titre | {BRAND_NAME}", "description": "", "text": MENU + body}


def test_boilerplate_detecte_le_menu_repete():
    pages = [make_page(f"https://x.com/p{i}", f"Contenu unique numéro {i}") for i in range(5)]
    boilerplate = find_boilerplate(pages)
    assert {"Mon panier", "Carte traiteur", "Buffet"} <= boilerplate
    assert "Contenu unique numéro 0" not in boilerplate


def test_clean_text_retire_bruit_et_garde_paragraphes():
    text = "Intro\n\n10\n11\n12\nshopping\\_cart\nAjouter au panier\n\nSuite du texte"
    cleaned = clean_text(text, boilerplate={"shopping_cart"})
    assert cleaned == "Intro\n\nSuite du texte"


def test_clean_text_retire_prix_et_bouton_dans_une_ligne():
    line = "Box tarte Prix 40,00 € HT 44,00 € TTC Ajouter au panier"
    assert clean_text(line, set()) == "Box tarte 40,00 € HT 44,00 € TTC"


def test_clean_text_retire_les_regles_contredites_par_la_carte():
    text = ("- Pour les buffets : sur commande 48h à l’avance (sauf dernière minute, la veille avant 17h).\n"
            "Vos plateaux peuvent être commandés jusqu'à 17h la veille.\n"
            "Commandes « dernière minute » après 17h, majorés de 15 euros.\n"
            "Livraison 7 jours sur 7, de 7h00 à 20h00.\n"
            "Tarifs doublés pour les commandes passées après 17h00 ou avant 10h00.")
    assert clean_text(text, set()) == ("Livraison 7 jours sur 7, de 7h00 à 20h00.\n"
                                       "Tarifs doublés pour les commandes passées après 17h00 ou avant 10h00.")


@pytest.mark.parametrize("raw, expected", [
    (f"Plateau cocktail  {BRAND_NAME}", "Plateau cocktail"),
    (f"Traiteur végétarien | {BRAND_NAME.replace('&', '&amp;')}", "Traiteur végétarien"),
    (f"Buffet traiteur - {BRAND_NAME}", "Buffet traiteur"),
    (BRAND_NAME, "Accueil"),
])
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_doublons_fusionnes_en_gardant_url_courte():
    body = "Un texte identique suffisamment long pour ne pas être filtré comme page vide."
    pages = [make_page("https://x.com/plateau-repas/", body), make_page("https://x.com/plateau-repas", body)]
    result = clean_pages(pages)
    assert [p["url"] for p in result] == ["https://x.com/plateau-repas"]


@pytest.mark.skipif(not CLEAN_FILE.exists(), reason="lancer d'abord clean_pages")
def test_donnees_reelles_sans_bruit_et_pages_cles_presentes():
    pages = load_pages(CLEAN_FILE)
    for page in pages:
        assert "Mon panier" not in page["text"]
        assert "shopping_cart" not in page["text"]
        assert "avant 17h" not in page["text"], page["url"]  # la carte (16h) fait foi
    paths = {urlparse(p["url"]).path.rstrip("/") for p in pages}
    for path in ("/traiteur/livraison", "/buffet", "/cocktail", "/plateau-repas",
                 "/traiteur/vegetarien", "/traiteur/conditions-generales-de-vente"):
        assert path in paths, path
