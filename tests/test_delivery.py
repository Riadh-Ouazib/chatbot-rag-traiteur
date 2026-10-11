"""Tests de l'outil de frais de livraison.

Les tests du calcul utilisent une petite grille fictive (fixture) : ils ne dépendent pas
des données du traiteur, qui ne sont pas commitées.
"""

import json
from datetime import date
from decimal import Decimal

import pytest

from src.config import TEAM_PHONE

from src.ingestion import delivery_grid
from src.ingestion.postal_codes import build_communes, normalize_name
from src.tools import delivery
from src.tools.delivery import calculer_frais_livraison, day_multiplier

GRID = {"zones": {
    "1": {"price_ht": 29.9, "postal_codes": ["75001", "75015", "92100", "95100"]},
    "2": {"price_ht": 39.7, "postal_codes": ["92200", "95100"]},
    "3": {"price_ht": 55.0, "postal_codes": ["94210"]},
}}
COMMUNES = {"BOULOGNE BILLANCOURT": ["92100"], "PARIS 1": ["75001"], "PARIS 15": ["75015"],
            "NEUILLY SUR SEINE": ["92200"], "ST MAUR DES FOSSES": ["94100", "94210"]}

TUESDAY, SATURDAY, SUNDAY = "2026-10-13", "2026-10-17", "2026-10-18"
CHRISTMAS, ALL_SAINTS_SUNDAY = "2026-12-25", "2026-11-01"  # 1er novembre 2026 = dimanche


@pytest.fixture(autouse=True)
def fake_data(tmp_path, monkeypatch):
    """Remplace les fichiers de data/processed par une grille et des communes fictives."""
    grid_file, communes_file = tmp_path / "grid.json", tmp_path / "communes.json"
    grid_file.write_text(json.dumps(GRID), encoding="utf-8")
    communes_file.write_text(json.dumps(COMMUNES), encoding="utf-8")
    monkeypatch.setattr(delivery, "GRID_FILE", grid_file)
    monkeypatch.setattr(delivery, "COMMUNES_FILE", communes_file)
    for loader in (delivery.load_zones, delivery.load_prices, delivery.load_communes):
        loader.cache_clear()
    yield
    for loader in (delivery.load_zones, delivery.load_prices, delivery.load_communes):
        loader.cache_clear()


# ---------------------------------------------------------------- calcul
@pytest.mark.parametrize("place, day, expected_ht, expected_ttc", [
    ("75015", TUESDAY, "29.90", "32.89"),       # semaine : tarif de base, TVA 10 %
    ("92200", SATURDAY, "59.55", "65.51"),      # samedi +50 %, arrondi au centime
    ("75001", SUNDAY, "59.80", "65.78"),        # dimanche ×2
    ("75001", CHRISTMAS, "59.80", "65.78"),     # jour férié ×2
    ("75001", ALL_SAINTS_SUNDAY, "119.60", "131.56"),  # dimanche férié ×4
    ("94210", None, "55.00", "60.50"),          # sans date : tarif du lundi au vendredi
])
def test_calcul(place, day, expected_ht, expected_ttc):
    quote = calculer_frais_livraison(place, day)
    assert quote.ok, quote.message
    assert (quote.total_ht, quote.total_ttc) == (Decimal(expected_ht), Decimal(expected_ttc))


def test_derniere_minute_double_le_tarif_en_semaine():
    assert calculer_frais_livraison("75001", TUESDAY, derniere_minute=True).total_ht == Decimal("59.80")


def test_derniere_minute_le_week_end_renvoie_vers_l_equipe():
    quote = calculer_frais_livraison("75001", SATURDAY, derniere_minute=True)
    assert not quote.ok and TEAM_PHONE in quote.message


def test_95100_present_dans_deux_zones_prend_la_zone_1():
    assert calculer_frais_livraison("95100").total_ht == Decimal("29.90")


# ---------------------------------------------------------------- lieux
@pytest.mark.parametrize("place, expected_ht", [
    ("Boulogne-Billancourt", "29.90"),
    ("boulogne", "29.90"),           # nom partiel, une seule commune correspond
    ("Neuilly-sur-Seine", "39.70"),
    ("Paris 15e", "29.90"),
    ("Paris 1er", "29.90"),
    ("Paris", "29.90"),              # tous les arrondissements sont dans la même zone
])
def test_villes(place, expected_ht):
    quote = calculer_frais_livraison(place)
    assert quote.ok, quote.message
    assert quote.total_ht == Decimal(expected_ht)


def test_ville_sur_deux_zones_demande_le_code_postal():
    quote = calculer_frais_livraison("Saint-Maur-des-Fossés")
    assert not quote.ok
    assert "94100" in quote.message and "94210" in quote.message


@pytest.mark.parametrize("place", ["9210", "abc123", "Lyon", "13001"])
def test_lieu_invalide_ou_hors_zone(place):
    quote = calculer_frais_livraison(place)
    assert not quote.ok and TEAM_PHONE in quote.message


def test_date_invalide():
    assert not calculer_frais_livraison("75001", "samedi").ok


def test_day_multiplier_nom_du_jour_ferie_en_francais():
    assert day_multiplier(date(2026, 12, 25)) == (Decimal("2"), "jour férié (Noël) : ×2")


# ---------------------------------------------------------------- données
@pytest.mark.parametrize("raw, expected", [
    ("Saint-Maur-des-Fossés", "ST MAUR DES FOSSES"),
    ("Paris 15ème", "PARIS 15"),
    ("PARIS 01", "PARIS 1"),
    ("Sainte-Geneviève-des-Bois", "STE GENEVIEVE DES BOIS"),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_build_communes_filtre_l_ile_de_france():
    csv_text = ("#Code_commune_INSEE;Nom_de_la_commune;Code_postal;Libellé;Ligne_5\n"
                "92012;BOULOGNE BILLANCOURT;92100;BOULOGNE BILLANCOURT;\n"
                "62160;BOULOGNE SUR MER;62200;BOULOGNE SUR MER;\n"
                "94068;ST MAUR DES FOSSES;94210;ST MAUR DES FOSSES;LA VARENNE ST HILAIRE\n")
    communes = build_communes(csv_text)
    assert communes["BOULOGNE BILLANCOURT"] == ["92100"]
    assert "BOULOGNE SUR MER" not in communes
    assert communes["LA VARENNE ST HILAIRE"] == ["94210"]


def test_parse_grid():
    text = ("- Pour les livraisons sur la **Zone 1 Paris** et les codes postaux suivants, le tarif est "
            "de **29.90€ HT** (32.89€ TTC) : 92000, 92110.\n"
            "- Pour les codes postaux de la **Zone 4**, les tarifs sont de **80€HT** (88€ TTC) :\n77000, 77090.")
    grid = delivery_grid.parse_grid(text)
    assert grid["1"]["price_ht"] == 29.9
    assert grid["1"]["postal_codes"][-2:] == ["92000", "92110"] and "75015" in grid["1"]["postal_codes"]
    assert grid["4"] == {"price_ht": 80.0, "postal_codes": ["77000", "77090"]}


@pytest.mark.skipif(not delivery_grid.PAGES_FILE.exists(), reason="lancer d'abord clean_pages")
def test_grille_reelle_identique_a_la_carte():
    """Zones 1 et 2 du site = tarifs de la carte PDF p. 31 (qui fait foi)."""
    from src.ingestion.chunking import CARTE_FILE, load_jsonl
    grid = delivery_grid.parse_grid(delivery_grid.load_delivery_page())
    assert sorted(grid) == ["1", "2", "3", "4"]
    page_31 = " ".join(r["text"] for r in load_jsonl(CARTE_FILE) if r["pdf_page"] == 31)
    for zone, price in (("1", "29,90 € HT"), ("2", "39,70 € HT")):
        assert f"{grid[zone]['price_ht']:.2f}".replace(".", ",") + " € HT" == price
        assert price in page_31
