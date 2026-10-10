"""Tests de la lecture des dates de livraison (bug trouvé par l'évaluation de l'étape 6)."""

from datetime import date

import pytest

from src.tools.dates import parse_delivery_date

TODAY = date(2026, 10, 11)  # un dimanche


@pytest.mark.parametrize("text, expected", [
    ("le samedi 14 novembre 2026", date(2026, 11, 14)),
    ("le mardi 17 novembre 2026", date(2026, 11, 17)),   # le cas qui faisait échouer le LLM
    ("le dimanche 1er novembre 2026", date(2026, 11, 1)),
    ("le 3 janvier", date(2027, 1, 3)),                   # sans année : prochaine occurrence
    ("le 25 décembre", date(2026, 12, 25)),
    ("le 25/12", date(2026, 12, 25)),
    ("le 14/11/26", date(2026, 11, 14)),
    ("samedi", date(2026, 10, 17)),
    ("dimanche", date(2026, 10, 18)),                     # aujourd'hui dimanche -> le suivant
    ("demain", date(2026, 10, 12)),
    ("après-demain", date(2026, 10, 13)),
    ("aujourd'hui", TODAY),
])
def test_dates_reconnues(text, expected):
    assert parse_delivery_date(f"Livraison au 75011 {text} ?", TODAY) == expected


@pytest.mark.parametrize("text", [
    "Combien coûte la livraison au 92100 ?",   # un code postal n'est pas une date
    "Livraison dans le 11e arrondissement ?",
    "le 31 février",                           # date impossible
])
def test_pas_de_date(text):
    assert parse_delivery_date(text, TODAY) is None
