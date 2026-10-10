"""Tests des scripts d'évaluation (étape 6) : règles métier, moyennes, jeu de questions."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from common import QUESTIONS_FILE, load_jsonl  # noqa: E402
from score import check_rules, fmt, mean  # noqa: E402

RECORD = {"rendered": "Commandez la veille avant 16h [1]. Contact : 01 00 00 00 00",
          "notices": ["allergens", "warning"],
          "quotes": [{"ok": True, "total_ht": "59.55"}]}


def test_regles_respectees():
    checks = {"must_include": ["16H"], "must_not_include": ["17h"], "contact": True,
              "official_allergens": True, "amount_ht": "59.55"}
    assert all(check_rules(checks, RECORD).values())


def test_regles_non_respectees_sont_nommees():
    checks = {"must_include": ["48h"], "must_not_include": ["16h"], "amount_ht": "29.90", "no_price": True}
    failed = [name for name, ok in check_rules(checks, RECORD).items() if not ok]
    assert failed == ["contient « 48h »", "sans « 16h »", "montant 29.90 € HT", "aucun prix inventé"]


def test_moyenne_ignore_les_echecs_du_juge():
    assert mean([1.0, None, math.nan, 0.0]) == 0.5
    assert mean([None]) is None


def test_format_francais():
    assert fmt(0.8333) == "0,83"
    assert fmt(0.8333, percent=True) == "83 %"
    assert fmt(None) == "—"


def test_jeu_de_questions_complet_et_coherent():
    questions = load_jsonl(QUESTIONS_FILE)
    assert len(questions) == 30
    assert len({q["id"] for q in questions}) == 30
    for q in questions:
        assert q["question"] and q["reference"] and q["checks"], q["id"]
        assert all(s.startswith(("carte:", "site:https://")) for s in q["expected_sources"]), q["id"]
        if q["category"] == "livraison":
            assert not q["expected_sources"]  # passent par l'outil, pas par la recherche


@pytest.mark.skipif(not Path("data/processed/carte.jsonl").exists(), reason="lancer d'abord load_pdf")
def test_pages_attendues_existent_dans_la_carte():
    pages = {r["pdf_page"] for r in load_jsonl(Path("data/processed/carte.jsonl"))}
    expected = {int(s.split(":")[1]) for q in load_jsonl(QUESTIONS_FILE)
                for s in q["expected_sources"] if s.startswith("carte:")}
    assert expected <= pages
