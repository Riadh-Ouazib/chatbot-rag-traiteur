"""
common.py : Fonctions partagées par les scripts d'évaluation.
"""

import json
import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pour "python eval/xxx.py" lancé depuis la racine
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_jsonl(path: Path) -> list[dict]:
    """Charge un fichier JSONL (un objet par ligne)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Écrit un fichier JSONL (UTF-8, accents lisibles)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def answers_file(strategy: str) -> Path:
    """Fichier des réponses du bot pour une stratégie (non commité : contenu du traiteur)."""
    return RESULTS_DIR / f"answers_{strategy}.jsonl"


def source_key(doc: Document) -> str:
    """Identifiant de source comparable entre stratégies : 'carte:13' (page du PDF) ou 'site:URL'.

    On compare par page PDF et non par citation : en stratégie "recursive", un chunk de la
    carte reçoit la page imprimée du premier produit de la double page, pas forcément la bonne.
    """
    if doc.metadata["source_type"] == "carte":
        return f"carte:{doc.metadata['pdf_page']}"
    return f"site:{doc.metadata['source']}"
