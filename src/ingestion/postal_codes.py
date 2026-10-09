"""
postal_codes.py — Étape 4a : base des communes d'Île-de-France et de leurs codes postaux.

Source : base officielle des codes postaux de La Poste (open data, data.gouv.fr).
Sortie : data/processed/communes_idf.json   {"BOULOGNE BILLANCOURT": ["92100"], ...}

Sert à l'outil de livraison quand le client donne une ville ("à Boulogne") au lieu d'un code postal.

Lancement (depuis la racine du projet) :
    python -m src.ingestion.postal_codes
"""

import csv
import io
import json
import re
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "laposte_hexasmal.csv"
COMMUNES_FILE = ROOT / "data" / "processed" / "communes_idf.json"
SOURCE_URL = "https://data.laposte.fr/data-fair/api/v1/datasets/laposte-hexasmal/raw"
IDF_DEPARTMENTS = ("75", "77", "78", "91", "92", "93", "94", "95")


def normalize_name(name: str) -> str:
    """'Saint-Maur-des-Fossés' -> 'ST MAUR DES FOSSES' (format de la base La Poste)."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    name = re.sub(r"[^A-Z0-9]+", " ", name).strip()
    name = re.sub(r"\bSAINTE\b", "STE", name)
    name = re.sub(r"\bSAINT\b", "ST", name)
    # "Paris 1er", "PARIS 01" (format La Poste) -> "PARIS 1"
    return re.sub(r"\bPARIS 0?(\d+)(?:E|EME|ER)?\b", r"PARIS \1", name)


def download_csv() -> str:
    """Télécharge la base La Poste une seule fois (gardée dans data/raw/)."""
    if not RAW_FILE.exists():
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        RAW_FILE.write_bytes(response.content)
    return RAW_FILE.read_bytes().decode("latin-1")  # fichier encodé en ISO-8859-1


def build_communes(csv_text: str) -> dict[str, list[str]]:
    """Nom de commune (et lieu-dit de la ligne 5) -> codes postaux, pour l'Île-de-France."""
    communes: dict[str, list[str]] = {}
    for row in csv.reader(io.StringIO(csv_text), delimiter=";"):
        if len(row) < 3 or not row[2].startswith(IDF_DEPARTMENTS):
            continue
        postal_code = row[2]
        names = [row[1]] + ([row[4]] if len(row) > 4 and row[4] else [])
        for name in names:
            codes = communes.setdefault(normalize_name(name), [])
            if postal_code not in codes:
                codes.append(postal_code)
    return communes


def main() -> None:
    communes = build_communes(download_csv())
    with open(COMMUNES_FILE, "w", encoding="utf-8") as f:
        json.dump(communes, f, ensure_ascii=False, indent=0)
    print(f"{len(communes)} communes / lieux-dits d'Île-de-France dans {COMMUNES_FILE}")


if __name__ == "__main__":
    main()
