"""
delivery_grid.py : Extraire la grille des frais de livraison de la page "livraison".

Entrée  : data/processed/pages_clean.jsonl   (page /traiteur/livraison)
Sortie  : data/processed/delivery_grid.json

La grille reste dans data/processed/ (non commité) : ce sont les données du traiteur.
Les tarifs des zones 1 et 2 sont identiques dans la carte PDF (p. 31) ; un test le vérifie.

Lancement (depuis la racine du projet) :
    python -m src.ingestion.delivery_grid
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES_FILE = ROOT / "data" / "processed" / "pages_clean.jsonl"
GRID_FILE = ROOT / "data" / "processed" / "delivery_grid.json"
DELIVERY_URL = "https://www.exemple-traiteur.fr/traiteur/livraison"

# "**Zone 2**, les tarifs sont de **39.70€ HT** (43.67€ TTC) :\n78110, 78140, ..."
ZONE_BLOCK = re.compile(
    r"Zone (\d)\b.*?(\d+(?:[.,]\d+)?)\s*€\s*HT.*?:\s*((?:\d{5}[,.\s]*)+)", re.DOTALL)
POSTAL_CODE = re.compile(r"\b\d{5}\b")

# Zone 1 = "Paris" + une liste : on ajoute les codes postaux de Paris (75001-75020, 75116)
PARIS_CODES = [f"750{n:02d}" for n in range(1, 21)] + ["75116"]


def load_delivery_page(path: Path = PAGES_FILE) -> str:
    """Texte nettoyé de la page des conditions de livraison."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            if page["url"].rstrip("/") == DELIVERY_URL:
                return page["text"]
    raise FileNotFoundError(f"Page {DELIVERY_URL} absente de {path}")


def parse_grid(text: str) -> dict[str, dict]:
    """{"1": {"price_ht": 29.9, "postal_codes": [...]}, ...} à partir du texte de la page."""
    grid = {}
    for zone, price, codes in ZONE_BLOCK.findall(text):
        postal_codes = POSTAL_CODE.findall(codes)
        if zone == "1":
            postal_codes = PARIS_CODES + postal_codes
        grid[zone] = {"price_ht": float(price.replace(",", ".")), "postal_codes": postal_codes}
    return grid


def main() -> None:
    grid = parse_grid(load_delivery_page())
    with open(GRID_FILE, "w", encoding="utf-8") as f:
        json.dump({"source": DELIVERY_URL, "zones": grid}, f, ensure_ascii=False, indent=1)
    for zone, info in grid.items():
        print(f"Zone {zone} : {info['price_ht']:.2f} € HT, {len(info['postal_codes'])} codes postaux")


if __name__ == "__main__":
    main()
