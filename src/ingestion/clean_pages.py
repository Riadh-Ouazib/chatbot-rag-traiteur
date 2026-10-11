"""
clean_pages.py : Nettoyer les pages scrapées avant le chunking.

Entrée  : data/raw/pages.jsonl          (sortie du scraper)
Sortie  : data/processed/pages_clean.jsonl

Ce que fait le script :
  1. Retire le "boilerplate" : les lignes courtes répétées sur beaucoup de pages
     (menu du site, "Mon panier"...) et le bruit d'interface (boutons, filtres, 10..100).
  2. Corrige l'échappement Markdown laissé par trafilatura ("\\_", "\\-").
  3. Retire les règles du site contredites par la carte PDF, plus récente (délais 17h -> 16h).
  4. Supprime les pages vides ou sans intérêt, et les doublons (même texte nettoyé).

Lancement (depuis la racine du projet) :
    python -m src.ingestion.clean_pages
"""

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from src.config import BRAND_ALIASES

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "pages.jsonl"
CLEAN_FILE = ROOT / "data" / "processed" / "pages_clean.jsonl"

# Une ligne courte présente sur au moins cette part des pages = menu du site
BOILERPLATE_MIN_SHARE = 0.10
BOILERPLATE_MAX_LEN = 60
MIN_TEXT_LEN = 50  # même seuil que le scraper

# Nom du traiteur tel qu'il apparaît sur le site (variantes dans .env), le plus long d'abord
BRAND = "|".join(re.escape(alias) for alias in sorted(BRAND_ALIASES, key=len, reverse=True))

# Bruit d'interface PrestaShop qui n'apporte rien à un client
UI_NOISE = re.compile(
    r"^(\d{1,3}|×|\(\d+\)|Ajouter au panier|Filtrer par|Filtres actifs.*|Réinitialiser les filtres"
    r"|Retour en haut|Affichage \d+-\d+ de \d+ article\(s\)|Je suis|une entreprise|un particulier"
    r"|J'organise|Pour|Prix|Sucré salé.*|Salé|Sucré|[-*•]+"
    rf"|{BRAND}|(?:(?:{BRAND})\s*)?vous recommande :?|.{{1,40}} (?:{BRAND}))$",
    re.IGNORECASE,
)

# Restes de vignettes produit au milieu d'une ligne
INLINE_NOISE = re.compile(r"\s*\b(Prix|Ajouter au panier)\b\s*")

# Pages sans valeur pour répondre aux clients.
# "nous-rejoindre" contient une ancienne FAQ des délais (17h), reprise à jour dans la carte p. 2.
EXCLUDED_PATHS = {"/traiteur/credits", "/traiteur/nous-rejoindre"}

# Règles du site qui contredisent la carte Automne-Hiver 2026/2027 (document le plus récent,
# qui fait foi) : délai de commande "la veille avant 17h" (carte : 16h) et majoration
# "dernière minute" de 15 € (carte : +100 %). Ces lignes sont retirées.
OUTDATED_RULES = re.compile(
    r"veille[^.]*\b(avant|jusqu.à) 17h|(avant|jusqu.à) 17h[^.]*veille"
    r"|entre 17h et 18h30|majorés de 15 euros",
    re.IGNORECASE,
)


def load_pages(path: Path) -> list[dict]:
    """Charge un fichier JSONL (une page par ligne)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unescape_markdown(text: str) -> str:
    """Retire les antislashs ajoutés par trafilatura devant _ - * #."""
    return re.sub(r"\\([_\-*#])", r"\1", text)


def find_boilerplate(pages: list[dict]) -> set[str]:
    """Repère les lignes courtes qui reviennent sur beaucoup de pages (menu, en-tête)."""
    counts: Counter[str] = Counter()
    for page in pages:
        text = unescape_markdown(page["text"])
        lines = {line.strip() for line in text.splitlines() if line.strip()}
        counts.update(lines)
    min_pages = max(3, int(len(pages) * BOILERPLATE_MIN_SHARE))
    return {
        line for line, n in counts.items()
        if n >= min_pages and len(line) <= BOILERPLATE_MAX_LEN
    }


def clean_text(text: str, boilerplate: set[str]) -> str:
    """Nettoie le texte d'une page : boilerplate, bruit d'interface, règles obsolètes, lignes vides."""
    kept = []
    for line in unescape_markdown(text).splitlines():
        stripped = line.strip()
        if stripped in boilerplate or UI_NOISE.match(stripped) or OUTDATED_RULES.search(stripped):
            continue
        kept.append(INLINE_NOISE.sub(" ", line).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def clean_title(title: str) -> str:
    """'Plateau cocktail | Nom &amp; Co' -> 'Plateau cocktail' (variantes du nom : BRAND_ALIASES)."""
    title = html.unescape(title)
    title = re.sub(rf"\s*[|-]?\s*(?:{BRAND})\s*$", "", title, flags=re.IGNORECASE)
    return title.strip() or "Accueil"


def clean_pages(pages: list[dict]) -> list[dict]:
    """Nettoie toutes les pages et retire les doublons (on garde l'URL la plus courte)."""
    boilerplate = find_boilerplate(pages)
    by_hash: dict[str, dict] = {}
    for page in pages:
        path = urlparse(page["url"]).path.rstrip("/") or "/"
        if path in EXCLUDED_PATHS:
            continue
        text = clean_text(page["text"], boilerplate)
        if len(text) < MIN_TEXT_LEN:
            continue
        record = {
            "url": page["url"],
            "title": clean_title(page["title"]),
            "description": page.get("description", ""),
            "text": text,
        }
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key not in by_hash or len(record["url"]) < len(by_hash[key]["url"]):
            by_hash[key] = record
    return list(by_hash.values())


def main() -> None:
    pages = load_pages(RAW_FILE)
    cleaned = clean_pages(pages)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_FILE, "w", encoding="utf-8") as f:
        for record in cleaned:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(pages)} pages brutes -> {len(cleaned)} pages nettoyées dans {CLEAN_FILE}")


if __name__ == "__main__":
    main()
