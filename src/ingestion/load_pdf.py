"""
load_pdf.py : Extraire la carte PDF produit par produit.

Entrée  : data/raw/Carte_*.pdf          (téléchargée à la main, jamais par script)
Sortie  : data/processed/carte.jsonl     (1 ligne = 1 produit ou 1 bloc de texte libre)

Pourquoi c'est délicat :
  - Chaque page du PDF est une double page imprimée (gauche / droite).
  - La carte est en colonnes : une extraction "ligne à ligne" mélange les produits,
    et pourrait attribuer les allergènes d'un plat à son voisin.

La méthode : on lit la position (x, y) et la police de chaque ligne avec PyMuPDF.
  - Un titre de produit = police DINCondensed-Bold (grande taille).
  - Chaque ligne est rattachée au titre de sa colonne situé juste au-dessus d'elle.
  - Les lignes sans titre (en-têtes, CGV, textes de présentation) forment des blocs "texte".

Lancement (depuis la racine du projet) :
    python -m src.ingestion.load_pdf
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CARTE_FILE = ROOT / "data" / "processed" / "carte.jsonl"

TITLE_FONT = "DINCondensed-Bold"
TITLE_SIZE = 22          # titres de produit ; plus grand = rubrique, plus petit = décoratif
SECTION_MIN_SIZE = 30    # grands titres de rubrique ("COLLECTION LUNCH")
HEADER_MAX_Y = 45        # en-tête de rubrique ("COCKTAILS SUCRÉS ...") en haut de page
FOOTER_MIN_Y = 700       # numéros de pages imprimés en bas
X_TOLERANCE = 5          # petits décalages d'alignement dans la mise en page
SUBCOLUMN_WIDTH = 100   # largeur d'une sous-colonne pour l'ordre de lecture
SAME_ROW_MAX_DY = 20     # prix HT/TTC placés à droite du titre, sur sa ligne
BROKEN_LIGATURES = {"昀椀": "fi", "昀氀": "fl", "昀": "f"}  # "con昀椀tes" -> "confites"
MAX_PRODUCT_LEN = 1500   # la plus longue fiche produit fait ~1 000 caractères
PRICE_LINE = re.compile(r"^\d+,\d\d € HT")  # ligne de prix seule, propre aux fiches produit
NO_ALLERGENS ="Allergènes : non indiqués dans la carte, contacter l'équipe."


@dataclass
class Line:
    """Une ligne de texte de la carte avec sa position et sa police."""
    text: str
    x: float
    y: float
    font: str
    size: float

    @property
    def is_title(self) -> bool:
        return TITLE_FONT in self.font and abs(self.size - TITLE_SIZE) <= 1


def fix_currency(text: str) -> str:
    """La police de la carte fait lire '€' comme '$' ou 'e' : '30,50 $ HT' -> '30,50 € HT'."""
    return re.sub(r"(\d+(?:,\d\d)?) ?[$e](?=\s?(?:HT|TTC|par)\b)", r"\1 €", text)


def fix_text(text: str) -> str:
    """Corrige l'extraction : ligatures (ﬁ -> fi), symbole €, espaces multiples."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)  # caractères de contrôle cachés dans le PDF
    for broken, fixed in BROKEN_LIGATURES.items():   # ligatures mal encodées par la police
        text = text.replace(broken, fixed)
    return re.sub(r"[ \t]+", " ", fix_currency(text)).strip()


def find_pdf() -> Path:
    """Renvoie la carte PDF présente dans data/raw/."""
    pdfs = sorted(RAW_DIR.glob("Carte_*.pdf"))
    if not pdfs:
        raise FileNotFoundError("Carte PDF absente de data/raw/ (à télécharger à la main).")
    return pdfs[-1]


def page_lines(page: pymupdf.Page) -> list[Line]:
    """Toutes les lignes non vides d'une page, avec position et police de leur 1er segment."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = fix_text("".join(span["text"] for span in line["spans"]))
            if len(text) > 1:  # les lettres isolées viennent de textes écrits en cercle
                span = line["spans"][0]
                x0, y0 = line["bbox"][:2]
                lines.append(Line(text, x0, y0, span["font"], span["size"]))
    return lines


def split_halves(page: pymupdf.Page, lines: list[Line]) -> list[list[Line]]:
    """Coupe une double page (paysage) en page gauche et page droite."""
    if page.rect.width <= page.rect.height:
        return [lines]
    middle = page.rect.width / 2
    return [[l for l in lines if l.x < middle], [l for l in lines if l.x >= middle]]


def group_titles(lines: list[Line]) -> list[Line]:
    """Fusionne les titres sur plusieurs lignes ('BROCHETTES' / 'DE FRUITS FRAIS')."""
    titles: list[Line] = []
    last_y: dict[int, float] = {}  # y de la dernière ligne de chaque titre
    for line in sorted((l for l in lines if l.is_title), key=lambda l: l.y):
        prev = next((t for t in titles if abs(t.x - line.x) <= X_TOLERANCE
                     and 0 < line.y - last_y[id(t)] <= line.size * 1.5), None)
        if prev:
            prev.text = f"{prev.text} {line.text}"
            last_y[id(prev)] = line.y
        else:
            title = Line(line.text, line.x, line.y, line.font, line.size)
            titles.append(title)
            last_y[id(title)] = line.y
    return titles


def owner_title(line: Line, titles: list[Line]) -> Line | None:
    """Titre auquel appartient une ligne.

    1. Le bloc prix est à droite de son titre, sur la même ligne : titre à gauche tout proche en y.
    2. Sinon : titre de la même colonne (le plus à droite à gauche de la ligne), juste au-dessus.
    """
    candidates = [t for t in titles if t.x <= line.x + X_TOLERANCE and t.y <= line.y + X_TOLERANCE]
    if not candidates:
        return None
    same_row = [t for t in candidates if line.y - t.y <= SAME_ROW_MAX_DY]
    if same_row:
        return max(same_row, key=lambda t: t.x)
    column_x = max(t.x for t in candidates)
    in_column = [t for t in candidates if column_x - t.x <= X_TOLERANCE]
    return max(in_column, key=lambda t: t.y)


def printed_page_number(lines: list[Line]) -> str | None:
    """Numéro de page imprimé en bas de la demi-page ('24')."""
    for line in lines:
        if line.y >= FOOTER_MIN_Y and line.text.isdigit():
            return line.text
    return None


def join_lines(lines: list[Line]) -> str:
    """Remet les lignes dans l'ordre de lecture et recolle les phrases coupées.

    On lit sous-colonne par sous-colonne (tranches de 100 px) : d'abord la description,
    puis le bloc prix placé à droite du titre.
    """
    if not lines:
        return ""
    left = min(l.x for l in lines)
    ordered = sorted(lines, key=lambda l: (int((l.x - left) // SUBCOLUMN_WIDTH), l.y))
    text = "\n".join(l.text for l in ordered)
    text = re.sub(r"\n(?=[a-zà-ÿ(])", " ", text)  # une ligne qui commence en minuscule continue la précédente
    return fix_currency(text)  # un prix peut être coupé sur deux lignes


def extract_allergens(text: str) -> str | None:
    """Renvoie la phrase 'Allergènes : ...' si elle existe."""
    match = re.search(r"Allergènes\s*:[^\n]*", text)
    return match.group(0).rstrip(".") if match else None


def is_product(lines: list[Line]) -> bool:
    """Un vrai produit a une ligne de prix seule ou une ligne allergènes, et reste court.

    Sinon c'est un titre décoratif, une rubrique ou un long texte (les CGV ont un tableau de prix).
    """
    if sum(len(l.text) for l in lines) > MAX_PRODUCT_LEN:
        return False
    return any(PRICE_LINE.match(l.text) or l.text.startswith("Allergènes") for l in lines)


def split_range_lines(title: Line, lines: list[Line]) -> tuple[list[Line], list[Line]]:
    """Sépare les lignes du produit de celles qui concernent toute la gamme.

    Une ligne placée après les allergènes ET dans une autre sous-colonne que le titre
    est un prix de gamme (ex. '19,50 € HT par personne' pour toutes les lunch box).
    """
    allergen_ys = [l.y for l in lines if l.text.startswith("Allergènes")]
    if not allergen_ys:
        return lines, []
    end_y = min(allergen_ys)
    own, outside = [], []
    for line in lines:
        is_outside = line.y > end_y and line.x - title.x >= SUBCOLUMN_WIDTH
        (outside if is_outside else own).append(line)
    return own, outside


def parse_half(lines: list[Line], pdf_page: int) -> list[dict]:
    """Découpe une demi-page en produits (un par titre) et en texte libre."""
    page_label = printed_page_number(lines)
    body = [l for l in lines if not (l.y >= FOOTER_MIN_Y and l.text.isdigit())]
    header_lines = [l for l in body if not l.is_title
                    and (l.y < HEADER_MAX_Y or l.size >= SECTION_MIN_SIZE)]
    body = [l for l in body if l not in header_lines]
    titles = group_titles(body)

    members: dict[int, list[Line]] = {id(t): [] for t in titles}
    free_lines: list[Line] = []
    for line in body:
        if line.is_title:
            continue
        owner = owner_title(line, titles)
        (members[id(owner)] if owner else free_lines).append(line)

    range_lines: list[Line] = []
    for title in titles:
        members[id(title)], outside = split_range_lines(title, members[id(title)])
        range_lines += outside

    section = join_lines(header_lines + range_lines).replace("\n", " ")
    unique_price = re.search(r"PRIX UNIQUE(\n.*€.*){1,2}", join_lines(free_lines))
    if unique_price and unique_price.group(0) not in section:
        section += " " + unique_price.group(0).replace("\n", " ")
    base = {"source": "carte", "pdf_page": pdf_page, "printed_page": page_label, "section": section}
    records = []
    for title in titles:
        if not is_product(members[id(title)]):
            free_lines += [title, *members[id(title)]]
            continue
        text = join_lines(members[id(title)])
        allergens = extract_allergens(text)
        if allergens is None:
            text += "\n" + NO_ALLERGENS
        records.append({**base, "kind": "product", "name": title.text,
                        "allergens": allergens, "text": text})
    free = join_lines(free_lines)
    if free:
        records.append({**base, "kind": "text", "name": None, "allergens": None, "text": free})
    return records


def extract_carte(pdf_path: Path) -> list[dict]:
    """Extrait tous les produits et blocs de texte de la carte."""
    records = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            lines = page_lines(page)
            for half in split_halves(page, lines):
                records += parse_half(half, page_index)
    return records


def main() -> None:
    pdf_path = find_pdf()
    records = extract_carte(pdf_path)
    CARTE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CARTE_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    products = sum(r["kind"] == "product" for r in records)
    print(f"{pdf_path.name} : {products} produits + {len(records) - products} blocs de texte -> {CARTE_FILE}")


if __name__ == "__main__":
    main()
