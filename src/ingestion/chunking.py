"""
chunking.py — Étape 2c : découper pages et carte en chunks, avec leurs métadonnées.

Deux stratégies, comparées à l'étape 6 (RAGAS) :
  - "recursive" : découpage générique tous les ~800 caractères (paragraphes, puis phrases).
                  La carte est traitée page par page, comme un simple texte.
  - "structure" : on suit la structure du contenu.
                  Site  : un chunk par section (titres Markdown, articles des CGV).
                  Carte : un chunk par produit (nom + composition + allergènes + prix).

Chaque chunk commence par un petit en-tête de contexte ("Page : Conditions de livraison > Paiement")
pour que l'embedding sache de quoi parle un morceau isolé.
"""

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
PAGES_FILE = ROOT / "data" / "processed" / "pages_clean.jsonl"
CARTE_FILE = ROOT / "data" / "processed" / "carte.jsonl"

CHUNK_SIZE = 800          # caractères (~200 tokens), largement sous la limite du modèle (512 tokens)
CHUNK_OVERLAP = 100
MAX_SECTION_SIZE = 1500   # au-delà, une section est redécoupée
CARTE_NAME = "Carte Automne-Hiver 2026/2027"
STRATEGIES = ("recursive", "structure")

# Début de section : titre Markdown ou article des CGV ("Article 3 DELAIS ...")
SECTION_START = re.compile(r"^(#{1,3} .+|\**Article \d+\b.*)$", re.MULTILINE)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------- chargement et métadonnées
def load_jsonl(path: Path) -> list[dict]:
    """Charge un fichier JSONL produit par clean_pages ou load_pdf."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def carte_citation(record: dict) -> str:
    """Référence lisible : 'Carte Automne-Hiver 2026/2027, p. 24' (page imprimée si connue)."""
    if record.get("printed_page"):
        return f"{CARTE_NAME}, p. {record['printed_page']}"
    return f"{CARTE_NAME}, page PDF {record['pdf_page']}"


def page_metadata(page: dict) -> dict:
    """Métadonnées d'une page du site (Chroma n'accepte que str/int/float/bool, pas None)."""
    return {"source_type": "site", "source": page["url"], "title": page["title"],
            "citation": f"{page['title']} ({page['url']})"}


def carte_metadata(record: dict) -> dict:
    """Métadonnées d'un élément de la carte PDF."""
    meta = {"source_type": "carte", "source": carte_citation(record), "citation": carte_citation(record),
            "title": record["name"] or record["section"] or CARTE_NAME, "pdf_page": record["pdf_page"]}
    if record["kind"] == "product":
        meta["product"] = record["name"]
        meta["has_allergens"] = record["allergens"] is not None
    return meta


def make_document(text: str, metadata: dict, strategy: str) -> Document:
    """Crée un Document avec un identifiant stable (même texte -> même id : réindexation sans doublon)."""
    chunk_id = hashlib.sha1(f"{strategy}|{metadata['source']}|{text}".encode("utf-8")).hexdigest()
    return Document(page_content=text, metadata={**metadata, "strategy": strategy, "chunk_id": chunk_id})


# ---------------------------------------------------------------- stratégie A : récursive
def recursive_chunks(pages: list[dict], carte: list[dict]) -> list[Document]:
    """Découpage générique de taille fixe, sans tenir compte de la structure."""
    docs = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            docs.append(make_document(f"Page : {page['title']}\n{piece}", page_metadata(page), "recursive"))

    # La carte page par page, comme un texte brut : les produits voisins peuvent se mélanger
    by_page: dict[int, list[dict]] = {}
    for record in carte:
        by_page.setdefault(record["pdf_page"], []).append(record)
    for records in by_page.values():
        text = "\n\n".join(filter(None, (f"{r['name'] or ''}\n{r['text']}".strip() for r in records)))
        meta = carte_metadata({**records[0], "kind": "text", "name": None})
        for piece in splitter.split_text(text):
            docs.append(make_document(f"{CARTE_NAME}\n{piece}", meta, "recursive"))
    return docs


# ---------------------------------------------------------------- stratégie B : structure
def split_sections(text: str) -> list[tuple[str, str]]:
    """Coupe un texte aux titres Markdown / articles : [(titre_de_section, contenu), ...]."""
    starts = [m.start() for m in SECTION_START.finditer(text)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    sections = []
    for begin, end in zip(starts, starts[1:] + [len(text)]):
        block = text[begin:end].strip()
        first_line = block.splitlines()[0] if block else ""
        heading = re.sub(r"^[#*\s]+|\*+$", "", first_line) if SECTION_START.match(first_line) else ""
        if block:
            sections.append((heading, block))
    return sections


def structure_page_chunks(page: dict) -> list[Document]:
    """Un chunk par section de page ; une section trop longue est redécoupée."""
    docs = []
    for heading, block in split_sections(page["text"]):
        context = f"Page : {page['title']}" + (f" > {heading}" if heading else "")
        pieces = [block] if len(block) <= MAX_SECTION_SIZE else splitter.split_text(block)
        for piece in pieces:
            docs.append(make_document(f"{context}\n{piece}", page_metadata(page), "structure"))
    return docs


def structure_carte_chunks(record: dict) -> list[Document]:
    """Un produit = un chunk entier (jamais coupé : les allergènes restent avec leur plat)."""
    meta = carte_metadata(record)
    context = f"{CARTE_NAME}" + (f" — {record['section']}" if record["section"] else "")
    if record["kind"] == "product":
        return [make_document(f"{context}\nProduit : {record['name']}\n{record['text']}", meta, "structure")]
    pieces = [record["text"]] if len(record["text"]) <= MAX_SECTION_SIZE else splitter.split_text(record["text"])
    return [make_document(f"{context}\n{piece}", meta, "structure") for piece in pieces]


def structure_chunks(pages: list[dict], carte: list[dict]) -> list[Document]:
    """Découpage qui suit la structure : sections de pages et produits de la carte."""
    docs = [doc for page in pages for doc in structure_page_chunks(page)]
    docs += [doc for record in carte for doc in structure_carte_chunks(record)]
    return docs


# ---------------------------------------------------------------- point d'entrée
def build_chunks(strategy: str) -> list[Document]:
    """Construit les chunks d'une stratégie à partir des fichiers de data/processed/."""
    pages, carte = load_jsonl(PAGES_FILE), load_jsonl(CARTE_FILE)
    if strategy == "recursive":
        docs = recursive_chunks(pages, carte)
    elif strategy == "structure":
        docs = structure_chunks(pages, carte)
    else:
        raise ValueError(f"Stratégie inconnue : {strategy} (choix : {STRATEGIES})")
    unique = {doc.metadata["chunk_id"]: doc for doc in docs}  # retire les chunks identiques
    return list(unique.values())


if __name__ == "__main__":
    for name in STRATEGIES:
        chunks = build_chunks(name)
        sizes = sorted(len(d.page_content) for d in chunks)
        print(f"{name:10} : {len(chunks):4} chunks | taille médiane {sizes[len(sizes) // 2]} "
              f"| max {sizes[-1]} caractères")
