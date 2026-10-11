"""
scrape_site.py : Collecter le contenu public du site Traiteur démo.

Ce que fait le script :
  1. Lit robots.txt et ne visite QUE les pages autorisées.
  2. Récupère la liste des pages depuis le sitemap, puis complète en suivant les liens internes.
  3. Extrait le texte utile de chaque page (sans menus, footer, cookies...).
  4. Sauvegarde tout dans data/raw/ :
       - pages.jsonl      -> 1 ligne JSON par page (url, titre, description, texte)
       - pages/*.md       -> 1 fichier Markdown lisible par page

Installation :
    pip install requests beautifulsoup4 lxml trafilatura protego

Lancement :
    python scrape_site.py
"""

import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from protego import Protego

# ---------------------------------------------------------------- réglages
BASE = "https://www.exemple-traiteur.fr"
DOMAIN = urlparse(BASE).netloc
# Mets ton vrai email : c'est la façon polie de s'identifier quand on scrape.
USER_AGENT = "RAGStudentProject/1.0 (projet etudiant; contact: ton.email@exemple.com)"
DELAY_SECONDS = 2.0      # pause entre deux requêtes pour ne pas surcharger leur serveur
MAX_PAGES = 300          # garde-fou
OUT_DIR = Path("data/raw")

# Fichiers qu'on ne veut pas télécharger (images, PDF, etc.)
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".css", ".js", ".xml", ".mp4", ".woff", ".woff2",
)

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


# ---------------------------------------------------------------- utilitaires
def fetch(url: str) -> requests.Response | None:
    """GET poli : pause, timeout, et on ignore les erreurs au lieu de planter."""
    time.sleep(DELAY_SECONDS)
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"  ! échec {url} : {e}")
        return None


def normalize(url: str) -> str | None:
    """Garde uniquement les URLs internes, sans #ancre ni ?paramètres."""
    url, _ = urldefrag(url)
    p = urlparse(url)
    if p.scheme not in ("http", "https") or p.netloc != DOMAIN:
        return None
    if p.path.lower().endswith(SKIP_EXTENSIONS):
        return None
    return f"https://{DOMAIN}{p.path or '/'}"  # on retire la query string


def load_robots() -> Protego:
    r = session.get(urljoin(BASE, "/robots.txt"), timeout=20)
    # Protego gère les jokers (*) de robots.txt, contrairement à urllib.robotparser
    return Protego.parse(r.text)


def urls_from_sitemap(url: str, seen: set[str]) -> list[str]:
    """Lit un sitemap (ou un index de sitemaps) récursivement."""
    if url in seen:
        return []
    seen.add(url)
    r = fetch(url)
    if r is None:
        return []
    soup = BeautifulSoup(r.content, "xml")
    urls: list[str] = []
    for sm in soup.find_all("sitemap"):          # index -> sous-sitemaps
        loc = sm.find("loc")
        if loc:
            urls += urls_from_sitemap(loc.text.strip(), seen)
    for u in soup.find_all("url"):               # pages
        loc = u.find("loc")
        if loc:
            urls.append(loc.text.strip())
    return urls


def extract(html: str, url: str) -> dict:
    """Extrait titre, description et texte principal d'une page."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    description = meta.get("content", "").strip() if meta else ""

    # trafilatura enlève menus, footer, bandeau cookies... et garde le contenu
    try:
        text = trafilatura.extract(html, url=url, include_tables=True, output_format="markdown")
    except Exception:
        text = trafilatura.extract(html, url=url, include_tables=True)

    if not text:  # secours : texte brut de la page sans les balises inutiles
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

    links = [a["href"] for a in soup.find_all("a", href=True)]
    return {"title": title, "description": description, "text": text or "", "links": links}


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/") or "accueil"
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", path)[:120]


# ---------------------------------------------------------------- programme principal
def main() -> None:
    (OUT_DIR / "pages").mkdir(parents=True, exist_ok=True)
    robots = load_robots()

    def allowed(u: str) -> bool:
        return robots.can_fetch(u, USER_AGENT)

    # 1) Points de départ : le sitemap + la page d'accueil
    seeds = [BASE + "/"]
    for sm in robots.sitemaps or [BASE + "/sitemap.xml"]:
        seeds += urls_from_sitemap(sm, set())
    print(f"{len(seeds)} URLs trouvées dans le sitemap")

    queue: deque[str] = deque()
    seen: set[str] = set()
    for s in seeds:
        n = normalize(s)
        if n and n not in seen:
            seen.add(n)
            queue.append(n)

    # 2) Parcours des pages
    saved = 0
    with open(OUT_DIR / "pages.jsonl", "w", encoding="utf-8") as jsonl:
        while queue and saved < MAX_PAGES:
            url = queue.popleft()
            if not allowed(url):
                print(f"  - interdit par robots.txt : {url}")
                continue

            r = fetch(url)
            if r is None or "text/html" not in r.headers.get("Content-Type", ""):
                continue

            page = extract(r.text, url)

            # On découvre de nouvelles pages via les liens internes
            for href in page.pop("links"):
                n = normalize(urljoin(url, href))
                if n and n not in seen:
                    seen.add(n)
                    queue.append(n)

            if len(page["text"]) < 50:  # page vide ou purement technique
                continue

            record = {
                "url": url,
                **page,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            md = (
                f"# {record['title']}\n\n"
                f"Source : {url}\n\n"
                f"> {record['description']}\n\n"
                f"{record['text']}\n"
            )
            (OUT_DIR / "pages" / f"{slugify(url)}.md").write_text(md, encoding="utf-8")

            saved += 1
            print(f"[{saved}] {url}")

    print(f"\nTerminé : {saved} pages sauvegardées dans {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
