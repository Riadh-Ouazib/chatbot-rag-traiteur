"""
config.py : informations sur le traiteur, lues dans le fichier .env (jamais commité).

Le traiteur n'a pas donné son accord pour être cité : son nom, son site et ses contacts ne sont
pas dans le code. Copier .env.example en .env et y mettre les vraies valeurs. Sans .env, des
valeurs fictives sont utilisées et tout fonctionne quand même (tests compris).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BRAND_NAME = os.getenv("BRAND_NAME", "Traiteur démo")
# Variantes du nom présentes dans les titres des pages, séparées par des ";"
BRAND_ALIASES = [alias.strip() for alias in os.getenv("BRAND_ALIASES", BRAND_NAME).split(";") if alias.strip()]
SITE_URL = os.getenv("SITE_URL", "https://www.exemple-traiteur.fr").rstrip("/")
TEAM_PHONE = os.getenv("TEAM_PHONE", "01 00 00 00 00")
TEAM_EMAIL = os.getenv("TEAM_EMAIL", "contact@exemple-traiteur.fr")
# Email mis dans le user agent du scraper : la façon polie de s'identifier quand on scrape
SCRAPER_CONTACT = os.getenv("SCRAPER_CONTACT", "ton.email@exemple.com")
