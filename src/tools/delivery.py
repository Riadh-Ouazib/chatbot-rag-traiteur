"""
delivery.py : Calcul des frais de livraison (Python pur, aucun LLM).

Le LLM ne fait qu'appeler `calculer_frais_livraison` avec le lieu et la date extraits de la
question (function calling). Le calcul, lui, est fait ici et testé dans tests/test_delivery.py.

Règles (carte Automne-Hiver 2026/2027, p. 31, qui fait foi) :
  - tarif de la zone, valable du lundi au vendredi de 7h à 20h ;
  - samedi +50 % ; dimanche ou jour férié ×2 ; dimanche férié ×4 ;
  - "dernière minute" (commande après 17h la veille ou avant 10h le jour même) +100 %.
La grille ne dit pas si "dernière minute" et week-end se cumulent : on renvoie vers l'équipe.

Données (générées par src/ingestion/delivery_grid.py et postal_codes.py) : data/processed/delivery_grid.json et communes_idf.json.
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

import holidays

from src.ingestion.postal_codes import normalize_name
from src.rag.prompt import TEAM_CONTACT, WEEKDAYS

ROOT = Path(__file__).resolve().parents[2]
GRID_FILE = ROOT / "data" / "processed" / "delivery_grid.json"
COMMUNES_FILE = ROOT / "data" / "processed" / "communes_idf.json"

VAT_RATE = Decimal("0.10")  # 29,90 € HT -> 32,89 € TTC sur la grille
LAST_MINUTE_MULTIPLIER = Decimal("2")
GRID_SOURCE = "Carte Automne-Hiver 2026/2027, page PDF 31 et https://www.exemple-traiteur.fr/traiteur/livraison"

# "Paris, La Défense" est en zone 1 ; La Défense n'est pas une commune dans la base La Poste
PLACE_ALIASES = {"LA DEFENSE": ["92400", "92800"]}


@dataclass
class DeliveryQuote:
    """Résultat du calcul. `ok` faux = pas de prix fiable, `message` explique pourquoi."""
    ok: bool
    message: str
    total_ht: Decimal | None = None
    total_ttc: Decimal | None = None


# ---------------------------------------------------------------- données
@lru_cache(maxsize=1)
def load_zones() -> dict[str, str]:
    """Code postal -> zone. Le 95100 (Argenteuil) figure en zone 1 et en zone 2 sur la grille :
    on garde la première citée (zone 1)."""
    with open(GRID_FILE, encoding="utf-8") as f:
        zones = json.load(f)["zones"]
    by_code: dict[str, str] = {}
    for zone in sorted(zones):
        for code in zones[zone]["postal_codes"]:
            by_code.setdefault(code, zone)
    return by_code


@lru_cache(maxsize=1)
def load_prices() -> dict[str, Decimal]:
    """Zone -> tarif HT de base."""
    with open(GRID_FILE, encoding="utf-8") as f:
        zones = json.load(f)["zones"]
    return {zone: Decimal(str(info["price_ht"])) for zone, info in zones.items()}


@lru_cache(maxsize=1)
def load_communes() -> dict[str, list[str]]:
    """Nom de commune normalisé -> codes postaux (Île-de-France)."""
    with open(COMMUNES_FILE, encoding="utf-8") as f:
        return {**json.load(f), **PLACE_ALIASES}


# ---------------------------------------------------------------- lieu -> codes postaux
def find_communes(place: str) -> dict[str, list[str]]:
    """Communes correspondant au texte : nom exact, sinon tous les mots de la recherche
    présents dans le nom ("Boulogne" -> BOULOGNE BILLANCOURT ; "Paris" -> PARIS 1..20)."""
    communes = load_communes()
    query = normalize_name(place)
    if query in communes:
        return {query: communes[query]}
    words = set(query.split())
    return {name: codes for name, codes in communes.items() if words and words <= set(name.split())}


def zones_for_place(place: str) -> dict[str, tuple[str, str]] | str:
    """{code postal: (zone, commune)} pour un code postal ou une ville (zone '' si hors zone),
    ou un message d'erreur si le lieu n'est pas compris."""
    place = place.strip()
    zones = load_zones()
    if re.fullmatch(r"\d{5}", place):
        return {place: (zones.get(place, ""), place)}
    if re.search(r"\d", place) and not re.search(r"(?i)paris", place):
        return f"« {place} » n'est pas un code postal valide (5 chiffres)."
    matches = find_communes(place)
    if not matches:
        return f"Je ne trouve pas « {place} » parmi les communes d'Île-de-France."
    return {code: (zones.get(code, ""), name.title()) for name, codes in matches.items() for code in codes}


# ---------------------------------------------------------------- majorations
def day_multiplier(day: date) -> tuple[Decimal, str]:
    """Majoration liée au jour de livraison, avec son explication."""
    holiday = holidays.France(years=day.year, language="fr").get(day)
    if day.weekday() == 6 and holiday:
        return Decimal("4"), "dimanche férié : ×4"
    if day.weekday() == 6:
        return Decimal("2"), "dimanche : ×2"
    if holiday:
        return Decimal("2"), f"jour férié ({holiday}) : ×2"
    if day.weekday() == 5:
        return Decimal("1.5"), "samedi : +50 %"
    return Decimal("1"), f"{WEEKDAYS[day.weekday()]} : tarif normal"


def money(amount: Decimal) -> str:
    """Decimal('59.55') -> '59,55 €'."""
    return f"{amount:.2f} €".replace(".", ",")


def round_cents(amount: Decimal) -> Decimal:
    """Arrondi commercial au centime (0,005 -> 0,01)."""
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------- l'outil
def ask_team(reason: str) -> DeliveryQuote:
    """Cas que la grille ne couvre pas : pas de prix, renvoi vers l'équipe."""
    return DeliveryQuote(ok=False, message=f"{reason} Merci de contacter l'équipe : {TEAM_CONTACT}.")


def calculer_frais_livraison(lieu: str, date_livraison: str | None = None,
                             derniere_minute: bool = False) -> DeliveryQuote:
    """Calcule les frais de livraison Traiteur démo pour un lieu d'Île-de-France.

    Args:
        lieu: code postal (ex. "92100") ou nom de la ville (ex. "Boulogne-Billancourt", "Paris 15").
        date_livraison: date de livraison au format AAAA-MM-JJ, si le client la précise.
        derniere_minute: vrai seulement si le client commande après 17h la veille ou avant 10h le jour même.
    """
    found = zones_for_place(lieu)
    if isinstance(found, str):
        return ask_team(found)
    zones_found = {zone for zone, _ in found.values()}
    if zones_found == {""}:
        return ask_team(f"« {lieu} » est hors de nos zones de livraison habituelles.")
    if len(zones_found) > 1:
        choices = ", ".join(f"{name} {code} (zone {zone or 'hors zone'})"
                            for code, (zone, name) in found.items())
        return ask_team(f"« {lieu} » correspond à plusieurs codes postaux de zones différentes : "
                        f"{choices}. Précisez le code postal.")

    zone = zones_found.pop()
    base = load_prices()[zone]
    codes = ", ".join(found) if len(found) <= 3 else f"{len(found)} codes postaux"
    steps = [f"{lieu} ({codes}) : zone {zone}, {money(base)} HT du lundi au vendredi"]
    multiplier = Decimal("1")
    if date_livraison:
        try:
            day = date.fromisoformat(date_livraison)
        except ValueError:
            return ask_team(f"Date de livraison « {date_livraison} » non comprise (format attendu AAAA-MM-JJ).")
        multiplier, reason = day_multiplier(day)
        steps.append(f"livraison le {WEEKDAYS[day.weekday()]} {day:%d/%m/%Y}, {reason}")
    if derniere_minute:
        if multiplier != 1:
            return ask_team("La grille ne précise pas le tarif d'une commande de dernière minute "
                            "livrée un week-end ou un jour férié.")
        multiplier = LAST_MINUTE_MULTIPLIER
        steps.append("commande de dernière minute : +100 %")

    total_ht = round_cents(base * multiplier)
    total_ttc = round_cents(total_ht * (1 + VAT_RATE))
    steps.append(f"total : {money(total_ht)} HT, soit {money(total_ttc)} TTC")
    return DeliveryQuote(ok=True, message="Frais de livraison — " + " ; ".join(steps) + ".",
                         total_ht=total_ht, total_ttc=total_ttc)
