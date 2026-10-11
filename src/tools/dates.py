"""
dates.py : Lire la date de livraison écrite par le client ("samedi", "14 novembre 2026", "17/11").

C'est le code, pas le LLM, qui lit la date. L'évaluation a montré que le petit
modèle recopiait une date fausse du calendrier quand la date demandée n'y figurait pas
("mardi 17 novembre" -> "samedi 17 octobre"), ce qui faussait le prix.
"""

import re
from datetime import date, timedelta

MONTHS = {"janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
          "juillet": 7, "aout": 8, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11,
          "decembre": 12, "décembre": 12}
WEEKDAY_NUMBERS = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4, "samedi": 5, "dimanche": 6}

DAY_MONTH = re.compile(r"\b(\d{1,2})(?:er)?\s+(" + "|".join(MONTHS) + r")(?:\s+(\d{4}))?\b", re.IGNORECASE)
NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2}|\d{4}))?\b")
WEEKDAY = re.compile(r"\b(" + "|".join(WEEKDAY_NUMBERS) + r")\b", re.IGNORECASE)


def next_occurrence(day: int, month: int, year: int | None, today: date) -> date | None:
    """Date du jour/mois donné ; sans année, la prochaine occurrence (aujourd'hui compris)."""
    try:
        if year is not None:
            return date(year if year > 99 else 2000 + year, month, day)
        candidate = date(today.year, month, day)
        return candidate if candidate >= today else date(today.year + 1, month, day)
    except ValueError:  # 31 février...
        return None


def parse_delivery_date(text: str, today: date | None = None) -> date | None:
    """Date de livraison mentionnée dans le texte, ou None si aucune.

    Priorité : date explicite ("14 novembre 2026", "14/11"), puis "après-demain" / "demain" /
    "aujourd'hui", puis un jour de la semaine ("samedi" = le prochain samedi après aujourd'hui).
    """
    today = today or date.today()
    if match := DAY_MONTH.search(text):
        day, month, year = match.groups()
        return next_occurrence(int(day), MONTHS[month.lower()], int(year) if year else None, today)
    if match := NUMERIC.search(text):
        day, month, year = match.groups()
        return next_occurrence(int(day), int(month), int(year) if year else None, today)
    lowered = text.lower()
    if "après-demain" in lowered or "apres-demain" in lowered:
        return today + timedelta(days=2)
    if "demain" in lowered:
        return today + timedelta(days=1)
    if "aujourd'hui" in lowered or "aujourd’hui" in lowered:
        return today
    if match := WEEKDAY.search(text):
        days_ahead = (WEEKDAY_NUMBERS[match.group(1).lower()] - today.weekday()) % 7 or 7
        return today + timedelta(days=days_ahead)
    return None
