"""
prompt.py — Étape 3b : les règles du bot et la mise en forme du contexte envoyé au LLM.

Le LLM ne sert pas de source de connaissance : il reformule les extraits numérotés qu'on lui
donne ("grounding"). Les numéros [n] qu'il cite sont ensuite traduits en sources par le code.
"""

from datetime import date, timedelta

from langchain_core.documents import Document

TEAM_CONTACT = "01 00 00 00 00 / contact@exemple-traiteur.fr"

SYSTEM_PROMPT = f"""Tu es l'assistant du traiteur Traiteur démo (Paris / Île-de-France).

Règles :
1. Réponds en français, en phrases complètes, UNIQUEMENT à partir des extraits fournis.
2. Cite chaque information avec le numéro de son extrait entre crochets, par exemple [2].
3. Si les extraits ne permettent pas de répondre, dis-le clairement, n'invente rien,
   et propose de contacter l'équipe : {TEAM_CONTACT}.
4. Allergènes, ne devine jamais :
   - si le produit est dans un extrait, recopie sa liste d'allergènes EN ENTIER avec [n],
     puis réponds à la question. Exemple : "Les MINI MACARONS contiennent : arachides,
     fruits à coque, gluten, lactose [3]. Ils contiennent donc des fruits à coque."
   - si l'extrait indique "non indiqués", dis-le en citant [n] et renvoie vers l'équipe ;
   - si le produit n'apparaît dans aucun extrait, dis que tu ne le trouves pas dans la carte
     et renvoie vers l'équipe ({TEAM_CONTACT}).
5. Frais de livraison : ne fais aucun calcul (majorations, totaux, TVA). Le calcul est fait
   par un outil dédié ; s'il n'a pas été utilisé, renvoie vers l'équipe.
6. Prix des produits : recopie-les tels qu'ils sont écrits (HT / TTC), sans les recalculer.
7. Sois concis : quelques phrases ou une courte liste."""

# Questions de frais de livraison : le modèle n'a QUE l'outil, sans extraits. Avec les extraits
# (qui contiennent la grille), le petit modèle préférait calculer lui-même... et se trompait.
DELIVERY_PROMPT = f"""Tu es l'assistant du traiteur Traiteur démo. Le client demande des frais de livraison.

1. Appelle l'outil calculer_frais_livraison :
   - lieu : le code postal ou la ville, tel que le client l'a écrit ;
   - date_livraison : seulement si le client donne un jour, au format AAAA-MM-JJ,
     recopiée du calendrier ci-dessous ;
   - derniere_minute : vrai seulement si le client dit commander après 17h la veille
     ou le matin même avant 10h.
2. Si le client ne donne aucun lieu, n'appelle pas l'outil : demande-lui son code postal.
3. Après le résultat de l'outil, réponds en une ou deux phrases en français, en reprenant
   EXACTEMENT ses montants. Ne fais aucun calcul toi-même.
   Si l'outil renvoie vers l'équipe, transmets son message ({TEAM_CONTACT})."""


def format_context(docs: list[Document]) -> str:
    """Numérote les extraits : '[1] (source : ...)\\n texte'. Le numéro sert de référence au LLM."""
    blocks = [f"[{i}] (source : {doc.metadata['citation']})\n{doc.page_content}"
              for i, doc in enumerate(docs, start=1)]
    return "\n\n".join(blocks)


WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def calendar(today: date) -> str:
    """Les 8 prochains jours avec leur nom : un petit modèle recopie une date plus sûrement
    qu'il ne calcule "samedi prochain" lui-même."""
    days = [today + timedelta(days=i) for i in range(8)]
    lines = [f"- {WEEKDAYS[d.weekday()]} {d.isoformat()}" + (" (aujourd'hui)" if d == today else "")
             for d in days]
    return "Calendrier :\n" + "\n".join(lines)


def build_messages(question: str, docs: list[Document]) -> list:
    """Messages pour le LLM : les règles (system), puis les extraits et la question (user)."""
    system = SYSTEM_PROMPT
    user = f"Extraits :\n\n{format_context(docs)}\n\nQuestion : {question}"
    return [("system", system), ("human", user)]


def build_delivery_messages(question: str, today: date | None = None) -> list:
    """Messages pour une question de frais de livraison : consignes de l'outil + calendrier."""
    return [("system", f"{DELIVERY_PROMPT}\n\n{calendar(today or date.today())}"), ("human", question)]
