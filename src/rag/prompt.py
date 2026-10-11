"""
prompt.py : Les règles du bot et la mise en forme du contexte envoyé au LLM.

Le LLM ne sert pas de source de connaissance : il reformule les extraits numérotés qu'on lui
donne ("grounding"). Les numéros [n] qu'il cite sont ensuite traduits en sources par le code.
"""

from langchain_core.documents import Document

from src.config import BRAND_NAME, TEAM_EMAIL, TEAM_PHONE

TEAM_CONTACT = f"{TEAM_PHONE} / {TEAM_EMAIL}"

SYSTEM_PROMPT = f"""Tu es l'assistant du traiteur {BRAND_NAME} (Paris / Île-de-France).

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
DELIVERY_PROMPT = f"""Tu es l'assistant du traiteur {BRAND_NAME}. Le client demande des frais de livraison.

1. Appelle l'outil calculer_frais_livraison :
   - lieu : le code postal ou la ville, tel que le client l'a écrit ;
   - date_livraison : ne la remplis pas, la date est lue dans la question par le code ;
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


def build_messages(question: str, docs: list[Document]) -> list:
    """Messages pour le LLM : les règles (system), puis les extraits et la question (user)."""
    system = SYSTEM_PROMPT
    user = f"Extraits :\n\n{format_context(docs)}\n\nQuestion : {question}"
    return [("system", system), ("human", user)]


CONDENSE_PROMPT = """Tu aides l'assistant d'un traiteur à comprendre une question de suivi.
Réécris la NOUVELLE question du client pour qu'elle soit compréhensible seule, en reprenant
du dialogue ce qui manque (produit, lieu, date...). Si elle est déjà compréhensible seule,
ou si elle change de sujet, recopie-la telle quelle.
Réponds uniquement par la question réécrite, sans explication.

Exemples :
- Dialogue : "Combien coûte la livraison au 75008 ?" / Nouvelle question : "Et le dimanche ?"
  -> Combien coûte la livraison au 75008 le dimanche ?
- Dialogue : "Les choux contiennent-ils du lactose ?" / Nouvelle question : "Et les macarons ?"
  -> Les macarons contiennent-ils du lactose ?
- Dialogue : "Quels délais pour un buffet ?" / Nouvelle question : "Faites-vous des plats végans ?"
  -> Faites-vous des plats végans ?"""

MAX_HISTORY_ANSWER_CHARS = 300  # réponses précédentes tronquées : le contexte est limité


def build_condense_messages(history: list[tuple[str, str]], question: str) -> list:
    """Messages pour reformuler une question de suivi à partir des derniers échanges."""
    dialogue = "\n".join(f"Client : {q}\nAssistant : {a[:MAX_HISTORY_ANSWER_CHARS]}" for q, a in history)
    return [("system", CONDENSE_PROMPT),
            ("human", f"Dialogue :\n{dialogue}\n\nNouvelle question : {question}")]


def build_delivery_messages(question: str) -> list:
    """Messages pour une question de frais de livraison : consignes de l'outil + question."""
    return [("system", DELIVERY_PROMPT), ("human", question)]
