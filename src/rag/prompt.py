"""
prompt.py — Étape 3b : les règles du bot et la mise en forme du contexte envoyé au LLM.

Le LLM ne sert pas de source de connaissance : il reformule les extraits numérotés qu'on lui
donne ("grounding"). Les numéros [n] qu'il cite sont ensuite traduits en sources par le code.
"""

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
5. Frais de livraison : ne fais aucun calcul (majorations, totaux). Tu peux citer le tarif
   écrit dans un extrait, sinon renvoie vers l'équipe.
6. Prix des produits : recopie-les tels qu'ils sont écrits (HT / TTC), sans les recalculer.
7. Sois concis : quelques phrases ou une courte liste."""


def format_context(docs: list[Document]) -> str:
    """Numérote les extraits : '[1] (source : ...)\\n texte'. Le numéro sert de référence au LLM."""
    blocks = [f"[{i}] (source : {doc.metadata['citation']})\n{doc.page_content}"
              for i, doc in enumerate(docs, start=1)]
    return "\n\n".join(blocks)


def build_messages(question: str, docs: list[Document]) -> list[tuple[str, str]]:
    """Messages pour le LLM : les règles (system) puis les extraits et la question (user)."""
    user = f"Extraits :\n\n{format_context(docs)}\n\nQuestion : {question}"
    return [("system", SYSTEM_PROMPT), ("human", user)]
