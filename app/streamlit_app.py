"""
streamlit_app.py — Étape 5 : interface de chat du bot.

Lancement (depuis la racine du projet, Ollama démarré) :
    streamlit run app/streamlit_app.py

Streamlit relance tout ce script à chaque interaction : l'historique du chat est donc gardé
dans st.session_state, et les chargements lents (modèles, base Chroma) dans st.cache_resource.
"""

import re
import sys
import time
from pathlib import Path

import streamlit as st

# Streamlit ajoute app/ au chemin d'import, pas la racine du projet : on l'ajoute pour "src"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.chunking import STRATEGIES  # noqa: E402
from src.rag import chain  # noqa: E402

EXAMPLES = [
    "Quels sont les délais pour commander un buffet ?",
    "Les mini macarons contiennent-ils des fruits à coque ?",
    "Combien coûte la livraison à Boulogne-Billancourt samedi ?",
    "Proposez-vous des plateaux repas végétariens ?",
    "Comment annuler ma commande ?",
]
NOTICE_STYLES = {"allergens": (st.info, "📋"), "warning": (st.warning, "⚠️"), "delivery": (st.info, "🚚")}
SITE_CITATION = re.compile(r"^(.*) \((https?://[^)]+)\)$")  # "Titre (https://...)"


@st.cache_resource(show_spinner="Chargement des modèles (une seule fois)…")
def warm_up(strategy: str) -> None:
    """Charge embeddings, base Chroma et LLM au premier affichage, pas à la première question."""
    chain.warm_up(strategy)


def citation_markdown(citation: str) -> str:
    """Pages du site en lien cliquable ; références de la carte PDF en texte."""
    match = SITE_CITATION.match(citation)
    return f"[{match.group(1)}]({match.group(2)})" if match else citation


def render_answer(answer: chain.Answer, asked: str, elapsed: float) -> None:
    """Affiche la réponse du LLM, les encadrés ajoutés par le code et les sources."""
    st.markdown(answer.text)
    for notice in answer.notices:
        show, icon = NOTICE_STYLES[notice.kind]
        show(notice.text, icon=icon)
    if answer.sources:
        with st.expander(f"Sources ({len(answer.sources)})"):
            for numbers, citation in answer.sources:
                st.markdown(f"[{', '.join(map(str, numbers))}] {citation_markdown(citation)}")
    details = [f"⏱ {elapsed:.1f} s".replace(".", ",")]
    if answer.question and answer.question != asked:
        details.append(f"compris comme : « {answer.question} »")
    st.caption(" · ".join(details))


def history_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """[(question, réponse), ...] des échanges précédents, pour les questions de suivi."""
    return [(m["question"], m["answer"].text) for m in messages if m["role"] == "assistant"]


def sidebar() -> str:
    """Barre latérale : exemples, stratégie de chunking, nouvelle conversation. Renvoie la stratégie."""
    with st.sidebar:
        st.header("🍽️ Assistant traiteur")
        st.caption("Répond à partir du site et de la carte PDF, en citant ses sources.")
        st.subheader("Exemples")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example
        strategy = st.radio("Découpage des documents (chunking)", STRATEGIES,
                            index=STRATEGIES.index(chain.DEFAULT_STRATEGY),
                            help="'structure' : un chunk par produit / section. "
                                 "'recursive' : morceaux de taille fixe. Comparés à l'étape 6.")
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
        st.divider()
        st.caption(f"Prototype non officiel · modèle local {chain.LLM_MODEL}. "
                   f"Pour toute commande : {chain.TEAM_CONTACT}")
    return strategy


def ask(question: str, strategy: str) -> None:
    """Traite une question : l'affiche, appelle la chaîne RAG, affiche et mémorise la réponse."""
    messages = st.session_state.messages
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            with st.spinner("Je cherche dans la carte et sur le site…"):
                answer = chain.answer(question, strategy=strategy, history=history_pairs(messages))
        except Exception as error:  # l'interface ne doit jamais afficher une trace Python brute
            st.error(f"Désolé, une erreur est survenue : {error}")
            return
        elapsed = time.perf_counter() - start
        render_answer(answer, question, elapsed)
    messages += [{"role": "user", "content": question},
                 {"role": "assistant", "question": question, "answer": answer, "elapsed": elapsed}]


def main() -> None:
    st.set_page_config(page_title="Assistant traiteur", page_icon="🍽️")
    st.session_state.setdefault("messages", [])
    strategy = sidebar()

    if problem := chain.ollama_status():
        st.error(problem, icon="🔌")
        st.stop()
    warm_up(strategy)

    st.title("Posez votre question")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["answer"], message["question"], message["elapsed"])

    question = st.chat_input("Délais, allergènes, livraison…") or st.session_state.pop("pending", None)
    if question:
        ask(question, strategy)


main()
