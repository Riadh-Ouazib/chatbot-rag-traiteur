"""Tests de l'interface Streamlit (étape 5), sans navigateur ni LLM.

AppTest exécute le script comme Streamlit le ferait ; la chaîne RAG est remplacée par une
fausse réponse pour que les tests soient rapides et ne dépendent pas d'Ollama.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.rag import chain
from src.rag.chain import Answer, Notice

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def fake_answer(question: str, strategy: str = "structure", history=None) -> Answer:
    """Réponse fictive qui garde une trace de l'historique reçu."""
    fake_answer.history = history
    return Answer(text=f"Réponse à : {question} [1]",
                  sources=[([1], "Conditions de livraison (https://x.com/livraison)")],
                  notices=[Notice("warning", "Contaminations croisées possibles.")],
                  question=question)


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(chain, "ollama_status", lambda: None)
    monkeypatch.setattr(chain, "warm_up", lambda strategy="structure": None)
    monkeypatch.setattr(chain, "answer", fake_answer)
    return AppTest.from_file(APP, default_timeout=30).run()


def test_page_vide_sans_erreur(app):
    assert not app.exception
    assert app.title[0].value == "Posez votre question"
    assert len(app.chat_message) == 0


def test_question_affiche_reponse_encadre_et_sources(app):
    app.chat_input[0].set_value("Quels délais ?").run()
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "Réponse à : Quels délais ? [1]" in app.markdown.values
    assert app.warning[0].value == "Contaminations croisées possibles."
    assert any("[Conditions de livraison](https://x.com/livraison)" in v for v in app.markdown.values)


def test_historique_transmis_pour_les_questions_de_suivi(app):
    app.chat_input[0].set_value("Livraison au 92100 ?").run()
    app.chat_input[0].set_value("Et samedi ?").run()
    assert fake_answer.history == [("Livraison au 92100 ?", "Réponse à : Livraison au 92100 ? [1]")]
    assert len(app.chat_message) == 4


def test_bouton_exemple_pose_la_question(app):
    app.sidebar.button[0].click().run()
    assert app.chat_message[0].markdown[0].value.startswith("Quels sont les délais")


def test_nouvelle_conversation_vide_l_historique(app):
    app.chat_input[0].set_value("Quels délais ?").run()
    new_conversation = next(b for b in app.sidebar.button if "Nouvelle conversation" in b.label)
    new_conversation.click().run()
    assert len(app.chat_message) == 0


def test_ollama_absent_message_clair(monkeypatch):
    monkeypatch.setattr(chain, "ollama_status", lambda: "Ollama ne répond pas.")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert app.error[0].value == "Ollama ne répond pas."
    assert len(app.chat_input) == 0  # st.stop() : pas de zone de saisie


def test_erreur_pendant_la_reponse_affichee_proprement(app, monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("serveur coupé")
    monkeypatch.setattr(chain, "answer", broken)
    app.chat_input[0].set_value("Quels délais ?").run()
    assert not app.exception
    assert "serveur coupé" in app.error[0].value
