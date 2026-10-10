"""
run_answers.py — Étape 6a : faire répondre le bot aux questions d'évaluation.

Les réponses sont enregistrées une fois pour toutes (étape lente, ~5 min par stratégie) ;
score.py les note ensuite autant de fois que nécessaire.

Lancement (depuis la racine du projet, Ollama démarré) :
    python eval/run_answers.py                  # les deux stratégies
    python eval/run_answers.py structure        # une seule
"""

import sys
import time

from common import QUESTIONS_FILE, answers_file, load_jsonl, source_key, write_jsonl

from src.ingestion.chunking import STRATEGIES
from src.rag.chain import answer, format_answer


def run_question(item: dict, strategy: str) -> dict:
    """Pose une question au bot et garde tout ce qu'il faut pour la notation."""
    start = time.perf_counter()
    result = answer(item["question"], strategy=strategy)
    return {
        "id": item["id"],
        "strategy": strategy,
        "question": item["question"],
        "response": result.text,             # texte du LLM seul (noté par RAGAS)
        "rendered": format_answer(result),   # ce que voit le client (vérifications métier)
        "contexts": [doc.page_content for doc in result.contexts],
        "context_sources": [source_key(doc) for doc in result.contexts],
        "notices": [notice.kind for notice in result.notices],
        "quotes": [{"ok": q.ok, "total_ht": str(q.total_ht) if q.total_ht else None}
                   for q in result.quotes],
        "elapsed": round(time.perf_counter() - start, 2),
    }


def main() -> None:
    questions = load_jsonl(QUESTIONS_FILE)
    for strategy in sys.argv[1:] or STRATEGIES:
        records = []
        for i, item in enumerate(questions, start=1):
            records.append(run_question(item, strategy))
            print(f"[{strategy}] {i}/{len(questions)} {item['id']} ({records[-1]['elapsed']} s)")
        write_jsonl(answers_file(strategy), records)
        print(f"-> {answers_file(strategy)}")


if __name__ == "__main__":
    main()
