"""
score.py : Noter les réponses du bot et comparer les deux stratégies de chunking.

Trois niveaux de mesure :
  1. Recherche (sans LLM)   : la source attendue est-elle dans les extraits ? (hit rate@6, MRR)
  2. RAGAS (LLM juge)       : faithfulness, answer relevancy, context precision, context recall
  3. Règles métier (code)   : allergènes officiels, renvoi vers l'équipe, montant exact...

Entrées : eval/questions.jsonl et eval/results/answers_<stratégie>.jsonl (run_answers.py)
Sorties : eval/results/scores.csv (détail par question) et eval/results/summary.md (tableau)

Lancement (depuis la racine du projet) :
    python eval/score.py              # tout, RAGAS compris (long : LLM local)
    python eval/score.py --no-ragas   # recherche + règles seulement (quelques secondes)
"""

import csv
import math
import sys

from common import QUESTIONS_FILE, RESULTS_DIR, answers_file, load_jsonl, source_key

from src.config import TEAM_PHONE
from src.ingestion.chunking import STRATEGIES
from src.rag.chain import LLM_MODEL
from src.rag.retriever import TOP_K, retrieve

# Juge RAGAS : option C retenue (même modèle que le bot, local). Biaisé et peu fiable :
# à remplacer par un modèle plus fort pour des chiffres publiables.
JUDGE_MODEL = LLM_MODEL
JUDGE_NUM_CTX = 8192  # les prompts RAGAS contiennent les extraits + leurs propres consignes
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
SCORES_FILE = RESULTS_DIR / "scores.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.md"
CONTACT = TEAM_PHONE


# ---------------------------------------------------------------- 1. recherche
def retrieval_scores(question: str, expected: list[str], strategy: str) -> dict:
    """Rang de la première source attendue parmi les extraits récupérés (0 = absente)."""
    found = [source_key(doc) for doc in retrieve(question, strategy=strategy, k=TOP_K)]
    rank = next((i for i, key in enumerate(found, start=1) if key in expected), 0)
    return {"hit": int(rank > 0), "rr": 1 / rank if rank else 0.0, "rank": rank}


# ---------------------------------------------------------------- 3. règles métier
def check_rules(checks: dict, record: dict) -> dict[str, bool]:
    """Vérifie chaque règle attendue sur la réponse telle que le client la voit."""
    rendered = record["rendered"].lower()
    ok_quotes = [q for q in record["quotes"] if q["ok"]]
    results = {}
    for text in checks.get("must_include", []):
        results[f"contient « {text} »"] = text.lower() in rendered
    for text in checks.get("must_not_include", []):
        results[f"sans « {text} »"] = text.lower() not in rendered
    if checks.get("contact"):
        results["renvoie vers l'équipe"] = CONTACT in rendered
    if checks.get("official_allergens"):
        results["liste officielle affichée"] = "allergens" in record["notices"]
    if "amount_ht" in checks:
        results[f"montant {checks['amount_ht']} € HT"] = any(q["total_ht"] == checks["amount_ht"] for q in ok_quotes)
    if checks.get("no_price"):
        results["aucun prix inventé"] = not ok_quotes
    return results


# ---------------------------------------------------------------- 2. RAGAS
def ragas_scores(questions: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    """Scores RAGAS par question (questions passées par le RAG, avec extraits et référence)."""
    from langchain_ollama import ChatOllama
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    from src.ingestion.vectorstore import get_embeddings

    evaluated = [r for r in records if r["contexts"]]
    samples = [SingleTurnSample(user_input=r["question"], response=r["response"],
                                retrieved_contexts=r["contexts"],
                                reference=questions[r["id"]]["reference"]) for r in evaluated]
    judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=JUDGE_NUM_CTX))
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=judge,
                      embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
                      run_config=RunConfig(max_workers=1, timeout=600))
    rows = result.to_pandas().to_dict("records")
    names = dict(zip(RAGAS_METRICS, ("faithfulness", "answer_relevancy",
                                     "llm_context_precision_with_reference", "context_recall")))
    return {r["id"]: {m: row.get(col) for m, col in names.items()} for r, row in zip(evaluated, rows)}


# ---------------------------------------------------------------- assemblage
def score_strategy(questions: dict[str, dict], strategy: str, with_ragas: bool) -> list[dict]:
    """Une ligne par question : recherche, règles, RAGAS, temps de réponse."""
    records = load_jsonl(answers_file(strategy))
    ragas = ragas_scores(questions, records) if with_ragas else {}
    rows = []
    for record in records:
        item = questions[record["id"]]
        rules = check_rules(item["checks"], record)
        row = {"strategy": strategy, "id": record["id"], "category": item["category"],
               "elapsed": record["elapsed"], "rules_ok": int(all(rules.values())),
               "rules_failed": " ; ".join(name for name, ok in rules.items() if not ok)}
        if item["expected_sources"]:
            row.update(retrieval_scores(item["question"], item["expected_sources"], strategy))
        row.update(ragas.get(record["id"], {}))
        rows.append(row)
    return rows


def mean(values: list) -> float | None:
    """Moyenne en ignorant les valeurs absentes ou NaN (échec du juge)."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else None


def fmt(value: float | None, percent: bool = False) -> str:
    """0.8333 -> '0,83' ou '83 %' ; absent -> '—'."""
    if value is None:
        return "—"
    return f"{value:.0%}".replace("%", " %") if percent else f"{value:.2f}".replace(".", ",")


def summary_table(rows: list[dict], strategies: list[str]) -> str:
    """Tableau Markdown comparant les stratégies."""
    lines = ["| Mesure | " + " | ".join(strategies) + " |", "|---|" + "---|" * len(strategies)]

    def add(label: str, key: str, percent: bool = False) -> None:
        cells = [fmt(mean([r.get(key) for r in rows if r["strategy"] == s]), percent) for s in strategies]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    add("Recherche : hit rate@6", "hit", percent=True)
    add("Recherche : MRR", "rr")
    for metric in RAGAS_METRICS:
        add(f"RAGAS : {metric}", metric)
    add("Règles métier respectées", "rules_ok", percent=True)
    add("Temps de réponse moyen (s)", "elapsed")
    return "\n".join(lines)


def write_outputs(rows: list[dict], strategies: list[str], with_ragas: bool) -> None:
    """scores.csv (détail) et summary.md (tableau + échecs)."""
    columns = ["strategy", "id", "category", "hit", "rank", "rr", *RAGAS_METRICS,
               "rules_ok", "rules_failed", "elapsed"]
    with open(SCORES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    failures = [f"- `{r['strategy']}` **{r['id']}** : {r['rules_failed']}" for r in rows if r["rules_failed"]]
    misses = [f"- `{r['strategy']}` **{r['id']}**" for r in rows if r.get("hit") == 0]
    judge = f"juge RAGAS : `{JUDGE_MODEL}` (même modèle que le bot)" if with_ragas else "RAGAS non calculé"
    text = (f"# Résultats de l'évaluation\n\n{len(rows) // len(strategies)} questions, "
            f"bot `{LLM_MODEL}`, {judge}.\n\n{summary_table(rows, strategies)}\n\n"
            f"## Règles métier non respectées\n\n" + ("\n".join(failures) or "Aucune.") +
            "\n\n## Source attendue absente des extraits\n\n" + ("\n".join(misses) or "Aucune.") + "\n")
    SUMMARY_FILE.write_text(text, encoding="utf-8")


def main() -> None:
    with_ragas = "--no-ragas" not in sys.argv
    questions = {q["id"]: q for q in load_jsonl(QUESTIONS_FILE)}
    strategies = [s for s in STRATEGIES if answers_file(s).exists()]
    rows = [row for s in strategies for row in score_strategy(questions, s, with_ragas)]
    write_outputs(rows, strategies, with_ragas)
    print(SUMMARY_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
