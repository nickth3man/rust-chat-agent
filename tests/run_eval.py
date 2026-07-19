"""Test runner for answerbot multi-model evaluation.

Usage: python tests/run_eval.py

Runs each query against each model, saving results to results/<model>/<qNN>.*
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIG = ROOT / "config" / "models.json"
QUERIES_FILE = ROOT / "tests" / "queries.txt"

# (model_id, reasoning_enabled)
MODELS = [
    ("openai/gpt-oss-20b", True),
    ("google/gemini-2.5-flash-lite", True),
    ("mistralai/mistral-small-3.2-24b-instruct", False),
    ("meta-llama/llama-4-scout", False),
]


def write_config(model: str, reasoning: bool):
    config = {"model": model, "temperature": 0.7, "reasoning": reasoning}
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")


def load_queries() -> list[tuple[int, str]]:
    lines = QUERIES_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [(i + 1, line.strip()) for i, line in enumerate(lines) if line.strip()]


def run_one(model: str, qid: int, question: str) -> dict:
    """Run a single query and return result metadata."""
    model_safe = model.replace("/", "_").replace(":", "_")
    out_dir = RESULTS / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear journal before run so we capture only this query's events
    journal_path = ROOT / "journal.jsonl"
    if journal_path.exists():
        journal_path.unlink()

    start = time.time()
    result = subprocess.run(
        ["cargo", "run", "--", question],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    elapsed = time.time() - start

    # Decode output manually with UTF-8, replacing undecodable bytes.
    # Using text=True in subprocess.run defaults to cp1252 on Windows, which
    # breaks on em-dashes and other Unicode characters common in LLM output.
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

    # Save outputs
    out_dir.joinpath(f"q{qid:02d}.txt").write_text(stdout, encoding="utf-8")
    out_dir.joinpath(f"q{qid:02d}.err.txt").write_text(stderr, encoding="utf-8")

    # Save journal if it exists
    if journal_path.exists():
        import shutil
        shutil.copy(journal_path, out_dir / f"q{qid:02d}.jsonl")

    # Parse journal for events
    events = []
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Extract metrics
    has_reasoning = any(e.get("event") == "reasoning" for e in events)
    has_requery = any(e.get("event") == "requery" for e in events)
    answer_events = [e for e in events if e.get("event") == "answer"]
    answer_text = answer_events[-1].get("text", "") if answer_events else ""
    citation_count = answer_text.count("[S")
    sources = [e for e in events if e.get("event") == "source"]
    source_count = len(sources)

    return {
        "qid": qid,
        "question": question,
        "model": model,
        "exit_code": result.returncode,
        "elapsed": round(elapsed, 1),
        "answer_length": len(answer_text),
        "citations": citation_count,
        "sources": source_count,
        "has_requery": has_requery,
        "has_reasoning": has_reasoning,
        "errors": stderr.strip()[-200:] if result.returncode != 0 else "",
    }


def main():
    queries = load_queries()

    # Backup original config
    backup_config = CONFIG.read_text(encoding="utf-8") if CONFIG.exists() else None

    print(f"Models: {len(MODELS)}, Queries: {len(queries)}")
    print(f"Total runs: {len(MODELS) * len(queries)}")
    print(f"Results dir: {RESULTS}")
    print()

    all_results = []

    for mi, (model_id, reasoning) in enumerate(MODELS, 1):
        print(f"[{mi}/{len(MODELS)}] Model: {model_id} (reasoning={reasoning})")
        write_config(model_id, reasoning)

        for qi, (qid, question) in enumerate(queries, 1):
            prefix = f"  [{qi}/{len(queries)}]"
            try:
                r = run_one(model_id, qid, question)
                status = "OK" if r["exit_code"] == 0 else f"ERR({r['exit_code']})"
                rsn = "R" if r["has_reasoning"] else "-"
                rqy = "!" if r["has_requery"] else "-"
                print(f"{prefix} q{qid:02d}: {status} {r['elapsed']}s "
                      f"ans={r['answer_length']}ch cit={r['citations']} "
                      f"src={r['sources']} rsn={rsn} req={rqy}")
                all_results.append(r)
            except subprocess.TimeoutExpired:
                print(f"{prefix} q{qid:02d}: TIMEOUT")
                all_results.append({
                    "qid": qid, "question": question, "model": model_id,
                    "exit_code": -1, "elapsed": 60,
                    "answer_length": 0, "citations": 0, "sources": 0,
                    "has_requery": False, "has_reasoning": False,
                    "errors": "TIMEOUT",
                })
            except Exception as e:
                print(f"{prefix} q{qid:02d}: ERROR {e}")
                all_results.append({
                    "qid": qid, "question": question, "model": model_id,
                    "exit_code": -2, "elapsed": 0,
                    "answer_length": 0, "citations": 0, "sources": 0,
                    "has_requery": False, "has_reasoning": False,
                    "errors": str(e)[:200],
                })

        # Summary for this model
        model_results = [r for r in all_results if r["model"] == model_id]
        ok = sum(1 for r in model_results if r["exit_code"] == 0)
        avg_time = sum(r["elapsed"] for r in model_results) / len(model_results)
        avg_len = sum(r["answer_length"] for r in model_results) / len(model_results)
        reasoning_runs = sum(1 for r in model_results if r["has_reasoning"])
        requeries = sum(1 for r in model_results if r["has_requery"])
        print(f"  => {ok}/{len(model_results)} OK, avg {avg_time:.1f}s, "
              f"avg ans={avg_len:.0f}ch, reasoning in {reasoning_runs} runs, "
              f"{requeries} requeries")
        print()

    # Write full results as JSON for analysis
    RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS.joinpath("summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Per-model summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Restore original config
    if backup_config:
        CONFIG.write_text(backup_config)

    for model_id, _reasoning in MODELS:
        mr = [r for r in all_results if r["model"] == model_id]
        ok = sum(1 for r in mr if r["exit_code"] == 0)
        avg_t = sum(r["elapsed"] for r in mr) / len(mr) if mr else 0
        avg_l = sum(r["answer_length"] for r in mr) / len(mr) if mr else 0
        avg_c = sum(r["citations"] for r in mr) / len(mr) if mr else 0
        avg_s = sum(r["sources"] for r in mr) / len(mr) if mr else 0
        rsn = sum(1 for r in mr if r["has_reasoning"])
        req = sum(1 for r in mr if r["has_requery"])
        errs = [r for r in mr if r["exit_code"] != 0]
        print(f"\n{model_id}")
        print(f"  Success: {ok}/{len(mr)}  Avg time: {avg_t:.1f}s")
        print(f"  Avg answer: {avg_l:.0f} chars  Avg citations: {avg_c:.1f}  Avg sources: {avg_s:.1f}")
        print(f"  Reasoning runs: {rsn}/{len(mr)}  Requeries: {req}")
        if errs:
            for e in errs:
                print(f"  ERR q{e['qid']:02d}: {e['errors'][:100]}")


if __name__ == "__main__":
    main()
