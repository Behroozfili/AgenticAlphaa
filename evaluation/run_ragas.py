"""
run_ragas.py — RAGAS evaluation for AgenticAlpha traces
========================================================
Runs reference-free RAGAS metrics on the prepared dataset:
  - faithfulness        : Are the report's claims inferable from retrieved contexts?
  - answer_relevancy    : Does the answer address the question?
  - context_utilization : Are retrieved contexts actually relevant/used?
    (reference-free variant of context precision; exact name depends on ragas version)

Requirements (install on your own machine):
    pip install "ragas>=0.2" langchain-openai datasets pandas python-dotenv

Environment:
    Put your key in a .env file (in this folder or the project root):
        OPENAI_API_KEY=sk-...
    The script loads it automatically via python-dotenv.
    (Setting the variable manually in the shell also still works.)

Usage:
    python run_ragas.py --dataset ragas_dataset.json --out ragas_results.csv
    python run_ragas.py --dataset ragas_dataset.json --out ragas_results.csv --model gpt-4o-mini

Notes for the thesis (important):
  1. The judge LLM is configured with temperature=0 — same determinism fix you applied
     to SentimentAgent. Mention this in your methods section.
  2. No ground_truth exists for open-ended investment analysis, so only
     reference-free metrics are used. Do NOT add context_recall / answer_correctness —
     they require a gold reference and would produce meaningless numbers here.
  3. With a small sample count, report results as descriptive statistics
     (mean ± range), not as statistically significant claims.
  4. Run the whole evaluation 2-3 times and report judge variance too — the judge
     itself is an LLM and its stability is part of your methodology story.
  5. COST: --model defaults to gpt-5.4-nano ($0.20/$1.25 per 1M tokens as of the
     pricing checked on 2026-07-03), which was the cheapest current OpenAI model at
     that time — NOT gpt-4o-mini, which no longer appears on OpenAI's pricing page
     and may be deprecated. Verify current pricing at
     https://developers.openai.com/api/docs/pricing before a large run, and pass
     --model explicitly (e.g. --model gpt-4o-mini) if you specifically need that
     model and it's still callable on your account.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ---- Load .env (searches: current working dir, script dir, then parent dirs) ----
try:
    from dotenv import load_dotenv, find_dotenv

    # 1) standard search upward from CWD
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)
    # 2) also try next to this script and one level up (project root),
    #    in case the script is run from elsewhere
    for candidate in (Path(__file__).resolve().parent / ".env",
                      Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
except ImportError:
    print("NOTE: python-dotenv not installed — falling back to shell environment variables.\n"
          "      Install with: pip install python-dotenv")


def _infer_ticker(text: str) -> str | None:
    """Best-effort ticker extraction from a task_query like
    'INVESTMENT ANALYSIS REPORT: MICROSOFT CORPORATION (MSFT)'."""
    m = re.search(r"\(([A-Z]{1,5})\)", text or "")
    return m.group(1) if m else None


def load_samples(path: str):
    """
    Load samples from ragas_dataset.json (the format produced by
    build_ragas_dataset.py — user_input/retrieved_contexts/response per
    record, plus legacy question/contexts/answer aliases and a
    _source_file field).

    This dataset has no explicit id/ticker/run fields, so they're derived
    here: id from the source filename, ticker from a "(TICKER)" pattern
    in the task query, run from position in the file (1-indexed).
    """
    with open(path, encoding="utf-8") as f:
        raw_samples = json.load(f)

    clean = []
    for i, s in enumerate(raw_samples, start=1):
        contexts = s.get("retrieved_contexts") or s.get("contexts") or []
        question = s.get("user_input") or s.get("question") or ""
        answer   = s.get("response") or s.get("answer") or ""
        source   = s.get("_source_file", f"sample_{i}")
        sample_id = s.get("id") or Path(source).stem

        if not contexts:
            print(f"  [skip] {sample_id}: no contexts")
            continue
        if not question or not answer:
            print(f"  [skip] {sample_id}: missing question or answer")
            continue

        clean.append(
            {
                "id": sample_id,
                "ticker": s.get("ticker") or _infer_ticker(question) or "UNKNOWN",
                "run": s.get("run", i),
                "user_input": question,        # ragas>=0.2 field name
                "response": answer,             # ragas>=0.2 field name
                "retrieved_contexts": contexts,  # ragas>=0.2 field name
            }
        )
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="ragas_dataset.json (or similarly-shaped file)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help=(
            "Judge model. Default gpt-5.4-nano — the cheapest current OpenAI "
            "model as of the pricing checked 2026-07-03 ($0.20/$1.25 per 1M "
            "tokens). NOT gpt-4o-mini, which is no longer on OpenAI's pricing "
            "page. Pass --model gpt-4o-mini explicitly if you need it and it's "
            "still callable on your account. Verify current pricing at "
            "https://developers.openai.com/api/docs/pricing before a large run."
        ),
    )
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY not found.\n"
            "Add it to a .env file (this folder or project root):\n"
            "    OPENAI_API_KEY=sk-...\n"
            "Or set it in the shell: $env:OPENAI_API_KEY='sk-...'"
        )
    print(f"API key loaded OK (source: .env or shell environment). Judge model: {args.model}")

    samples = load_samples(args.dataset)
    print(f"Loaded {len(samples)} samples from {args.dataset}")
    if not samples:
        sys.exit("No usable samples after filtering — nothing to evaluate.")

    # ---- RAGAS setup (ragas >= 0.2 API) ----
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # context relevance metric name varies by version; try the common ones
    try:
        from ragas.metrics import ContextUtilization
        ctx_metric = ContextUtilization()
    except ImportError:
        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference
            ctx_metric = LLMContextPrecisionWithoutReference()
        except ImportError:
            ctx_metric = None
            print("WARNING: no reference-free context metric found in this ragas version; "
                  "running faithfulness + relevancy only")

    judge = LangchainLLMWrapper(ChatOpenAI(model=args.model, temperature=0))

    metrics = [Faithfulness(llm=judge), ResponseRelevancy(llm=judge)]
    if ctx_metric is not None:
        ctx_metric.llm = judge
        metrics.append(ctx_metric)

    eval_ds = EvaluationDataset.from_list(
        [{k: s[k] for k in ("user_input", "response", "retrieved_contexts")} for s in samples]
    )

    print(f"Running RAGAS with metrics: {[m.name for m in metrics]}")
    result = evaluate(
        dataset=eval_ds,
        metrics=metrics,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=600),
    )

    df = result.to_pandas()
    # attach ids back
    meta = pd.DataFrame([{"id": s["id"], "ticker": s["ticker"], "run": s["run"]} for s in samples])
    df = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print("\nPer-sample results:")
    print(df.to_string())
    print("\nMeans by ticker:")
    num_cols = df.select_dtypes("number").columns.difference(["run"])
    print(df.groupby("ticker")[list(num_cols)].mean().round(3).to_string())


if __name__ == "__main__":
    main()