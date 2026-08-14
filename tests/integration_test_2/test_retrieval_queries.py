"""
test_retrieval_queries.py — find WHICH query makes social retrieval return 0.

Retrieval works with a natural query (proven: 20 chunks). But the live
SentimentAgent got 0. The variable is the QUERY string the Brain LLM generates.
This script runs retrieve_raw for several query styles the Brain might produce
(including its documented fallback) at both days_back=None and days_back=7, so
we can see exactly which query/window combination collapses to 0.

Run:
    python test_retrieval_queries.py AMD
"""
import logging, os, sys
from dotenv import load_dotenv
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

logging.basicConfig(level=logging.WARNING)  # quiet; we print our own summary

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AMD"
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path: sys.path.insert(0, root)

    from rag.retriever import AlphaRetriever
    from rag.vector_store import AlphaVectorStore
    from rag.embedding_manager import get_embedder

    store = AlphaVectorStore(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=(os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]),
    )
    retriever = AlphaRetriever(
        vector_store=store, embedder=get_embedder(),
        stage1_k=int(os.environ.get("SENTIMENT_RETRIEVAL_STAGE1_K", "20")),
        apply_freshness_rerank=False, apply_diversity_filter=False,
    )

    # Query styles the Brain plausibly emits.
    queries = {
        "natural (my test)":        f"{ticker} sentiment market news reddit",
        "brain fallback":           f"{ticker} earnings sentiment investor reaction",
        "just ticker":              ticker,
        "company-name style":       "Advanced Micro Devices stock sentiment",
        "very specific / narrow":   f"{ticker} Q3 2026 earnings guidance data center revenue miss analyst downgrade reaction social media",
        "question style":           f"What is the current market sentiment and investor reaction toward {ticker} stock?",
        "empty-ish":                f"{ticker}",
    }

    def count(q, days):
        try:
            return len(retriever.retrieve_raw(query=q, ticker=ticker, days_back=days))
        except Exception as e:
            return f"ERR:{e}"

    print(f"\n{'QUERY STYLE':<26} {'db=None':>8} {'db=7':>6}   query")
    print("-" * 90)
    for label, q in queries.items():
        c_none = count(q, None)
        c_7    = count(q, 7)
        flag = "  <-- ZERO" if (c_none == 0 or c_7 == 0) else ""
        print(f"{label:<26} {str(c_none):>8} {str(c_7):>6}   {q[:44]!r}{flag}")

    print("\n=== READING THIS ===")
    print("db=None column = retrieval with NO recency filter.")
    print("db=7 column    = what the agent actually sends by default.")
    print("If a row shows a healthy db=None but db=7 = 0, the 7-day window is killing")
    print("that query -> raise the default days_back. If some query style shows 0 even")
    print("at db=None, that phrasing is semantically too far from the stored text ->")
    print("the Brain prompt should produce simpler queries (or add a keyword fallback).")

if __name__ == "__main__":
    main()