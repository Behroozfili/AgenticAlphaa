from rag.backfill_graph import build_parser, rows_to_documents


def test_rows_to_documents_preserves_point_in_time_metadata_and_deduplicates():
    row = {
        "url_hash": "hash-1", "chunk_index": 2, "ticker": "MSFT",
        "source_type": "news", "published_at": "2026-08-20T12:00:00Z",
        "ingested_at": "2026-08-20T12:05:00Z", "url": "https://example.test/a",
        "title": "Microsoft update", "text": "Microsoft announced a cloud product update.",
    }
    documents = rows_to_documents(
        [row, dict(row)], ticker="MSFT", dataset_version="adjusted-ohlcv-cache-v1",
    )
    assert len(documents) == 1
    document = documents[0]
    assert document.retrieved_at == row["ingested_at"]
    assert document.effective_from == row["published_at"]
    assert document.dataset_version == "adjusted-ohlcv-cache-v1"
    assert document.metadata["materialized_at"] == row["ingested_at"]
    assert document.revision == "legacy-chunk-2"


def test_rows_to_documents_skips_temporally_incomplete_rows():
    assert rows_to_documents(
        [{"url_hash": "hash", "chunk_index": 0, "text": "x"}],
        ticker="MSFT", dataset_version="v1",
    ) == []


def test_parser_requires_cutoff_and_dataset_version():
    args = build_parser().parse_args([
        "MSFT", "--as-of", "2026-09-18T20:00:00Z",
        "--dataset-version", "adjusted-ohlcv-cache-v1", "--limit", "8",
    ])
    assert args.ticker == "MSFT"
    assert args.limit == 8
