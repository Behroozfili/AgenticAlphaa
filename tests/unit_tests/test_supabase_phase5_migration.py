"""Static contract checks for the additive Phase 5 Supabase migration."""

from pathlib import Path
import re


MIGRATION = Path(__file__).parents[2] / "supabase_phase5_provenance_migration.sql"


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_phase5_migration_is_additive_and_preserves_existing_data() -> None:
    sql = _sql()
    assert "alter table public.alpha_documents" in sql
    assert "add column if not exists" in sql
    for destructive in ("drop table", "truncate", "delete from", "drop column"):
        assert destructive not in sql


def test_phase5_columns_use_python_compatible_nullable_types() -> None:
    sql = _sql()
    expected = {
        "fiscal_year": "integer",
        "fiscal_quarter": "text",
        "period_start": "date",
        "period_end": "date",
        "duration_months": "integer",
        "filing_type": "text",
        "accession_number": "text",
        "metric_scope": "text",
        "gaap_status": "text",
    }
    for column, data_type in expected.items():
        assert re.search(rf"add column if not exists\s+{column}\s+{data_type}\b", sql)
    # Nullable is PostgreSQL's default; a NOT NULL addition would break legacy rows.
    alter_block = sql.split("alter table public.alpha_documents", 1)[1].split(";", 1)[0]
    assert "not null" not in alter_block


def test_rpc_signature_matches_vector_store_caller() -> None:
    sql = _sql()
    signature = sql.split(
        "create or replace function public.alpha_hybrid_search(", 1
    )[1].split(")\nreturns table", 1)[0]
    expected = (
        "query_embedding",
        "query_text",
        "filter_ticker",
        "days_back",
        "top_k",
        "rrf_k",
        "page_offset",
        "filter_fiscal_year",
        "filter_fiscal_quarter",
        "filter_period_end",
        "filter_metric_scope",
    )
    positions = [signature.index(name) for name in expected]
    assert positions == sorted(positions)
    for exact_filter in expected[7:]:
        # Both vector and FTS candidates must be filtered before RRF fusion.
        assert signature.count(exact_filter) == 1
        assert sql.count(f"({exact_filter} is null or") == 2


def test_indexes_follow_actual_query_paths() -> None:
    sql = _sql()
    assert "(ticker, published_at desc)" in sql
    assert "(ticker, fiscal_year, fiscal_quarter, metric_scope, period_end)" in sql
    assert "on public.alpha_documents (period_start)" not in sql
    assert "on public.alpha_documents (filing_type)" not in sql
