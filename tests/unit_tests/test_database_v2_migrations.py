"""Offline contract gates for the additive Stage 02 Database V2 migrations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).parents[2]
SUPABASE = ROOT / "database" / "supabase"
NEO4J = ROOT / "database" / "neo4j"

MIGRATIONS = {
    "documents": SUPABASE / "001_alpha_documents_v2.sql",
    "forecasts": SUPABASE / "002_forecast_tables.sql",
    "analyses": SUPABASE / "003_analysis_versioning.sql",
    "evidence_identity": SUPABASE / "004_alpha_documents_v2_evidence_identity.sql",
    "graph": NEO4J / "001_temporal_evidence.cypher",
}


def _read(name: str) -> str:
    return MIGRATIONS[name].read_text(encoding="utf-8").lower()


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def test_versioned_migrations_exist_and_are_strictly_additive() -> None:
    assert all(path.is_file() for path in MIGRATIONS.values())

    destructive = re.compile(
        r"\b(drop\s+(?:table|column|schema|constraint|index)"
        r"|truncate\b|delete\s+from\b|alter\s+table\s+[^;]+\s+drop\b)"
    )
    for name, path in MIGRATIONS.items():
        content = path.read_text(encoding="utf-8").lower()
        if name != "evidence_identity":
            assert destructive.search(content) is None, path

    amendment_forward = _read("evidence_identity").split(
        "-- rollback sql", 1
    )[0]
    assert not re.search(
        r"\b(drop\s+(?:table|column|schema|index)|truncate\b|delete\s+from\b)",
        amendment_forward,
    )
    dropped_constraints = set(re.findall(
        r"drop\s+constraint\s+if\s+exists\s+([a-z0-9_]+)",
        amendment_forward,
    ))
    assert dropped_constraints == {
        "alpha_documents_v2_document_identity",
        "alpha_documents_v2_evidence_identity",
    }

    for name in ("documents", "forecasts", "analyses", "evidence_identity"):
        sql = _read(name)
        assert sql.lstrip().startswith("-- stage 02")
        assert re.search(r"\bbegin\s*;", sql)
        if name == "evidence_identity":
            assert "commit;\n\n-- rollback sql" in sql
        else:
            assert sql.rstrip().endswith("commit;")


def test_sql_dollar_quote_blocks_are_balanced() -> None:
    tag_pattern = re.compile(r"\$[a-z_]*\$")
    for name in ("documents", "forecasts", "analyses", "evidence_identity"):
        counts = Counter(tag_pattern.findall(_read(name)))
        assert counts
        assert all(count % 2 == 0 for count in counts.values())


def test_evidence_identity_amendment_is_ordered_idempotent_and_reversible() -> None:
    path = MIGRATIONS["evidence_identity"]
    assert path.name.startswith("004_")
    assert all(
        MIGRATIONS[name].name < path.name
        for name in ("documents", "forecasts", "analyses")
    )

    raw = _read("evidence_identity")
    forward, rollback = raw.split("-- rollback sql", 1)
    compact_forward = _compact(forward)
    compact_rollback = _compact(rollback)

    assert "001_alpha_documents_v2.sql must be applied before 004" in forward
    assert "to_regclass('public.alpha_documents_v2')" in forward
    assert forward.count("has an unexpected definition") == 4
    assert forward.count("drop constraint if exists") == 2
    assert forward.count("if current_definition is null then") == 2
    assert "alpha_documents_v2_document_evidence_chunk_identity" in forward
    assert "alpha_documents_v2_evidence_chunk_identity" in forward

    assert "execute separately and only with explicit authorization" in rollback
    assert "having count(*) > 1" in rollback
    assert rollback.count("rollback blocked:") == 2
    assert "drop constraint if exists alpha_documents_v2_document_evidence_chunk_identity" in compact_rollback
    assert "drop constraint if exists alpha_documents_v2_evidence_chunk_identity" in compact_rollback
    assert "add constraint alpha_documents_v2_document_identity" in compact_rollback
    assert "add constraint alpha_documents_v2_evidence_identity" in compact_rollback
    assert "delete from" not in compact_forward
    assert "delete from" not in compact_rollback
    assert "update public.alpha_documents_v2" not in compact_forward
    assert "update public.alpha_documents_v2" not in compact_rollback


def test_evidence_identity_amendment_preserves_temporal_security_and_routes() -> None:
    amendment = _read("evidence_identity").split("-- rollback sql", 1)[0]
    documents = _compact(_read("documents"))
    analyses = _compact(_read("analyses"))

    for forbidden in (
        "create or replace function",
        "create or replace view",
        "create policy",
        "enable row level security",
        "disable row level security",
        "grant ",
        "revoke ",
        "insert into",
        "update public.alpha_documents_v2",
        "delete from",
        "truncate ",
    ):
        assert forbidden not in amendment

    assert "alpha_hybrid_search_v2" in documents
    assert "alter table public.alpha_documents_v2 enable row level security" in documents
    assert "alpha_documents_v2_reconciliation" in documents
    assert "'vector_documents', 'legacy-v1', null, null, null" in analyses


def test_stage03_evidence_observations_fit_every_v2_unique_identity() -> None:
    """Map Stage 03 rows through offline embedding and the V2 adapter."""
    from rag.embedding_manager import AlphaEmbedder
    from rag.loader import RawDocument
    from rag.processor import AlphaProcessor
    from rag.vector_store import AlphaVectorStore

    common = {
        "title": "NVDA archived evidence",
        "content": "NVDA same immutable source content.",
        "url": "https://archive.invalid/nvda/evidence-1",
        "source_type": "archive",
        "ticker": "NVDA",
        "published_at": "2024-01-10T10:00:00+00:00",
        "source_id": "a" * 64,
        "revision": "1",
    }
    observations = [
        RawDocument(
            **common,
            retrieved_at="2024-01-11T09:00:00+00:00",
            effective_from="2024-01-10T10:00:00+00:00",
            dataset_version="snapshot-v1",
        ),
        RawDocument(
            **common,
            retrieved_at="2024-01-11T09:00:00+00:00",
            effective_from="2024-01-10T10:00:00+00:00",
            dataset_version="snapshot-v2",
        ),
        RawDocument(
            **common,
            retrieved_at="2024-01-12T09:00:00+00:00",
            effective_from="2024-01-10T10:00:00+00:00",
            dataset_version="snapshot-v1",
        ),
        RawDocument(
            **common,
            retrieved_at="2024-01-11T09:00:00+00:00",
            effective_from="2024-01-11T10:00:00+00:00",
            dataset_version="snapshot-v1",
        ),
    ]
    chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process(observations)
    assert len(chunks) == 4

    embedder = AlphaEmbedder.__new__(AlphaEmbedder)
    embedder.model_name = "BAAI/bge-small-en-v1.5"
    embedder.model_version = "weights-v1"
    embedder._dimension = 384
    embedder._cache = {
        embedder.cache_key(chunk): [0.01] * 384
        for chunk in chunks
    }
    records = embedder.embed_chunks(chunks)
    rows = [AlphaVectorStore._to_v2_row(record) for record in records]

    old_document_identity = (
        "canonical_source_id", "content_hash", "document_version",
        "chunk_index", "chunk_hash", "embedding_version",
    )
    document_evidence_chunk_identity = (
        "canonical_source_id", "content_hash", "document_version",
        "document_id", "evidence_id", "dataset_version", "chunk_index",
        "chunk_hash", "embedding_version",
    )
    evidence_chunk_identity = (
        "evidence_id", "dataset_version", "chunk_index", "chunk_hash",
        "embedding_version",
    )
    primary_identity = ("chunk_id",)

    def identities(columns: tuple[str, ...]) -> set[tuple[object, ...]]:
        return {tuple(row[column] for column in columns) for row in rows}

    assert len(identities(old_document_identity)) == 1
    for columns in (
        primary_identity,
        document_evidence_chunk_identity,
        evidence_chunk_identity,
    ):
        assert all(row[column] is not None for row in rows for column in columns)
        assert len(identities(columns)) == len(rows)

    amendment = _compact(_read("evidence_identity"))
    for columns in (document_evidence_chunk_identity, evidence_chunk_identity):
        assert f"unique ( {', '.join(columns)} )" in amendment

    assert len({row["document_id"] for row in rows}) == 1
    assert len({row["evidence_id"] for row in rows}) == 4
    assert len({row["chunk_id"] for row in rows}) == 4


def test_document_identity_and_temporal_columns_are_complete() -> None:
    sql = _compact(_read("documents"))
    required = (
        "chunk_id text primary key",
        "document_id text not null",
        "evidence_id text not null",
        "canonical_source_id text not null",
        "content_hash text not null",
        "chunk_hash text not null",
        "published_at timestamptz",
        "filed_at timestamptz",
        "retrieved_at timestamptz not null",
        "effective_from timestamptz",
        "effective_to timestamptz",
        "materialized_at timestamptz not null",
        "dataset_version text not null",
        "document_version text not null",
        "embedding_version text not null",
    )
    for fragment in required:
        assert fragment in sql

    assert (
        "unique ( canonical_source_id, content_hash, document_version, "
        "chunk_index, chunk_hash, embedding_version )"
    ) in sql
    assert "unique (evidence_id, dataset_version)" in sql


def test_document_migration_resolves_existing_pgvector_schema_without_relocation() -> None:
    sql = _compact(_read("documents"))
    assert "set local search_path = public, extensions" in sql
    assert "embedding vector(384)" in sql
    assert "embedding vector_cosine_ops" in sql
    assert "alter extension vector set schema" not in sql
    assert "extensions.vector" not in sql


def test_unknown_evidence_time_is_quarantined_and_never_mapped_to_now() -> None:
    sql = _compact(_read("documents"))
    assert (
        "coalesce(filed_at, published_at, effective_from) is not null "
        "or is_quarantined"
    ) in sql
    assert "not is_quarantined or quarantine_reason is not null" in sql
    assert (
        "coalesce(d.filed_at, d.published_at, d.effective_from) as available_at"
    ) in sql
    assert not re.search(
        r"coalesce\s*\(\s*(?:filed_at|published_at|effective_from)[^)]*now\s*\(",
        sql,
    )


def test_temporal_rpc_requires_identity_and_prefilters_before_ranking() -> None:
    sql = _read("documents")
    match = re.search(
        r"create or replace function public\.alpha_hybrid_search_v2\((.*?)\)"
        r"\s*returns table",
        sql,
        flags=re.DOTALL,
    )
    assert match
    signature = _compact(match.group(1))
    required_parameters = (
        "query_embedding",
        "match_ticker",
        "query_as_of",
        "query_dataset_version",
    )
    positions = [signature.index(parameter) for parameter in required_parameters]
    assert positions == sorted(positions)
    for parameter in required_parameters:
        declaration = signature.split(parameter, 1)[1].split(",", 1)[0]
        assert "default" not in declaration

    function_body = sql.split("as $function$", 1)[1].split("$function$;", 1)[0]
    filters = function_body.split("order by", 1)[0]
    assert "upper(d.ticker) = upper(match_ticker)" in filters
    assert "d.dataset_version = query_dataset_version" in filters
    assert "d.available_at <= query_as_of" in filters
    assert "d.effective_from <= query_as_of" in filters
    assert "d.effective_to > query_as_of" in filters
    assert "d.available_at >= query_as_of - query_lookback" in filters
    assert "not d.is_quarantined" not in function_body
    assert "alpha_documents_v2_eligible" in function_body


def test_forecast_tables_enforce_frozen_natural_keys() -> None:
    sql = _compact(_read("forecasts"))
    keys = (
        (
            "constraint feature_snapshots_natural_key unique ( ticker, as_of, "
            "horizon_trading_days, analysis_mode, feature_set_variant, "
            "dataset_version, feature_version )"
        ),
        (
            "constraint monthly_forecasts_natural_key unique ( ticker, as_of, "
            "target_date, horizon_trading_days, model_variant, model_version, "
            "feature_version, dataset_version, cutoff_policy, run_id )"
        ),
        (
            "constraint forecast_outcomes_natural_key unique ( ticker, as_of, "
            "target_date, horizon_trading_days, price_policy_version, benchmark, "
            "dataset_version )"
        ),
    )
    for key in keys:
        assert key in sql

    assert "run_id uuid primary key" in sql
    assert "reproducibility_hash text not null unique" in sql
    for version in (
        "universe_version",
        "cutoff_schedule_version",
        "dataset_version",
        "feature_version",
        "model_version",
        "action_policy_version",
        "code_version",
        "config_version",
    ):
        assert f"{version} text not null" in sql


def test_forecasts_are_immutable_versioned_and_rls_protected() -> None:
    sql = _compact(_read("forecasts"))
    for relation in (
        "feature_snapshots",
        "monthly_forecasts",
        "forecast_outcomes",
        "backtest_runs",
    ):
        assert f"alter table public.{relation} enable row level security" in sql
    assert "reject_immutable_forecast_change" in sql
    assert "feature_snapshots_immutable" in sql
    assert "monthly_forecasts_immutable" in sql
    assert "forecast_outcomes_immutable" in sql
    assert "protect_backtest_run_identity" in sql
    assert "evidence_ids text[]" in sql


def test_security_invoker_views_and_explicit_read_only_grants() -> None:
    documents = _compact(_read("documents"))
    forecasts = _compact(_read("forecasts"))
    analyses = _compact(_read("analyses"))

    assert (
        "create or replace view public.alpha_documents_v2_eligible "
        "with (security_invoker = true)"
    ) in documents
    assert (
        "grant select on table public.alpha_documents_v2 "
        "to authenticated, service_role"
    ) in documents
    assert (
        "grant select on public.alpha_documents_v2_eligible "
        "to authenticated, service_role"
    ) in documents

    for relation in (
        "feature_snapshots",
        "monthly_forecasts",
        "forecast_outcomes",
        "backtest_runs",
    ):
        assert (
            f"grant select on table public.{relation} "
            "to authenticated, service_role"
        ) in forecasts

    assert (
        "create or replace view public.forecast_v2_reconciliation "
        "with (security_invoker = true)"
    ) in analyses
    assert (
        "grant select on table public.forecast_schema_routes "
        "to authenticated, service_role"
    ) in analyses

    combined = " ".join((documents, forecasts, analyses))
    assert not re.search(
        r"grant\s+(?:insert|update|delete|truncate|all)\b",
        combined,
    )


def test_analysis_versioning_is_nullable_additive_and_not_ticker_only() -> None:
    raw = _read("analyses")
    sql = _compact(raw)
    alter = sql.split("alter table public.analyses", 1)[1].split(";", 1)[0]
    expected = (
        "as_of timestamptz",
        "target_date date",
        "horizon_trading_days integer",
        "analysis_mode text",
        "schema_version text",
        "dataset_version text",
        "feature_version text",
        "model_version text",
        "prompt_version text",
        "run_id uuid",
        "evidence_ids text[]",
    )
    for declaration in expected:
        assert f"add column if not exists {declaration}" in alter
    assert "not null" not in alter

    identity = sql.split("analyses_monthly_identity_uidx", 1)[1].split(
        "where", 1
    )[0]
    for field in (
        "ticker",
        "as_of",
        "horizon_trading_days",
        "analysis_mode",
        "schema_version",
    ):
        assert field in identity
    assert "(ticker)" not in identity


def test_routes_remain_legacy_until_authorized_reconciliation() -> None:
    sql = _compact(_read("analyses"))
    for component in (
        "vector_documents",
        "forecast_storage",
        "analysis_storage",
    ):
        assert f"'{component}', 'legacy-v1', null, null, null" in sql
    assert "approved_reconciliation_hash" in sql
    assert "rollback_forecast_schema_route" in sql
    assert "create or replace view public.forecast_v2_reconciliation" in sql
    assert "defining route functions does not authorize calling them" in sql


def test_supabase_reconciliation_has_ordered_hashes_for_every_v2_relation() -> None:
    documents = _compact(_read("documents"))
    analyses = _compact(_read("analyses"))

    assert "extensions.digest(" in documents
    assert "'|' order by chunk_id" in documents
    assert "as reconciliation_hash" in documents

    for relation in (
        "alpha_documents_v2",
        "feature_snapshots",
        "monthly_forecasts",
        "forecast_outcomes",
        "backtest_runs",
        "analyses_v2",
    ):
        assert f"'{relation}'" in analyses
    assert analyses.count("extensions.digest(") >= 6
    assert analyses.count("'|' order by") >= 5
    assert "reconciliation_hash" in analyses


def test_neo4j_schema_uses_immutable_observations_and_temporal_indexes() -> None:
    cypher = _read("graph")
    for label, identity in (
        ("alphadocumentv2", "document_id"),
        ("alphachunkv2", "chunk_id"),
        ("evidenceobservationv2", "evidence_id"),
        ("temporalentityv2", "entity_id"),
    ):
        assert f"for (" in cypher
        assert f":{label})" in cypher
        assert f".{identity} is unique" in cypher

    for field in (
        "ticker",
        "dataset_version",
        "published_at",
        "filed_at",
        "effective_from",
        "effective_to",
        "is_quarantined",
    ):
        assert field in cypher
    assert "route.active_version = 'legacy-v1'" in cypher
    assert "observes_chunk" in cypher
    assert "merge (a)-[r:" not in cypher


def test_neo4j_reconciliation_returns_counts_and_canonical_digest_input() -> None:
    cypher = _compact(_read("graph"))
    for relation in (
        "alphadocumentv2",
        "alphachunkv2",
        "evidenceobservationv2",
    ):
        assert f"'{relation}' as relation_name" in cypher
    assert cypher.count("union all") >= 2
    assert "order by identity_row" in cypher
    assert "size(identity_rows) as row_count" in cypher
    assert "as quarantined_count" in cypher
    assert "as unknown_time_count" in cypher
    assert "'sha-256' as reconciliation_algorithm" in cypher
    assert "as ordered_identity_digest_input" in cypher


def test_vector_and_graph_share_document_chunk_and_evidence_identities() -> None:
    sql = _read("documents")
    cypher = _read("graph")
    for identity in ("document_id", "chunk_id", "evidence_id"):
        assert identity in sql
        assert identity in cypher
