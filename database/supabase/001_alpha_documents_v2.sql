-- Stage 02 / Database V2
-- Additive shadow schema for point-in-time documents and chunks.
-- This migration intentionally does not switch any production read/write path.

begin;

create schema if not exists extensions;
create extension if not exists pgcrypto with schema extensions;
create extension if not exists vector with schema extensions;

-- Supabase projects created at different times may already have pgvector in
-- either public or extensions. Resolve the existing type without relocating
-- the extension (which could break legacy SQL that schema-qualifies it).
set local search_path = public, extensions;

create table if not exists public.alpha_documents_v2 (
    chunk_id text primary key,
    document_id text not null,
    evidence_id text not null,
    canonical_source_id text not null,
    source_url text,
    source_type text not null,
    source_version text not null,
    content_hash text not null,
    chunk_hash text not null,
    chunk_index integer not null,
    ticker text not null,
    document_text text not null,
    embedding vector(384),
    embedding_model text not null,
    embedding_version text not null,
    published_at timestamptz,
    filed_at timestamptz,
    retrieved_at timestamptz not null,
    effective_from timestamptz,
    effective_to timestamptz,
    materialized_at timestamptz not null default now(),
    dataset_version text not null,
    document_version text not null,
    metadata jsonb not null default '{}'::jsonb,
    is_quarantined boolean not null default true,
    quarantine_reason text,
    constraint alpha_documents_v2_chunk_index_nonnegative
        check (chunk_index >= 0),
    constraint alpha_documents_v2_document_id_shape
        check (document_id ~ '^[a-f0-9]{64}$'),
    constraint alpha_documents_v2_chunk_id_shape
        check (chunk_id ~ '^[a-f0-9]{64}$'),
    constraint alpha_documents_v2_evidence_id_shape
        check (evidence_id ~ '^[a-f0-9]{64}$'),
    constraint alpha_documents_v2_content_hash_shape
        check (content_hash ~ '^[a-f0-9]{64}$'),
    constraint alpha_documents_v2_chunk_hash_shape
        check (chunk_hash ~ '^[a-f0-9]{64}$'),
    constraint alpha_documents_v2_effective_interval
        check (effective_to is null or (
            effective_from is not null and effective_to > effective_from
        )),
    constraint alpha_documents_v2_unknown_time_quarantine
        check (
            coalesce(filed_at, published_at, effective_from) is not null
            or is_quarantined
        ),
    constraint alpha_documents_v2_quarantine_reason
        check (not is_quarantined or quarantine_reason is not null),
    constraint alpha_documents_v2_document_identity
        unique (
            canonical_source_id,
            content_hash,
            document_version,
            chunk_index,
            chunk_hash,
            embedding_version
        ),
    constraint alpha_documents_v2_evidence_identity
        unique (evidence_id, dataset_version)
);

comment on table public.alpha_documents_v2 is
    'V2 shadow document/chunk store. Unknown evidence time remains quarantined.';
comment on column public.alpha_documents_v2.document_id is
    'SHA-256 of canonical source identity, content hash, and document version.';
comment on column public.alpha_documents_v2.chunk_id is
    'SHA-256 shared by Vector and Graph for this immutable chunk version.';
comment on column public.alpha_documents_v2.evidence_id is
    'Immutable evidence observation identity shared by Vector and Graph.';
comment on column public.alpha_documents_v2.retrieved_at is
    'Acquisition time; never substituted for publication, filing, or effective time.';

create index if not exists alpha_documents_v2_temporal_lookup_idx
    on public.alpha_documents_v2 (
        ticker,
        dataset_version,
        published_at desc,
        filed_at desc
    );

create index if not exists alpha_documents_v2_effective_interval_idx
    on public.alpha_documents_v2 (
        ticker,
        dataset_version,
        effective_from,
        effective_to
    );

create index if not exists alpha_documents_v2_source_identity_idx
    on public.alpha_documents_v2 (
        canonical_source_id,
        content_hash,
        document_version
    );

create index if not exists alpha_documents_v2_metadata_gin_idx
    on public.alpha_documents_v2 using gin (metadata);

create index if not exists alpha_documents_v2_embedding_hnsw_idx
    on public.alpha_documents_v2
    using hnsw (embedding vector_cosine_ops)
    where embedding is not null and not is_quarantined;

alter table public.alpha_documents_v2 enable row level security;

do $policy$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'alpha_documents_v2'
          and policyname = 'alpha_documents_v2_authenticated_read'
    ) then
        create policy alpha_documents_v2_authenticated_read
            on public.alpha_documents_v2
            for select
            to authenticated
            using (not is_quarantined);
    end if;
end
$policy$;

create or replace view public.alpha_documents_v2_eligible
with (security_invoker = true) as
select
    d.*,
    coalesce(d.filed_at, d.published_at, d.effective_from) as available_at
from public.alpha_documents_v2 as d
where not d.is_quarantined
  and coalesce(d.filed_at, d.published_at, d.effective_from) is not null;

comment on view public.alpha_documents_v2_eligible is
    'Eligible shadow rows only. Callers must still enforce ticker, dataset version, as_of, and effective interval.';

grant select on table public.alpha_documents_v2
    to authenticated, service_role;
grant select on public.alpha_documents_v2_eligible
    to authenticated, service_role;

create or replace function public.alpha_hybrid_search_v2(
    query_embedding vector(384),
    match_ticker text,
    query_as_of timestamptz,
    query_dataset_version text,
    query_lookback interval default null,
    match_count integer default 20,
    similarity_threshold double precision default 0.0
)
returns table (
    chunk_id text,
    document_id text,
    evidence_id text,
    ticker text,
    document_text text,
    metadata jsonb,
    published_at timestamptz,
    filed_at timestamptz,
    effective_from timestamptz,
    effective_to timestamptz,
    dataset_version text,
    embedding_version text,
    similarity double precision
)
language plpgsql
stable
security invoker
set search_path = public, extensions
as $function$
begin
    if query_embedding is null then
        raise exception 'query_embedding is required';
    end if;
    if match_ticker is null or btrim(match_ticker) = '' then
        raise exception 'match_ticker is required';
    end if;
    if query_as_of is null then
        raise exception 'query_as_of is required';
    end if;
    if query_dataset_version is null or btrim(query_dataset_version) = '' then
        raise exception 'query_dataset_version is required';
    end if;
    if match_count < 1 or match_count > 200 then
        raise exception 'match_count must be between 1 and 200';
    end if;
    if query_lookback is not null and query_lookback <= interval '0 seconds' then
        raise exception 'query_lookback must be positive';
    end if;

    return query
    select
        d.chunk_id,
        d.document_id,
        d.evidence_id,
        d.ticker,
        d.document_text,
        d.metadata,
        d.published_at,
        d.filed_at,
        d.effective_from,
        d.effective_to,
        d.dataset_version,
        d.embedding_version,
        (1.0 - (d.embedding <=> query_embedding))::double precision as similarity
    from public.alpha_documents_v2_eligible as d
    where d.embedding is not null
      and upper(d.ticker) = upper(match_ticker)
      and d.dataset_version = query_dataset_version
      and d.available_at <= query_as_of
      and (d.effective_from is null or d.effective_from <= query_as_of)
      and (d.effective_to is null or d.effective_to > query_as_of)
      and (
          query_lookback is null
          or d.available_at >= query_as_of - query_lookback
      )
      and (1.0 - (d.embedding <=> query_embedding)) >= similarity_threshold
    order by d.embedding <=> query_embedding, d.chunk_id
    limit match_count;
end
$function$;

revoke all on function public.alpha_hybrid_search_v2(
    vector,
    text,
    timestamptz,
    text,
    interval,
    integer,
    double precision
) from public;

grant execute on function public.alpha_hybrid_search_v2(
    vector,
    text,
    timestamptz,
    text,
    interval,
    integer,
    double precision
) to authenticated, service_role;

create or replace view public.alpha_documents_v2_reconciliation as
select
    dataset_version,
    count(*) as row_count,
    count(distinct document_id) as document_count,
    count(distinct chunk_id) as chunk_count,
    count(*) filter (where is_quarantined) as quarantined_count,
    count(*) filter (
        where coalesce(filed_at, published_at, effective_from) is null
    ) as unknown_evidence_time_count,
    encode(
        extensions.digest(
            coalesce(
                string_agg(
                    concat_ws(
                        ':',
                        document_id,
                        chunk_id,
                        evidence_id,
                        content_hash,
                        chunk_hash,
                        document_version,
                        embedding_version,
                        dataset_version
                    ),
                    '|' order by chunk_id
                ),
                ''
            ),
            'sha256'
        ),
        'hex'
    ) as reconciliation_hash,
    min(materialized_at) as first_materialized_at,
    max(materialized_at) as last_materialized_at
from public.alpha_documents_v2
group by dataset_version;

-- Rollback discipline:
-- Keep production routes on legacy-v1 until reconciliation is approved.
-- A later authorized switch changes only the route pointer defined in
-- 003_analysis_versioning.sql. Rollback restores that pointer; V2 rows remain
-- intact for audit and are never implicitly removed.

commit;
