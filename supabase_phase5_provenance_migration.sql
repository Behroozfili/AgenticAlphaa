-- Additive legacy RAG provenance fields and filtered hybrid retrieval.
-- Existing rows remain valid because every added provenance field is nullable.

begin;

create extension if not exists vector;

alter table public.alpha_documents
    add column if not exists fiscal_year integer,
    add column if not exists fiscal_quarter text,
    add column if not exists period_start date,
    add column if not exists period_end date,
    add column if not exists duration_months integer,
    add column if not exists filing_type text,
    add column if not exists accession_number text,
    add column if not exists metric_scope text,
    add column if not exists gaap_status text;

create index if not exists alpha_documents_ticker_published_idx
    on public.alpha_documents (ticker, published_at desc);

create index if not exists alpha_documents_financial_period_idx
    on public.alpha_documents (ticker, fiscal_year, fiscal_quarter, metric_scope, period_end);

create or replace function public.alpha_hybrid_search(
    query_embedding vector(384),
    query_text text,
    filter_ticker text default null,
    days_back integer default null,
    top_k integer default 50,
    rrf_k integer default 60,
    page_offset integer default 0,
    filter_fiscal_year integer default null,
    filter_fiscal_quarter text default null,
    filter_period_end date default null,
    filter_metric_scope text default null
)
returns table (
    id bigint,
    content_hash text,
    url_hash text,
    ticker text,
    source_type text,
    published_at timestamptz,
    ingested_at timestamptz,
    chunk_index integer,
    url text,
    title text,
    fiscal_year integer,
    fiscal_quarter text,
    period_start date,
    period_end date,
    duration_months integer,
    filing_type text,
    accession_number text,
    metric_scope text,
    gaap_status text,
    text text,
    rrf_score double precision
)
language sql
stable
security invoker
set search_path = public
as $function$
with vector_candidates as (
    select
        d.id,
        row_number() over (
            order by d.embedding <=> query_embedding, d.id
        ) as candidate_rank
    from public.alpha_documents as d
    where d.embedding is not null
      and (filter_ticker is null or upper(d.ticker) = upper(filter_ticker))
      and (
          days_back is null
          or d.published_at >= now() - make_interval(days => days_back)
      )
      and (filter_fiscal_year is null or d.fiscal_year = filter_fiscal_year)
      and (filter_fiscal_quarter is null or d.fiscal_quarter = filter_fiscal_quarter)
      and (filter_period_end is null or d.period_end = filter_period_end)
      and (filter_metric_scope is null or d.metric_scope = filter_metric_scope)
    order by d.embedding <=> query_embedding, d.id
    limit greatest(1, top_k)
),
fts_candidates as (
    select
        d.id,
        row_number() over (
            order by
                ts_rank_cd(
                    d.fts,
                    websearch_to_tsquery('english', coalesce(query_text, ''))
                ) desc,
                d.id
        ) as candidate_rank
    from public.alpha_documents as d
    where d.fts @@ websearch_to_tsquery('english', coalesce(query_text, ''))
      and (filter_ticker is null or upper(d.ticker) = upper(filter_ticker))
      and (
          days_back is null
          or d.published_at >= now() - make_interval(days => days_back)
      )
      and (filter_fiscal_year is null or d.fiscal_year = filter_fiscal_year)
      and (filter_fiscal_quarter is null or d.fiscal_quarter = filter_fiscal_quarter)
      and (filter_period_end is null or d.period_end = filter_period_end)
      and (filter_metric_scope is null or d.metric_scope = filter_metric_scope)
    order by
        ts_rank_cd(
            d.fts,
            websearch_to_tsquery('english', coalesce(query_text, ''))
        ) desc,
        d.id
    limit greatest(1, top_k)
),
candidate_ids as (
    select id from vector_candidates
    union
    select id from fts_candidates
)
select
    d.id,
    d.content_hash,
    d.url_hash,
    d.ticker,
    d.source_type,
    d.published_at,
    d.ingested_at,
    d.chunk_index,
    d.url,
    d.title,
    d.fiscal_year,
    d.fiscal_quarter,
    d.period_start,
    d.period_end,
    d.duration_months,
    d.filing_type,
    d.accession_number,
    d.metric_scope,
    d.gaap_status,
    d.text,
    (
        coalesce(
            1.0 / (greatest(1, rrf_k) + vector_candidates.candidate_rank),
            0.0
        )
        +
        coalesce(
            1.0 / (greatest(1, rrf_k) + fts_candidates.candidate_rank),
            0.0
        )
    )::double precision as rrf_score
from candidate_ids
join public.alpha_documents as d using (id)
left join vector_candidates using (id)
left join fts_candidates using (id)
order by rrf_score desc, d.id
limit greatest(1, top_k)
offset greatest(0, page_offset);
$function$;

revoke all on function public.alpha_hybrid_search(
    vector,
    text,
    text,
    integer,
    integer,
    integer,
    integer,
    integer,
    text,
    date,
    text
) from public;

grant execute on function public.alpha_hybrid_search(
    vector,
    text,
    text,
    integer,
    integer,
    integer,
    integer,
    integer,
    text,
    date,
    text
