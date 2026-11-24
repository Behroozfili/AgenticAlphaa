-- Stage 02 / Database V2
-- Additive analysis identity/version columns and explicit schema route pointers.

begin;

create table if not exists public.analyses (
    analysis_id text primary key,
    user_id text,
    query text,
    ticker text,
    status text,
    final_report text,
    financial_metrics jsonb not null default '{}'::jsonb,
    sentiment_analysis jsonb not null default '{}'::jsonb,
    agent_execution_history jsonb not null default '[]'::jsonb,
    orchestrator_logs jsonb not null default '[]'::jsonb,
    error_message text,
    duration_s double precision,
    created_at timestamptz,
    completed_at timestamptz
);

alter table public.analyses
    add column if not exists as_of timestamptz,
    add column if not exists target_date date,
    add column if not exists horizon_trading_days integer,
    add column if not exists analysis_mode text,
    add column if not exists schema_version text,
    add column if not exists dataset_version text,
    add column if not exists feature_version text,
    add column if not exists model_version text,
    add column if not exists prompt_version text,
    add column if not exists run_id uuid,
    add column if not exists evidence_ids text[];

comment on column public.analyses.as_of is
    'Point-in-time cutoff. Null identifies an untouched legacy analysis.';
comment on column public.analyses.dataset_version is
    'Null for untouched legacy rows; legacy timestamps are never inferred.';

create unique index if not exists analyses_monthly_identity_uidx
    on public.analyses (
        ticker,
        as_of,
        horizon_trading_days,
        analysis_mode,
        schema_version
    )
    where ticker is not null
      and as_of is not null
      and horizon_trading_days is not null
      and analysis_mode is not null
      and schema_version is not null;

create index if not exists analyses_version_lookup_idx
    on public.analyses (
        ticker,
        as_of desc,
        dataset_version,
        feature_version,
        model_version
    )
    where as_of is not null;

create or replace view public.analyses_v2
with (security_invoker = true) as
select *
from public.analyses
where as_of is not null
  and horizon_trading_days is not null
  and analysis_mode is not null
  and schema_version is not null
  and dataset_version is not null
  and feature_version is not null
  and model_version is not null;

create table if not exists public.forecast_schema_routes (
    component text primary key,
    active_version text not null,
    previous_version text,
    reconciliation_hash text,
    activated_at timestamptz,
    updated_at timestamptz not null default now(),
    constraint forecast_schema_routes_version_nonempty
        check (btrim(active_version) <> ''),
    constraint forecast_schema_routes_reconciliation_hash_shape
        check (
            reconciliation_hash is null
            or reconciliation_hash ~ '^[a-f0-9]{64}$'
        )
);

insert into public.forecast_schema_routes (
    component,
    active_version,
    previous_version,
    reconciliation_hash,
    activated_at
)
values
    ('vector_documents', 'legacy-v1', null, null, null),
    ('forecast_storage', 'legacy-v1', null, null, null),
    ('analysis_storage', 'legacy-v1', null, null, null)
on conflict (component) do nothing;

create or replace function public.activate_forecast_schema_route(
    route_component text,
    target_version text,
    approved_reconciliation_hash text
)
returns void
language plpgsql
security definer
set search_path = public
as $function$
begin
    if route_component is null or btrim(route_component) = '' then
        raise exception 'route_component is required';
    end if;
    if target_version is null or btrim(target_version) = '' then
        raise exception 'target_version is required';
    end if;
    if approved_reconciliation_hash is null
       or approved_reconciliation_hash !~ '^[a-f0-9]{64}$'
    then
        raise exception 'an approved reconciliation SHA-256 is required';
    end if;

    update public.forecast_schema_routes
    set previous_version = active_version,
        active_version = target_version,
        reconciliation_hash = approved_reconciliation_hash,
        activated_at = now(),
        updated_at = now()
    where component = route_component;

    if not found then
        raise exception 'unknown schema route component: %', route_component;
    end if;
end
$function$;

create or replace function public.rollback_forecast_schema_route(
    route_component text
)
returns void
language plpgsql
security definer
set search_path = public
as $function$
begin
    update public.forecast_schema_routes
    set active_version = previous_version,
        previous_version = active_version,
        activated_at = now(),
        updated_at = now()
    where component = route_component
      and previous_version is not null;

    if not found then
        raise exception 'no rollback pointer exists for component: %', route_component;
    end if;
end
$function$;

revoke all on function public.activate_forecast_schema_route(
    text,
    text,
    text
) from public, authenticated;
revoke all on function public.rollback_forecast_schema_route(text)
    from public, authenticated;

create or replace view public.forecast_v2_reconciliation
with (security_invoker = true) as
select
    'alpha_documents_v2'::text as relation_name,
    dataset_version,
    row_count,
    quarantined_count,
    unknown_evidence_time_count as unknown_time_count,
    reconciliation_hash
from public.alpha_documents_v2_reconciliation
union all
select
    'feature_snapshots',
    dataset_version,
    count(*)::bigint,
    0::bigint,
    0::bigint,
    encode(
        extensions.digest(
            coalesce(
                string_agg(
                    concat_ws(
                        ':',
                        ticker,
                        extract(epoch from as_of)::text,
                        horizon_trading_days::text,
                        analysis_mode,
                        feature_set_variant,
                        dataset_version,
                        feature_version,
                        schema_version,
                        feature_values::text,
                        array_to_string(evidence_ids, ',')
                    ),
                    '|' order by
                        ticker,
                        as_of,
                        horizon_trading_days,
                        analysis_mode,
                        feature_set_variant,
                        feature_version
                ),
                ''
            ),
            'sha256'
        ),
        'hex'
    )
from public.feature_snapshots
group by dataset_version
union all
select
    'monthly_forecasts',
    dataset_version,
    count(*)::bigint,
    0::bigint,
    0::bigint,
    encode(
        extensions.digest(
            coalesce(
                string_agg(
                    concat_ws(
                        ':',
                        ticker,
                        extract(epoch from as_of)::text,
                        target_date::text,
                        horizon_trading_days::text,
                        model_variant,
                        model_version,
                        feature_version,
                        dataset_version,
                        run_id::text,
                        coalesce(predicted_excess_return::text, ''),
                        forecast_payload::text
                    ),
                    '|' order by
                        ticker,
                        as_of,
                        target_date,
                        horizon_trading_days,
                        model_variant,
                        model_version,
                        feature_version,
                        run_id
                ),
                ''
            ),
            'sha256'
        ),
        'hex'
    )
from public.monthly_forecasts
group by dataset_version
union all
select
    'forecast_outcomes',
    dataset_version,
    count(*)::bigint,
    0::bigint,
    0::bigint,
    encode(
        extensions.digest(
            coalesce(
                string_agg(
                    concat_ws(
                        ':',
                        ticker,
                        extract(epoch from as_of)::text,
                        target_date::text,
                        horizon_trading_days::text,
                        price_policy_version,
                        benchmark,
                        dataset_version,
                        ticker_total_return::text,
                        benchmark_total_return::text,
                        excess_return::text,
                        array_to_string(source_snapshot_ids, ',')
                    ),
                    '|' order by
                        ticker,
                        as_of,
                        target_date,
                        horizon_trading_days,
                        price_policy_version,
                        benchmark
                ),
                ''
            ),
            'sha256'
        ),
        'hex'
    )
from public.forecast_outcomes
group by dataset_version
union all
select
    'backtest_runs',
    dataset_version,
    count(*)::bigint,
    0::bigint,
    0::bigint,
    encode(
        extensions.digest(
            coalesce(
                string_agg(
                    concat_ws(
                        ':',
                        reproducibility_hash,
                        status,
                        coalesce(metrics::text, ''),
                        artifact_refs::text
                    ),
                    '|' order by reproducibility_hash
                ),
                ''
            ),
            'sha256'
        ),
        'hex'
    )
from public.backtest_runs
group by dataset_version
union all
select
    'analyses_v2',
    dataset_version,
    count(*)::bigint,
    0::bigint,
    0::bigint,
    encode(
        extensions.digest(
            coalesce(
                string_agg(
                    concat_ws(
                        ':',
                        ticker,
                        extract(epoch from as_of)::text,
                        horizon_trading_days::text,
                        analysis_mode,
                        schema_version,
                        dataset_version,
                        feature_version,
                        model_version,
                        coalesce(prompt_version, ''),
                        encode(
                            extensions.digest(
                                coalesce(final_report, ''),
                                'sha256'
                            ),
                            'hex'
                        )
                    ),
                    '|' order by
                        ticker,
                        as_of,
                        horizon_trading_days,
                        analysis_mode,
                        schema_version
                ),
                ''
            ),
            'sha256'
        ),
        'hex'
    )
from public.analyses_v2
group by dataset_version;

alter table public.forecast_schema_routes enable row level security;

grant select on table public.forecast_schema_routes
    to authenticated, service_role;
grant select on public.forecast_v2_reconciliation
    to service_role;

do $policy$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'forecast_schema_routes'
          and policyname = 'forecast_schema_routes_authenticated_read'
    ) then
        create policy forecast_schema_routes_authenticated_read
            on public.forecast_schema_routes
            for select
            to authenticated
            using (true);
    end if;
end
$policy$;

-- Safety gate:
-- Defining route functions does not authorize calling them. All routes remain
-- legacy-v1 after this migration. A route switch requires separate user
-- authorization plus an approved reconciliation hash. Rollback changes only
-- the pointer and preserves every legacy and V2 row.

commit;
