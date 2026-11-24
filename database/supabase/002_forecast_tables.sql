-- Stage 02 / Database V2
-- Additive, versioned forecast storage. Forecasts and outcomes are immutable.

begin;

create table if not exists public.feature_snapshots (
    snapshot_id uuid primary key default extensions.gen_random_uuid(),
    ticker text not null,
    as_of timestamptz not null,
    horizon_trading_days integer not null,
    analysis_mode text not null,
    feature_set_variant text not null,
    dataset_version text not null,
    feature_version text not null,
    schema_version text not null,
    feature_values jsonb not null,
    evidence_ids text[] not null default array[]::text[],
    materialized_at timestamptz not null default now(),
    constraint feature_snapshots_horizon_positive
        check (horizon_trading_days > 0),
    constraint feature_snapshots_analysis_mode
        check (analysis_mode in ('LIVE', 'HISTORICAL_BACKTEST')),
    constraint feature_snapshots_values_object
        check (jsonb_typeof(feature_values) = 'object'),
    constraint feature_snapshots_natural_key
        unique (
            ticker,
            as_of,
            horizon_trading_days,
            analysis_mode,
            feature_set_variant,
            dataset_version,
            feature_version
        )
);

create table if not exists public.monthly_forecasts (
    forecast_id uuid primary key default extensions.gen_random_uuid(),
    ticker text not null,
    as_of timestamptz not null,
    target_date date not null,
    horizon_trading_days integer not null,
    benchmark text not null,
    cutoff_policy text not null default 'MONTH_END',
    model_variant text not null,
    model_version text not null,
    feature_version text not null,
    dataset_version text not null,
    schema_version text not null,
    action_policy_version text not null,
    run_id uuid not null,
    feature_snapshot_id uuid,
    predicted_excess_return numeric,
    predicted_ticker_return numeric,
    predicted_benchmark_return numeric,
    uncertainty_lower numeric,
    uncertainty_upper numeric,
    action text,
    evidence_ids text[] not null default array[]::text[],
    forecast_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint monthly_forecasts_horizon_positive
        check (horizon_trading_days > 0),
    constraint monthly_forecasts_cutoff_policy_valid
        check (cutoff_policy in ('MONTH_END', 'MONTH_TO_DATE')),
    constraint monthly_forecasts_target_after_as_of
        check (target_date > as_of::date),
    constraint monthly_forecasts_uncertainty_order
        check (
            uncertainty_lower is null
            or uncertainty_upper is null
            or uncertainty_lower <= uncertainty_upper
        ),
    constraint monthly_forecasts_feature_snapshot_fk
        foreign key (feature_snapshot_id)
        references public.feature_snapshots(snapshot_id),
    constraint monthly_forecasts_natural_key
        unique (
            ticker,
            as_of,
            target_date,
            horizon_trading_days,
            model_variant,
            model_version,
            feature_version,
            dataset_version,
            cutoff_policy,
            run_id
        )
);

create table if not exists public.forecast_outcomes (
    outcome_id uuid primary key default extensions.gen_random_uuid(),
    ticker text not null,
    as_of timestamptz not null,
    target_date date not null,
    horizon_trading_days integer not null,
    price_policy_version text not null,
    benchmark text not null,
    cutoff_policy text not null default 'MONTH_END',
    dataset_version text not null,
    ticker_start_adjusted_close numeric not null,
    ticker_end_adjusted_close numeric not null,
    benchmark_start_adjusted_close numeric not null,
    benchmark_end_adjusted_close numeric not null,
    ticker_total_return numeric not null,
    benchmark_total_return numeric not null,
    excess_return numeric not null,
    source_snapshot_ids text[] not null,
    observable_at timestamptz not null,
    materialized_at timestamptz not null default now(),
    constraint forecast_outcomes_horizon_positive
        check (horizon_trading_days > 0),
    constraint forecast_outcomes_prices_positive
        check (
            ticker_start_adjusted_close > 0
            and ticker_end_adjusted_close > 0
            and benchmark_start_adjusted_close > 0
            and benchmark_end_adjusted_close > 0
        ),
    constraint forecast_outcomes_observable_before_materialization
        check (materialized_at >= observable_at),
    constraint forecast_outcomes_natural_key
        unique (
            ticker,
            as_of,
            target_date,
            horizon_trading_days,
            price_policy_version,
            benchmark,
            dataset_version
        )
);

create table if not exists public.backtest_runs (
    run_id uuid primary key,
    reproducibility_hash text not null unique,
    universe_version text not null,
    cutoff_schedule_version text not null,
    fold_definition jsonb not null,
    transaction_cost_policy jsonb not null,
    dataset_version text not null,
    feature_version text not null,
    model_variant text not null,
    model_version text not null,
    action_policy_version text not null,
    code_version text not null,
    config_version text not null,
    status text not null,
    metrics jsonb,
    artifact_refs jsonb not null default '{}'::jsonb,
    started_at timestamptz not null,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    constraint backtest_runs_reproducibility_hash_shape
        check (reproducibility_hash ~ '^[a-f0-9]{64}$'),
    constraint backtest_runs_status
        check (status in ('PLANNED', 'RUNNING', 'COMPLETED', 'FAILED', 'BLOCKED')),
    constraint backtest_runs_completion_order
        check (completed_at is null or completed_at >= started_at)
);

create index if not exists feature_snapshots_lookup_idx
    on public.feature_snapshots (
        ticker,
        as_of desc,
        dataset_version,
        feature_version
    );

create index if not exists monthly_forecasts_lookup_idx
    on public.monthly_forecasts (
        ticker,
        as_of desc,
        target_date,
        dataset_version,
        model_version
    );

create index if not exists forecast_outcomes_lookup_idx
    on public.forecast_outcomes (
        ticker,
        as_of desc,
        target_date,
        dataset_version
    );

create index if not exists backtest_runs_version_lookup_idx
    on public.backtest_runs (
        dataset_version,
        feature_version,
        model_version,
        action_policy_version
    );

create or replace function public.reject_immutable_forecast_change()
returns trigger
language plpgsql
security invoker
set search_path = public
as $function$
begin
    raise exception '% is immutable; write a new versioned row instead', tg_table_name;
end
$function$;

do $trigger$
begin
    if not exists (
        select 1 from pg_trigger
        where tgname = 'feature_snapshots_immutable'
          and tgrelid = 'public.feature_snapshots'::regclass
    ) then
        create trigger feature_snapshots_immutable
        before update or delete on public.feature_snapshots
        for each row execute function public.reject_immutable_forecast_change();
    end if;

    if not exists (
        select 1 from pg_trigger
        where tgname = 'monthly_forecasts_immutable'
          and tgrelid = 'public.monthly_forecasts'::regclass
    ) then
        create trigger monthly_forecasts_immutable
        before update or delete on public.monthly_forecasts
        for each row execute function public.reject_immutable_forecast_change();
    end if;

    if not exists (
        select 1 from pg_trigger
        where tgname = 'forecast_outcomes_immutable'
          and tgrelid = 'public.forecast_outcomes'::regclass
    ) then
        create trigger forecast_outcomes_immutable
        before update or delete on public.forecast_outcomes
        for each row execute function public.reject_immutable_forecast_change();
    end if;
end
$trigger$;

create or replace function public.protect_backtest_run_identity()
returns trigger
language plpgsql
security invoker
set search_path = public
as $function$
begin
    if new.run_id is distinct from old.run_id
       or new.reproducibility_hash is distinct from old.reproducibility_hash
       or new.universe_version is distinct from old.universe_version
       or new.cutoff_schedule_version is distinct from old.cutoff_schedule_version
       or new.fold_definition is distinct from old.fold_definition
       or new.transaction_cost_policy is distinct from old.transaction_cost_policy
       or new.dataset_version is distinct from old.dataset_version
       or new.feature_version is distinct from old.feature_version
       or new.model_variant is distinct from old.model_variant
       or new.model_version is distinct from old.model_version
       or new.action_policy_version is distinct from old.action_policy_version
       or new.code_version is distinct from old.code_version
       or new.config_version is distinct from old.config_version
    then
        raise exception 'backtest run identity and reproducibility inputs are immutable';
    end if;
    return new;
end
$function$;

do $trigger$
begin
    if not exists (
        select 1 from pg_trigger
        where tgname = 'backtest_runs_identity_immutable'
          and tgrelid = 'public.backtest_runs'::regclass
    ) then
        create trigger backtest_runs_identity_immutable
        before update on public.backtest_runs
        for each row execute function public.protect_backtest_run_identity();
    end if;
end
$trigger$;

alter table public.feature_snapshots enable row level security;
alter table public.monthly_forecasts enable row level security;
alter table public.forecast_outcomes enable row level security;
alter table public.backtest_runs enable row level security;

grant select on table public.feature_snapshots
    to authenticated, service_role;
grant select on table public.monthly_forecasts
    to authenticated, service_role;
grant select on table public.forecast_outcomes
    to authenticated, service_role;
grant select on table public.backtest_runs
    to authenticated, service_role;

do $policy$
declare
    table_name text;
    policy_name text;
begin
    foreach table_name in array array[
        'feature_snapshots',
        'monthly_forecasts',
        'forecast_outcomes',
        'backtest_runs'
    ]
    loop
        policy_name := table_name || '_authenticated_read';
        if not exists (
            select 1
            from pg_policies
            where schemaname = 'public'
              and tablename = table_name
              and policyname = policy_name
        ) then
            execute format(
                'create policy %I on public.%I for select to authenticated using (true)',
                policy_name,
                table_name
            );
        end if;
    end loop;
end
$policy$;

comment on table public.feature_snapshots is
    'Immutable point-in-time feature payloads with compact evidence references.';
comment on table public.monthly_forecasts is
    'Immutable forecasts; model and ablation variants coexist by natural key.';
comment on table public.forecast_outcomes is
    'Model-independent realized labels materialized only when observable.';
comment on table public.backtest_runs is
    'Reproducible run identity is immutable; status and metrics may advance.';

commit;
