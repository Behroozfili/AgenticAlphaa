-- Atomic monthly persistence. Apply after migrations 001 through 005.
-- No data is removed. Existing run identifiers are never silently reinterpreted.
begin;
-- Each new analysis belongs to one run; retain historical rows unchanged.
drop index if exists public.analyses_monthly_identity_uidx;
create index if not exists analyses_monthly_context_idx on public.analyses
    (ticker, as_of, horizon_trading_days, analysis_mode, schema_version);

create table if not exists public.monthly_commit_receipts (
    run_id uuid primary key,
    bundle_hash text not null,
    committed_at timestamptz not null default now()
);
alter table public.monthly_commit_receipts enable row level security;
revoke all on public.monthly_commit_receipts from public, anon, authenticated;

create or replace function public.persist_monthly_bundle(
    p_analysis jsonb, p_snapshot jsonb, p_forecast jsonb,
    p_outcome jsonb, p_request_hash text
) returns jsonb
language plpgsql security definer set search_path = pg_catalog, public
as $function$
declare
    run uuid := (p_forecast->>'run_id')::uuid;
    fingerprint text;
    previous_hash text;
    existing jsonb;
    expected jsonb;
begin
    if run is null or p_analysis is null or p_snapshot is null
       or p_request_hash is null or p_request_hash !~ '^[a-f0-9]{64}$'
       or (p_analysis->>'run_id')::uuid is distinct from run
       or (p_forecast->>'feature_snapshot_id')::uuid is distinct from (p_snapshot->>'snapshot_id')::uuid
    then raise exception 'INVALID_FORECAST_BUNDLE' using errcode = '22023'; end if;
    -- Serialize requests for one run, including concurrent retry after timeout.
    perform pg_advisory_xact_lock(hashtextextended(run::text, 0));
    fingerprint := encode(extensions.digest(jsonb_build_object(
        'request', p_request_hash,
        'analysis', p_analysis - array['created_at','completed_at','duration_s'],
        'snapshot', p_snapshot, 'forecast', p_forecast, 'outcome', p_outcome
    )::text, 'sha256'), 'hex');
    select bundle_hash into previous_hash from public.monthly_commit_receipts where run_id = run;
    if found then
        if previous_hash is distinct from fingerprint then
            raise exception 'FORECAST_RUN_CONFLICT' using errcode = 'PAA01';
        end if;
        return jsonb_build_object('status','SAVED','run_id',run,'replayed',true);
    end if;
    -- Pre-migration runs have no receipt: require a fresh id instead of overwriting.
    if exists(select 1 from public.monthly_forecasts where run_id = run)
       or exists(select 1 from public.analyses where run_id = run) then
        raise exception 'FORECAST_RUN_CONFLICT' using errcode = 'PAA01';
    end if;
    insert into public.feature_snapshots (snapshot_id, ticker, as_of, horizon_trading_days, analysis_mode, feature_set_variant, dataset_version, feature_version, schema_version, feature_values, evidence_ids, materialized_at)
    select snapshot_id, ticker, as_of, horizon_trading_days, analysis_mode, feature_set_variant, dataset_version, feature_version, schema_version, feature_values, evidence_ids, materialized_at from jsonb_populate_record(null::public.feature_snapshots, p_snapshot) on conflict (snapshot_id) do nothing;
    select to_jsonb(s) into existing from public.feature_snapshots s
        where snapshot_id = (p_snapshot->>'snapshot_id')::uuid;
    expected := to_jsonb(jsonb_populate_record(null::public.feature_snapshots, p_snapshot));
    if existing is distinct from expected then
        raise exception 'FORECAST_SNAPSHOT_CONFLICT' using errcode = 'PAA01';
    end if;
    insert into public.analyses (analysis_id, user_id, query, ticker, status, final_report, financial_metrics, sentiment_analysis, agent_execution_history, orchestrator_logs, error_message, duration_s, created_at, completed_at, as_of, target_date, horizon_trading_days, analysis_mode, schema_version, dataset_version, feature_version, model_version, prompt_version, run_id, evidence_ids)
    select analysis_id, user_id, query, ticker, status, final_report, financial_metrics, sentiment_analysis, agent_execution_history, orchestrator_logs, error_message, duration_s, created_at, completed_at, as_of, target_date, horizon_trading_days, analysis_mode, schema_version, dataset_version, feature_version, model_version, prompt_version, run_id, evidence_ids from jsonb_populate_record(null::public.analyses, p_analysis) ;
    insert into public.monthly_forecasts (ticker, as_of, target_date, horizon_trading_days, benchmark, cutoff_policy, model_variant, model_version, feature_version, dataset_version, schema_version, action_policy_version, run_id, feature_snapshot_id, predicted_excess_return, predicted_ticker_return, predicted_benchmark_return, uncertainty_lower, uncertainty_upper, action, evidence_ids, forecast_payload)
    select ticker, as_of, target_date, horizon_trading_days, benchmark, cutoff_policy, model_variant, model_version, feature_version, dataset_version, schema_version, action_policy_version, run_id, feature_snapshot_id, predicted_excess_return, predicted_ticker_return, predicted_benchmark_return, uncertainty_lower, uncertainty_upper, action, evidence_ids, forecast_payload from jsonb_populate_record(null::public.monthly_forecasts, p_forecast) ;
    if p_outcome is not null and p_outcome <> 'null'::jsonb then
        insert into public.forecast_outcomes (ticker, as_of, target_date, horizon_trading_days, price_policy_version, benchmark, dataset_version, ticker_start_adjusted_close, ticker_end_adjusted_close, benchmark_start_adjusted_close, benchmark_end_adjusted_close, ticker_total_return, benchmark_total_return, excess_return, source_snapshot_ids, observable_at)
    select ticker, as_of, target_date, horizon_trading_days, price_policy_version, benchmark, dataset_version, ticker_start_adjusted_close, ticker_end_adjusted_close, benchmark_start_adjusted_close, benchmark_end_adjusted_close, ticker_total_return, benchmark_total_return, excess_return, source_snapshot_ids, observable_at from jsonb_populate_record(null::public.forecast_outcomes, p_outcome) on conflict on constraint forecast_outcomes_natural_key do nothing;
        select to_jsonb(o) - array['outcome_id','materialized_at','cutoff_policy'] into existing
          from public.forecast_outcomes o
          where ticker = p_outcome->>'ticker'
            and as_of = (p_outcome->>'as_of')::timestamptz
            and target_date = (p_outcome->>'target_date')::date
            and horizon_trading_days = (p_outcome->>'horizon_trading_days')::integer
            and price_policy_version = p_outcome->>'price_policy_version'
            and benchmark = p_outcome->>'benchmark'
            and dataset_version = p_outcome->>'dataset_version';
        expected := to_jsonb(jsonb_populate_record(null::public.forecast_outcomes, p_outcome))
                    - array['outcome_id','materialized_at','cutoff_policy'];
        if existing is distinct from expected then
            raise exception 'FORECAST_OUTCOME_CONFLICT' using errcode = 'PAA01';
        end if;
    end if;
    insert into public.monthly_commit_receipts(run_id,bundle_hash) values(run,fingerprint);
    return jsonb_build_object('status','SAVED','run_id',run,'replayed',false);
end
$function$;
revoke all on function public.persist_monthly_bundle(jsonb,jsonb,jsonb,jsonb,text) from public, anon, authenticated;
grant execute on function public.persist_monthly_bundle(jsonb,jsonb,jsonb,jsonb,text) to service_role;
commit;
