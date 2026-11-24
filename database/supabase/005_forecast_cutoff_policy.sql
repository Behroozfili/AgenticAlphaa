-- Distinguish month-end forecasts from point-in-time month-to-date forecasts.
-- Existing rows predate the explicit contract and are therefore MONTH_END.

alter table public.monthly_forecasts
    add column if not exists cutoff_policy text not null default 'MONTH_END';

alter table public.monthly_forecasts
    drop constraint if exists monthly_forecasts_cutoff_policy_valid;

alter table public.monthly_forecasts
    add constraint monthly_forecasts_cutoff_policy_valid
    check (cutoff_policy in ('MONTH_END', 'MONTH_TO_DATE'));

alter table public.monthly_forecasts
    drop constraint if exists monthly_forecasts_natural_key;

alter table public.monthly_forecasts
    add constraint monthly_forecasts_natural_key
    unique (
        ticker, as_of, target_date, horizon_trading_days, model_variant,
        model_version, feature_version, dataset_version, cutoff_policy, run_id
    );

comment on column public.monthly_forecasts.cutoff_policy is
    'MONTH_END requires the final regular session; MONTH_TO_DATE permits any completed regular-session close.';
