"""Run and document ten full-system historical forecast attempts.

This evaluation harness does not change the production agents. It invokes the
real Research, Financial, and Sentiment agents through ManagerAgent, fits the
Quant model independently at every cutoff, and reports fail-closed outcomes as
failures rather than substituting market-only predictions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

from dotenv import load_dotenv
from mcp import StdioServerParameters

from agents.financial_agent import FinancialAnalystAgent
from agents.manager_agent import ManagerAgent
from agents.monthly_forecast_agent import MonthlyForecastAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from core.forecast_contract import AnalysisMode, ForecastContext, ForecastCutoffPolicy
from evaluation.monthly_backtest import BacktestReport, build_expanding_folds
from evaluation.point_in_time_dataset import PointInTimeDataset, stable_hash
from evaluation.temporal_leakage import audit_backtest
from evaluation.thesis_backtest_report import (
    HistoricalInputs,
    _universe_manifest,
    build_labeled_rows,
    load_historical_inputs,
)
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.historical_dataset_builder import (
    MONTH_END_CUTOFF_POLICY_VERSION,
    build_data_readiness,
    eligible_row_content_hash,
)
from tools.market_tools.market_data import (
    MARKET_DATA_VERSION,
    build_market_bundle,
    rows_to_cache,
)
from tools.market_tools.market_server import MarketForecastService


REPORT_VERSION = 'full-system-agent-backtest-v1'
MODEL_VARIANT = 'walk_forward_full_system+specialist_fusion_v2'
SLOT_NAMES = {
    'research': 'research_features',
    'financial': 'financial_features',
    'sentiment': 'sentiment_features',
}


class SnapshotMarketProvider:
    """Build an exact as-of market bundle from the locked evaluation snapshot."""

    def __init__(self, inputs: HistoricalInputs) -> None:
        self.inputs = inputs

    def __call__(self, context: ForecastContext):
        end = context.as_of.date()
        start = end - timedelta(days=context.lookback_days)
        caches = {}
        for symbol in (context.ticker, context.benchmark):
            bars, actions = self.inputs.histories[symbol]
            caches[symbol] = rows_to_cache(
                ticker=symbol,
                start=start,
                end=end,
                retrieved_at=context.as_of.isoformat().replace('+00:00', 'Z'),
                bars=(
                    row for row in bars
                    if start <= date.fromisoformat(str(row['date'])) <= end
                ),
                corporate_actions=(
                    row for row in actions
                    if start <= date.fromisoformat(str(row['effective_date'])) <= end
                ),
            )
        return build_market_bundle(
            context=context, caches=caches, start=start, end=end,
        )


class CapturingAgent:
    """Capture one real specialist result without changing its behavior."""

    def __init__(self, name: str, inner: Any) -> None:
        self.name = name
        self.inner = inner
        self.called = False
        self.result: Mapping[str, Any] | None = None
        self.error_type: str | None = None

    async def run(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.called = True
        try:
            result = await self.inner.run(state)
        except Exception as exc:
            self.error_type = type(exc).__name__
            raise
        self.result = result if isinstance(result, Mapping) else None
        return result


def _mcp_params(root: Path, relative: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=[str(root / relative)], env=None,
    )


def _specialists(root: Path) -> dict[str, CapturingAgent]:
    return {
        'research': CapturingAgent(
            'research',
            ResearchAgent(
                max_loops=1,
                mcp_server_params=_mcp_params(
                    root, 'tools/research_tools/research_server.py',
                ),
            ),
        ),
        'financial': CapturingAgent(
            'financial',
            FinancialAnalystAgent(
                max_loops=1,
                mcp_server_params=_mcp_params(
                    root, 'tools/financial_tools/financial_server.py',
                ),
            ),
        ),
        'sentiment': CapturingAgent(
            'sentiment',
            SentimentAgent(
                server_script_path=str(
                    root / 'tools/sentiment_tools/sentiment_server.py'
                ),
                max_loops=1,
                mcp_server_params=_mcp_params(
                    root, 'tools/sentiment_tools/sentiment_server.py',
                ),
            ),
        ),
    }


def _slot_summary(agent: CapturingAgent) -> dict[str, Any]:
    slot_name = SLOT_NAMES[agent.name]
    slot = agent.result.get(slot_name) if agent.result else None
    slot = slot if isinstance(slot, Mapping) else {}
    evidence = slot.get('evidence_ids')
    errors = slot.get('errors')
    return {
        'called': agent.called,
        'version': slot.get('version'),
        'missing': True if not slot else bool(slot.get('missing')),
        'degraded': True if not slot else bool(slot.get('degraded')),
        'evidence_count': len(evidence) if isinstance(evidence, (list, tuple)) else 0,
        'errors': list(errors) if isinstance(errors, (list, tuple)) else [],
        'execution_error_type': agent.error_type,
    }


def _context(row: Any, model: Any) -> ForecastContext:
    return ForecastContext(
        ticker=row.snapshot.ticker,
        as_of=row.snapshot.as_of,
        target=row.snapshot.target_date,
        benchmark=row.snapshot.benchmark,
        analysis_mode=AnalysisMode.HISTORICAL_BACKTEST,
        cutoff_policy=ForecastCutoffPolicy.MONTH_END,
        schema_version=row.snapshot.schema_version,
        dataset_version=model.dataset_version,
        feature_version=model.feature_version,
        model_version=model.model_version,
    )


async def _run_fold(
    *, index: int, fold: Any, universe: Mapping[str, Any],
    inputs: HistoricalInputs, root: Path, run_id: str,
) -> dict[str, Any]:
    test_row = fold.test_rows[0]
    fold_id = f'{run_id}-bt{index:02d}'
    readiness = build_data_readiness(
        rows=fold.training_rows,
        universe_manifest=universe,
        run_id=fold_id,
        source_hashes={'yahoo_snapshot_sha256': inputs.source_hash},
        cutoff_policy_version=MONTH_END_CUTOFF_POLICY_VERSION,
    )
    model = fit_deterministic_model(
        rows=fold.training_rows,
        readiness=readiness,
        train_through=fold.forecast_as_of,
        run_id=fold_id,
    )
    service = MarketForecastService(
        model=model,
        readiness=readiness,
        universe_manifest_hash=str(universe['manifest_hash']),
        universe_constituents=tuple(universe['constituents']),
        model_variant=MODEL_VARIANT,
    )
    context = _context(test_row, model)
    agents = _specialists(root)
    manager = ManagerAgent(
        research_agent=agents['research'],
        financial_agent=agents['financial'],
        sentiment_agent=agents['sentiment'],
        memory=MagicMock(),
        monthly_forecast_agent=MonthlyForecastAgent(),
        market_forecast_service=service,
        market_data_provider=SnapshotMarketProvider(inputs),
    )
    state: Mapping[str, Any] = {}
    manager_error_type = None
    try:
        state = await manager.run(
            'Produce a coordinated historical forecast using research, '
            'financial, sentiment, and market evidence.',
            {
                'ticker': context.ticker,
                'run_id': fold_id,
                'forecast_context': context.to_dict(),
                'include_sentiment': True,
                'monthly_feature_profile': True,
            },
            client_session_id=f'fsbt{index:02d}',
        )
    except Exception as exc:
        manager_error_type = type(exc).__name__
    forecast = state.get('monthly_forecast') if isinstance(state, Mapping) else None
    forecast = forecast if isinstance(forecast, Mapping) else {}
    probability = forecast.get('probability_positive')
    direction_correct = None
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        direction_correct = (1 if float(probability) >= 0.5 else 0) == int(
            test_row.direction_positive
        )
    return {
        'backtest_number': index,
        'fold_id': fold.fold_id,
        'run_id': fold_id,
        'as_of': test_row.snapshot.as_of,
        'target_date': test_row.snapshot.target_date,
        'training_cutoffs': fold.training_cutoffs,
        'training_rows': len(fold.training_rows),
        'training_max_as_of': fold.training_max_as_of,
        'training_max_outcome_materialized_at': (
            fold.training_max_outcome_materialized_at
        ),
        'agents': {name: _slot_summary(agent) for name, agent in agents.items()},
        'all_agents_called': all(agent.called for agent in agents.values()),
        'manager_error_type': manager_error_type,
        'forecast_status': forecast.get('status') or 'BLOCKED',
        'quality_gate_passed': bool(forecast.get('quality_gate_passed')),
        'quality_gate_failures': list(forecast.get('quality_gate_failures') or []),
        'action': forecast.get('action'),
        'probability_positive': probability,
        'expected_return_21d': forecast.get('expected_return_21d'),
        'expected_excess_return': forecast.get('expected_excess_return'),
        'model_variant': forecast.get('model_variant') or MODEL_VARIANT,
        'model_version': model.model_version,
        'actual_ticker_return': test_row.ticker_total_return,
        'actual_benchmark_return': test_row.benchmark_total_return,
        'actual_excess_return': test_row.forward_21d_excess_return,
        'actual_direction_positive': test_row.direction_positive,
        'direction_correct': direction_correct,
    }


def _audit_source_rows(
    *, rows: Sequence[Any], universe: Mapping[str, Any], run_id: str,
) -> dict[str, Any]:
    manifest = {
        'dataset_version': MARKET_DATA_VERSION,
        'feature_version': MARKET_FEATURE_VERSION,
        'eligible_rows_sha256': eligible_row_content_hash(rows),
    }
    manifest_hash = stable_hash(manifest)
    dataset = PointInTimeDataset(
        rows=tuple(rows),
        universe_manifest=MappingProxyType(dict(universe)),
        manifest=MappingProxyType(manifest),
        manifest_hash=manifest_hash,
    )
    empty = BacktestReport(
        manifest=MappingProxyType({'run_id': run_id}), folds=(), records=(),
        metrics=MappingProxyType({}), leakage_audit=MappingProxyType({}),
    )
    return audit_backtest(dataset, empty, run_id=run_id).to_dict()


def _pct(value: Any) -> str:
    return 'N/A' if not isinstance(value, (int, float)) or isinstance(value, bool) else f'{100.0 * float(value):.2f}%'


def render_markdown(payload: Mapping[str, Any]) -> str:
    table = []
    for item in payload['attempts']:
        agents = item['agents']
        table.append(
            f"| {item['backtest_number']} | {item['as_of'][:10]} | "
            f"{item['target_date']} | {agents['research']['evidence_count']} / "
            f"{agents['financial']['evidence_count']} / "
            f"{agents['sentiment']['evidence_count']} | "
            f"{item['forecast_status']} | {item['action'] or 'N/A'} | "
            f"{_pct(item['probability_positive'])} | "
            f"{_pct(item['actual_excess_return'])} | "
            f"{('YES' if item['direction_correct'] else 'NO') if item['direction_correct'] is not None else 'N/A'} |"
        )
    available = int(payload['numeric_forecast_count'])
    correct = int(payload['correct_direction_count'])
    accuracy = 'N/A' if not available else f'{100.0 * correct / available:.2f}%'
    return f"""# Ten Full-System Agentic Backtest Attempts for {payload['ticker']}

> Result status: **{payload['status']}**. This report invokes the real Research, Financial, and Sentiment agents through ManagerAgent. It does not substitute Quant-only predictions when required specialist evidence is missing.

## Executive Result

- Attempted monthly cutoffs: **{payload['attempt_count']}**
- Attempts in which all three agents were called: **{payload['all_agents_called_count']}**
- Full-system numeric forecasts produced: **{available}**
- Correct full-system directions: **{correct}**
- Full-system directional accuracy: **{accuracy}**
- Temporal audit of Quant source rows: **{payload['temporal_audit']['status']}**

The previously reported 7/10 result belongs to the market-only Quant evaluation. It is not counted as a full-system result here.

## Per-Cutoff Results

Evidence counts are shown as Research / Financial / Sentiment.

| # | Forecast date | Target date | Specialist evidence counts | Manager result | Action | P(positive excess) | Actual excess | Direction correct? |
|---:|---|---|---|---|---|---:|---:|:---:|
{chr(10).join(table)}

## Interpretation

{payload['interpretation']}

The system is designed to fail closed when any required specialist slot is missing or degraded. A blocked attempt is therefore recorded as blocked, not converted into a market-only forecast and not scored as correct or incorrect.

## Method

1. The final ten eligible month-end cutoffs were fixed before full-system results were inspected.
2. Quant was re-fitted independently for every fold using only outcomes observable at that cutoff.
3. The real ResearchAgent, FinancialAnalystAgent, and SentimentAgent were invoked concurrently by ManagerAgent.
4. Specialist fusion required usable evidence from all three specialist agents.
5. Realized 21-trading-day MSFT and SPY returns came from the locked Yahoo snapshot.

## Limitations

- The locked Yahoo price snapshot was downloaded retrospectively; it is not a provider-vintage archive captured at each historical cutoff. Locking the file makes this run reproducible but does not prove that the provider's historical values were unchanged at every cutoff.
- The temporal audit PASS applies to Quant training cutoffs, outcome-materialization timestamps, and source-row ordering. It does not certify unavailable Research, Financial, or Sentiment evidence.
- Because no attempt produced an eligible full-system forecast, this run cannot estimate full-system predictive accuracy.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe -m evaluation.full_system_backtest_report --snapshot-input artifacts/thesis/MSFT_YAHOO_SNAPSHOT.json --output-dir artifacts/thesis
```

- Run ID: `{payload['run_id']}`
- Report version: `{payload['report_version']}`
- Source snapshot SHA-256: `{payload['source_snapshot_sha256']}`
- Report hash: `{payload['report_hash']}`
"""


async def build_payload(
    *, ticker: str, benchmark: str, start: date, count: int,
    run_id: str, snapshot_input: Path,
) -> dict[str, Any]:
    root = Path.cwd()
    inputs = load_historical_inputs(snapshot_input)
    rows = build_labeled_rows(
        ticker=ticker, benchmark=benchmark, start=start,
        inputs=inputs, run_id=run_id,
    )
    universe = _universe_manifest(
        ticker=ticker, benchmark=benchmark, start=start,
        retrieved_at=inputs.retrieved_at,
    )
    folds = build_expanding_folds(rows)[-count:]
    attempts = []
    for index, fold in enumerate(folds, start=1):
        print(f'full_system_backtest={index}/{count} as_of={fold.forecast_as_of}', flush=True)
        attempts.append(await _run_fold(
            index=index, fold=fold, universe=universe, inputs=inputs,
            root=root, run_id=run_id,
        ))
    numeric = [
        item for item in attempts
        if isinstance(item['probability_positive'], (int, float))
        and not isinstance(item['probability_positive'], bool)
    ]
    correct = sum(item['direction_correct'] is True for item in numeric)
    all_called = sum(item['all_agents_called'] is True for item in attempts)
    if len(numeric) == count:
        status = 'COMPLETE'
        interpretation = (
            'All ten full-system forecasts were produced. Accuracy is computed '
            'only from these full-system outputs.'
        )
    else:
        status = 'BLOCKED_BY_POINT_IN_TIME_SPECIALIST_COVERAGE'
        interpretation = (
            f'Only {len(numeric)} of {count} attempts produced a numeric full-system '
            'forecast. Missing or degraded historical specialist evidence caused '
            'the remaining attempts to fail closed. No accuracy claim can be made '
            'for blocked attempts.'
        )
    payload: dict[str, Any] = {
        'report_version': REPORT_VERSION,
        'run_id': run_id,
        'status': status,
        'ticker': ticker,
        'benchmark': benchmark,
        'attempt_count': len(attempts),
        'all_agents_called_count': all_called,
        'numeric_forecast_count': len(numeric),
        'correct_direction_count': correct,
        'attempts': attempts,
        'temporal_audit': _audit_source_rows(
            rows=rows, universe=universe, run_id=run_id,
        ),
        'source_snapshot_sha256': inputs.source_hash,
        'interpretation': interpretation,
    }
    payload['report_hash'] = stable_hash(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Run ten full-system historical AgenticAlpha attempts',
    )
    parser.add_argument('--ticker', default='MSFT')
    parser.add_argument('--benchmark', default='SPY')
    parser.add_argument('--start', type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument('--backtests', type=int, default=10)
    parser.add_argument('--run-id', default='full-system-msft-10bt-20260924')
    parser.add_argument(
        '--snapshot-input', type=Path,
        default=Path('artifacts/thesis/MSFT_YAHOO_SNAPSHOT.json'),
    )
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/thesis'))
    args = parser.parse_args(argv)
    if args.backtests != 10:
        raise ValueError('this thesis protocol requires exactly ten backtests')
    load_dotenv()
    payload = asyncio.run(build_payload(
        ticker=args.ticker.strip().upper(),
        benchmark=args.benchmark.strip().upper(),
        start=args.start,
        count=args.backtests,
        run_id=args.run_id,
        snapshot_input=args.snapshot_input,
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{payload["ticker"]}_10_FULL_SYSTEM_BACKTESTS'
    json_path = args.output_dir / f'{stem}.json'
    markdown_path = args.output_dir / f'{stem}.md'
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + '\n',
        encoding='utf-8',
    )
    markdown_path.write_text(render_markdown(payload), encoding='utf-8')
    print(f'markdown={markdown_path}')
    print(f'json={json_path}')
    print(f'report_hash={payload["report_hash"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['REPORT_VERSION', 'build_payload', 'render_markdown']
