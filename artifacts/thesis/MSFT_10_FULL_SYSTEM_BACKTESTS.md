# Ten Full-System Agentic Backtest Attempts for MSFT

> Result status: **BLOCKED_BY_POINT_IN_TIME_SPECIALIST_COVERAGE**. This report invokes the real Research, Financial, and Sentiment agents through ManagerAgent. It does not substitute Quant-only predictions when required specialist evidence is missing.

## Executive Result

- Attempted monthly cutoffs: **10**
- Attempts in which all three agents were called: **10**
- Full-system numeric forecasts produced: **0**
- Correct full-system directions: **0**
- Full-system directional accuracy: **N/A**
- Temporal audit of Quant source rows: **PASS**

The previously reported 7/10 result belongs to the market-only Quant evaluation. It is not counted as a full-system result here.

## Per-Cutoff Results

Evidence counts are shown as Research / Financial / Sentiment.

| # | Forecast date | Target date | Specialist evidence counts | Manager result | Action | P(positive excess) | Actual excess | Direction correct? |
|---:|---|---|---|---|---|---:|---:|:---:|
| 1 | 2025-10-31 | 2025-12-02 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -5.12% | N/A |
| 2 | 2025-11-28 | 2025-12-30 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -1.75% | N/A |
| 3 | 2025-12-31 | 2026-02-02 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -14.44% | N/A |
| 4 | 2026-01-30 | 2026-03-03 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -4.23% | N/A |
| 5 | 2026-02-27 | 2026-03-30 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -0.98% | N/A |
| 6 | 2026-03-31 | 2026-04-30 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -0.35% | N/A |
| 7 | 2026-04-30 | 2026-06-01 | 0 / 64 / 0 | BLOCKED | N/A | N/A | 7.63% | N/A |
| 8 | 2026-05-29 | 2026-06-30 | 0 / 64 / 0 | BLOCKED | N/A | N/A | -16.12% | N/A |
| 9 | 2026-06-30 | 2026-07-30 | 0 / 64 / 0 | BLOCKED | N/A | N/A | 21.61% | N/A |
| 10 | 2026-07-31 | 2026-08-31 | 0 / 64 / 0 | BLOCKED | N/A | N/A | 6.69% | N/A |

## Interpretation

Only 0 of 10 attempts produced a numeric full-system forecast. Missing or degraded historical specialist evidence caused the remaining attempts to fail closed. No accuracy claim can be made for blocked attempts.

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
.\.venv\Scripts\python.exe -m evaluation.full_system_backtest_report --snapshot-input artifacts/thesis/MSFT_YAHOO_SNAPSHOT.json --output-dir artifacts/thesis
```

- Run ID: `full-system-msft-10bt-20260924`
- Report version: `full-system-agent-backtest-v1`
- Source snapshot SHA-256: `bcea064c6b66861f67d5e6b2ba437930df675e9e5f696f6e45589a1da43ee547`
- Report hash: `16def54c7d8deab1c31491fb37e1899a52f9e315d664e14ee676a5cc686f42f0`
