# 10-Fold Walk-Forward Backtest and Model Forecast for MSFT

> Scientific status: **PRELIMINARY**. This report documents actual model outputs, but it is not sufficient on its own to support claims of generalizability or reliable profitability.

## Executive Summary

This run contains **10 strictly out-of-sample forecasts** with a fixed horizon of **21 trading sessions** for `MSFT` against `SPY`. Each fold uses only training data and outcomes observable at the forecast cutoff. The model predicted the excess-return direction correctly in **7 of 10** cases.

Current model output as of **2026-08-31**, for target date **2026-09-30**:

- Model action: **HOLD**
- Probability of positive excess return versus `SPY`: **65.26%**
- Expected 21-session stock return: **5.29%**
- Expected excess return: **1.57%**
- Stock-return prediction interval: **[-3.68%, 15.71%]**
- Model confidence: **30.53%**

The current forecast has no realized outcome and is excluded from the ten backtest observations.

## Ten Backtest Forecast Outputs

| # | Forecast date | Target date | Training cutoffs | P(positive excess) | Predicted stock return | Predicted excess | Action | Actual stock return | Actual excess | Direction correct? | Absolute excess error |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|:---:|---:|
| 1 | 2025-10-31 | 2025-12-02 | 81 | 45.82% | 3.59% | 0.59% | HOLD | -5.19% | -5.12% | YES | 5.71% |
| 2 | 2025-11-28 | 2025-12-30 | 81 | 30.03% | 1.10% | -1.17% | HOLD | -0.92% | -1.75% | YES | 0.58% |
| 3 | 2025-12-31 | 2026-02-02 | 83 | 32.89% | 2.52% | -0.26% | HOLD | -12.46% | -14.44% | YES | 14.18% |
| 4 | 2026-01-30 | 2026-03-03 | 83 | 13.06% | 2.12% | -2.45% | AVOID | -5.91% | -4.23% | YES | 1.78% |
| 5 | 2026-02-27 | 2026-03-30 | 84 | 11.53% | -2.00% | -3.68% | AVOID | -8.60% | -0.98% | YES | 2.71% |
| 6 | 2026-03-31 | 2026-04-30 | 86 | 19.39% | -2.57% | -2.82% | AVOID | 10.16% | -0.35% | YES | 2.47% |
| 7 | 2026-04-30 | 2026-06-01 | 87 | 47.74% | 0.65% | -0.32% | HOLD | 13.18% | 7.63% | NO | 7.95% |
| 8 | 2026-05-29 | 2026-06-30 | 87 | 26.48% | -1.01% | -3.07% | AVOID | -17.15% | -16.12% | YES | 13.06% |
| 9 | 2026-06-30 | 2026-07-30 | 89 | 1.20% | -7.33% | -8.86% | AVOID | 20.93% | 21.61% | NO | 30.47% |
| 10 | 2026-07-31 | 2026-08-31 | 90 | 43.37% | 0.11% | -3.79% | AVOID | 9.37% | 6.69% | NO | 10.47% |

Action policy: `BUY` requires probability at least 55% and expected excess return at least 2%; `AVOID` requires probability at most 45% and expected excess return at most -2%; all other outputs are `HOLD`.

## Aggregate Metrics for the Ten Forecasts

- Balanced directional accuracy: **50.00%**
- Precision / Recall / F1: **0.00% / 0.00% / 0.00%**
- ROC-AUC: **0.619**
- Brier score: **0.2116**
- Excess-return MAE / RMSE: **8.94% / 12.34%**
- Spearman IC: **-0.491**
- Mean net excess return after costs: **-0.69%**
- Annualized net Sharpe ratio: **-0.257**
- Maximum net drawdown: **26.85%**

**Required interpretation:** Seven correct directions out of ten is not, by itself, evidence of strong performance. The model failed to classify any of the three positive-excess-return periods as positive. Consequently, positive-class recall and F1 are zero, while balanced accuracy is 50%. Spearman IC and mean net excess return are also negative. This ten-observation run does not demonstrate predictive superiority or profitability; its value is transparent documentation of model outputs and the evaluation protocol.

Investment metrics include **10 basis points of one-way transaction cost**. All metrics are based on only ten observations and therefore have substantial statistical uncertainty.

## Reproducible Methodology

1. Adjusted OHLCV data for `MSFT` and `SPY` were acquired from Yahoo Finance through `yfinance`.
2. Each monthly cutoff is the final regular U.S. market session of the month. The target is exactly the 21st trading session after the cutoff.
3. At least **36 training cutoffs** are required. The expanding window includes only outcomes observable by the forecast time.
4. The model uses logistic ridge regression for directional probability and ridge regression for stock and excess-return amounts. This report evaluates market-only historical features.
5. Mechanical temporal-leakage audit: **PASS**; detected violations: **0** across 10 checked forecasts. A PASS verifies cutoff, training, and outcome ordering; it does not prove that archived provider vintages were available at every historical cutoff.

Reproduction command:

```powershell
.\.venv\Scripts\python.exe -m evaluation.thesis_backtest_report --ticker MSFT --benchmark SPY --start 2019-01-01 --backtests 10 --run-id thesis-msft-10bt-20260924 --snapshot-input artifacts/thesis/MSFT_YAHOO_SNAPSHOT.json
```

The command replays the saved snapshot and does not download revised provider data.

## Data Provenance and Run Identity

- Provider: Yahoo Finance via `yfinance`
- Retrieved at: `2026-09-24T15:43:18.198276Z`
- Last common market session: `2026-09-24`
- Run ID: `thesis-msft-10bt-20260924`
- Report version: `thesis-real-market-walk-forward-v1`
- Dataset version: `adjusted-ohlcv-cache-v1`
- Feature version: `monthly-market-features-v1`
- Snapshot SHA-256: `bcea064c6b66861f67d5e6b2ba437930df675e9e5f696f6e45589a1da43ee547`
- Eligible-row SHA-256: `41d27b1f297f9ab6bae31224a64f91a6b2b8428b027d644438c1ba74f7aa0c52`
- Report hash: `20001fc7e17e20510288659d6ffd056fe97a997240c4ac57e4f5697845a611c8`
- Raw output: [MSFT_10_BACKTESTS.json](./MSFT_10_BACKTESTS.json)
- Data snapshot: [MSFT_YAHOO_SNAPSHOT.json](./MSFT_YAHOO_SNAPSHOT.json)

## Limitations and Thesis Use

- This experiment covers one stock and ten forecasts, so its scientific status is **PRELIMINARY** and it must not be generalized to the broader market.
- Yahoo adjusted OHLCV was downloaded retrospectively and is not an archived provider vintage for every cutoff. Later adjustment revisions may therefore affect the reconstruction.
- Historical news, financial-statement, sentiment, and graph features were unavailable. The report evaluates the market-only Quant component, not the complete multi-agent system.
- The final ten folds must remain fixed after inspection; changing the period after observing results would introduce selection bias.
- The current output is not investment advice, and its success cannot be evaluated until the target price is observed.

Suggested in-text citation:

> (AgenticAlpha, 2026, `thesis-msft-10bt-20260924`, report hash: `20001fc7e17e2051...`)

Suggested BibTeX entry:

```bibtex
@techreport{agenticalpha_msft_10bt_2026,
  author      = {AgenticAlpha},
  title       = {10-Fold Walk-Forward Backtest and 21-Trading-Day Forecast for MSFT},
  year        = {2026},
  institution = {AgenticAlpha research software},
  type        = {Reproducible model evaluation report},
  note        = {Run thesis-msft-10bt-20260924; report hash 20001fc7e17e20510288659d6ffd056fe97a997240c4ac57e4f5697845a611c8; data retrieved 2026-09-24T15:43:18.198276Z}
}
```
