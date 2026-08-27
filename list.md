# نقشهٔ قدم‌به‌قدم تبدیل AgenticAlpha به پیش‌بینی ماهانه و Backtest

این سند حاصل بررسی فایل‌های پروژه، `dfc.md` و ارجاع قبلی به `validation_MSFT_full_report_20260826_161633.json` است. فایل trace در وضعیت فعلی workspace موجود نیست و تا بازیابی یا تصویب baseline جایگزین، در `.codex/baseline-manifest.json` به‌صورت `MISSING` ثبت می‌شود. این فایل تنها برنامهٔ canonical ریفکتور است؛ هیچ فایل دیگری با عنوان implementation plan مرجع موازی نیست.

## تصمیم‌های اجرایی تثبیت‌شده قبل از شروع

- ترتیب و مالکیت اجرایی دقیق از `.codex/refactor-state.json`، TOML ایجنت فعال و قراردادهای `.codex/contracts/` خوانده می‌شود.
- Stage 08 فقط اتصال دامنه‌ای MonthlyForecastAgent و Manager و isolation درون‌پردازه‌ای است. زیرساخت request/lifecycle در Stage 10A و route/persistence/UI/E2E در Stage 10B انجام می‌شود.
- `agents/executor_helpers/monthly_financial_features.py` متعلق به Financial در Stage 05 است؛ Integration فقط خروجی typed آن را مصرف می‌کند.
- rename چهار prompt اجرایی در Stage مالک آن‌ها انجام می‌شود: Research در 04B، Financial در 05، Sentiment در 06 و Manager در 08. Stage 11 فقط نبود reference قدیمی را کنترل می‌کند.
- قرارداد artifact و ترتیب Primary draft → Observability → Primary final → QA → Reviewer → Handoff در `.codex/contracts/control-plane.md` تثبیت شده است.
- تعریف دقیق cut-off ماهانه، target ۲۱ جلسه‌ای، providerها، universe manifest، threshold تصمیم و backtest در `.codex/contracts/monthly-backtest-contract.md` است.
- کلیدهای طبیعی جدول‌های forecast/outcome/ablation پیش از migration در `.codex/contracts/data-schema-contract.md` تثبیت شده‌اند.
- ساخت migration مجاز است، اما apply کردن migration، دانلود شبکه‌ای، seed، حذف داده، deploy، commit یا push فقط با مجوز جداگانهٔ صریح کاربر مجاز است.

## تصمیم‌های معماری ثابت

1. هدف جدید، پیش‌بینی **یک‌ماهه معادل ۲۱ روز معاملاتی** است.
2. label اصلی پیشنهادی `forward_21d_excess_return` است: بازده سهم منهای بازده SPY در همان بازه.
3. خروجی شامل `BUY / HOLD / AVOID`، احتمال مثبت‌بودن، بازده مورد انتظار، بازهٔ عدم‌قطعیت و شواهد است.
4. `as_of` مرز زمانی قطعی است؛ هیچ داده‌ای بعد از آن نباید در اجرای تاریخی دیده شود.
5. مدل آماری عدد پیش‌بینی را می‌سازد و LLM فقط شواهد را ترکیب و نتیجه را توضیح می‌دهد.
6. DCF، WACC و افق پنج‌ساله حذف یا ماهانه‌سازی نمی‌شوند؛ آن‌ها feature بلندمدت مدل ماهانه‌اند.
7. ۹ intent و mapper مالی فعلی حفظ می‌شوند؛ قرارداد زمانی و adapter ماهانه به آن‌ها افزوده می‌شود.
8. Search، Financial، Sentiment، Market و Graph ورودی `MonthlyForecastAgent` را از طریق State می‌سازند.
9. Graph در Full Model اجباری است، ولی حالت بدون Graph برای ablation و اثبات ارزش آن حفظ می‌شود.
10. دیتابیس فعلی فعلاً پاک نمی‌شود؛ schema نسخهٔ ۲ موازی ساخته و بعد از تطبیق داده فعال می‌شود.

## نتیجهٔ بررسی وضعیت فعلی

1. RAG فعلی ترکیبی است: loader → processor → embedding → Supabase vector → استخراج/ثبت Graph؛ Research نیز vector، graph و hybrid query دارد.
2. Graph بعد از موفقیت vector به‌روزرسانی می‌شود؛ شکست vector می‌تواند Graph را هم ناقص کند.
3. `rag/loader.py` فقط خبرهای جاری yfinance و RSS جاری Reddit را می‌گیرد و date range تاریخی ندارد.
4. `_safe_timestamp` هنگام timestamp نامعتبر از زمان اکنون استفاده می‌کند؛ این رفتار نشت آینده ایجاد می‌کند.
5. `rag/retriever.py` freshness را با `datetime.now()` می‌سنجد، نه با `as_of` اجرای تاریخی.
6. توابع hybrid برخی پارامترهای زمانی را دارند، اما schema و dispatcher ابزار Research آن‌ها را کامل عبور نمی‌دهند.
7. Graph با `MERGE (a)-[r:TYPE]->(b)` یک edge تجمیعی می‌سازد؛ تاریخچه و منبع شاهدهای بعدی ممکن است از بین برود.
8. Vector با `(url_hash, chunk_index)` شناخته می‌شود و شناسهٔ مشترک نسخه‌دار بین Vector و Graph وجود ندارد.
9. `agents/state.py` فقط summaryهای تجمیعی Research/Financial/Sentiment دارد و قرارداد افق/زمان/نسخه ندارد.
10. Manager فقط سه agent فعلی را ثبت و اجرا می‌کند؛ action/result/evaluation ماهانه ندارد.
11. fetcherهای مالی عمدتاً latest/current هستند و شرط سراسری `filed_at <= as_of` ندارند.
12. `get_price_history` به period رشته‌ای متکی است و قرارداد دقیق `start/end/as_of` ندارد.
13. summary مالی متن filing و summary احساسات `attempts`، `brain_reasoning` و metadata تکراری را persist می‌کنند.
14. `memory/manager_memory.py` insight را عمدتاً با ticker merge می‌کند؛ monthly و legacy تداخل می‌کنند.
15. `api/main.py` Manager singleton می‌سازد، اما memory per-request route را واقعاً به همان Manager تزریق نمی‌کند؛ خطر cross-request state وجود دارد.
16. `main.py` هنگام startup عادی seed RAG را خودکار اجرا می‌کند؛ dataset تاریخی باید explicit و versioned باشد.
17. `tests/unit_tests/test_supabase_phase5_migration.py` فایل migration ریشه را انتظار دارد، ولی فایل در repository نیست.
18. trace مبنا موفق اما حدود ۵۱۲ ثانیه است: Research حدود ۳۳۰، Financial حدود ۲۵ و Sentiment حدود ۷۰ ثانیه؛ publication به `LIMITED_REPORT/LOW` و recommendation withheld رسیده است.

## قانون پاک‌سازی

- قبل از اثبات برابری V2، هیچ جدول یا رکورد legacy پاک نشود.
- renameها با `git mv` و در commit جدا انجام شوند.
- تست/فایل تکراری فقط بعد از مقایسه و انتقال coverage حذف شود.
- raw evidence در storage اصلی بماند؛ کپی‌های حجیم آن از State، trace و analyses به reference تبدیل شوند.
- Graph و Vector فقط با یک `dataset_version` و manifest مشترک rebuild شوند.
- cleanup و تغییر رفتار مدل در یک مرحله/commit مخلوط نشوند.

---

## مرحلهٔ ۰ — baseline، backup و test runner

### فایل‌ها و تغییرها

- `validation_MSFT_full_report_20260826_161633.json`: فایل فعلاً مفقود است؛ نسخهٔ دقیق بازیابی و سپس secrets/payload حجیم redact شود، یا در Stage 00 baseline جایگزین با disposition صریح تصویب شود. عدد قدیمی از حافظه بازسازی نشود.
- `dfc.md`: حفظ شود و روشن کند DCF/WACC ارزش‌گذاری بلندمدت و feature کمکی مدل یک‌ماهه است.
- `pytest.ini`: markerهای `unit`, `integration`, `live`, `backtest`, `slow`؛ live به‌طور پیش‌فرض خاموش.
- `pyproject.toml`, `requirements.txt`, `uv.lock`: interpreter و dependency منبع‌واحد برای local/CI.
- `tests/unit_tests/test_supabase_phase5_migration.py`: migration گم‌شده بازگردد یا تست به migration نسخه‌دار واقعی منتقل شود.

### کار و Gate

1. schema/RPC/index/RLS/countهای Supabase و constraint/index/countهای Neo4j export شوند.
2. manifest شامل commit، config، dataset، prompt و model hash ساخته شود.
3. trace فعلی یک‌بار به‌عنوان baseline ثبت شود.
4. `.venv\\Scripts\\python.exe` فعلی که در بررسی با permission error متوقف شد اصلاح شود.
5. Gate: `pytest --collect-only` و unitهای offline سبز، backup دارای checksum و restore آزمایشی مستند.

## مرحلهٔ ۱ — قرارداد مرکزی Forecast

### فایل جدید `core/forecast_contract.py`

- `ForecastContext`: `ticker`, `as_of`, `target`, `horizon_trading_days=21`, `lookback_days`, `benchmark=SPY`, `analysis_mode`, `schema_version`, `dataset_version`, `feature_version`, `model_version`.
- parser UTC، تقویم روز معاملاتی، target date، و helper `assert_not_after_as_of`.
- حالت‌های `LIVE` و `HISTORICAL_BACKTEST` و قرارداد label/excess return.

### فایل‌های موجود

- `agents/state.py`: context و slotهای typed برای research/financial/sentiment/market/monthly forecast؛ legacy فعلاً بماند.
- `core/error_handler.py`: `TemporalLeakageError`, `DataUnavailableAsOfError`, `VersionMismatchError`.
- `core/observability.py` و `core/progress_bus.py`: قرارداد metadata/event در Stage 01 مشخص می‌شود، اما writer کد فقط Observability با lock صریح در pass همان Stage است؛ Primary برای call-siteها change request می‌گیرد.
- `core/__init__.py`: export قرارداد جدید.

### Gate

- تست UTC، weekend/holiday، target ۲۱روزه، رد دادهٔ آینده و سازگاری serialization State قدیمی/جدید.

## مرحلهٔ ۲ — schema نسخهٔ ۲ بدون حذف دیتابیس فعلی

### migrationهای جدید

- `database/supabase/001_alpha_documents_v2.sql`
- `database/supabase/002_forecast_tables.sql`
- `database/supabase/003_analysis_versioning.sql`
- `database/neo4j/001_temporal_evidence.cypher`

### Supabase و migration

1. `alpha_documents_v2`: ID ثابت، source URL/ID، `published_at`, `retrieved_at`, `effective_from/to`, ticker، chunk، content hash و dataset/embedding version.
2. `alpha_hybrid_search_v2`: فیلتر اجباری `as_of`, ticker, dataset_version و lookback اختیاری.
3. `feature_snapshots`: کلید `(ticker, as_of, horizon_trading_days, analysis_mode, feature_set_variant, dataset_version, feature_version)`.
4. `monthly_forecasts`: کلید شامل ticker/as-of/target/horizon، `model_variant`, `model_version`, `feature_version`, `dataset_version`, `run_id` تا ablationها روی هم نوشته نشوند.
5. `forecast_outcomes`: مستقل از مدل با کلید ticker/as-of/target/horizon/price-policy/benchmark/dataset؛ قیمت ابتدا/انتها، سهم/benchmark/excess return و زمان materialization.
6. `backtest_runs`: universe، window، config، status، metrics و artifact refs.
7. `analyses` با ستون‌های nullable نسخه‌ای توسعه یابد؛ destructive ALTER نشود.
8. legacy به `dataset_version='legacy-v1'` map شود؛ timestamp نامعلوم «اکنون» نشود.
9. shadow migration → count/hash reconciliation → read switch → write switch؛ rollback فقط pointer/view را برگرداند.

### Gate

- syntax/contract migration به‌صورت offline تست شود؛ اجرای واقعی روی DB خالی یا staging snapshot فقط پس از مجوز جداگانهٔ کاربر، سپس uniqueness/RLS/RPC، شرط `published_at <= as_of` و reconciliation بررسی شود.

## مرحلهٔ ۳ — RAG و Vector زمانی

### `rag/loader.py`

- loader تاریخی `start/end/as_of` و provider آرشیوی.
- حذف fallback زمان اکنون؛ timestamp نامعتبر به quarantine و برای backtest غیرمجاز.
- جداسازی publish/retrieval/effective time و dedupe با canonical source/content hash.

### `rag/processor.py`, `embedding_manager.py`, `vector_store.py`

- revision overwrite نشود؛ `effective_from/to` و source ID روی chunk حفظ شود.
- embedding cache key شامل content hash و model version؛ dimension/version در manifest.
- V2 read/write/RPC، فیلتر سخت as-of/ticker/version و shared document/chunk ID با Graph.
- تا زمان مجوز apply/reconciliation، مسیر V2 پشت config/adapter باقی بماند و default legacy runtime قابل استفاده باشد؛ fixtureها V2 را بدون وابستگی به جدول خارجی تست کنند.
- historical query بدون as-of یا بدون پشتیبانی backend باید fail-closed شود.

### `rag/retriever.py`, `ingestion.py`, `seed.py`, `main.py`

- freshness نسبت به ForecastContext؛ آینده قبل از ranking حذف شود.
- ingestion دارای manifest/checkpoint و resume؛ وضعیت Vector و Graph جدا و صریح.
- seed به CLI مثل `seed --start --end --dataset-version` تبدیل شود.
- auto-seed از startup عادی `main.py` حذف و pointer فقط بعد از validation عوض شود.

### Gate

- fixture قبل/بعد as-of، idempotent ingestion، failure/resume بین Vector و Graph و سازگاری live فعلی.

## مرحلهٔ ۴ — Graph زمانی، Hybrid و Research

### RAG/Graph

- `rag/graph_store.py`: به‌جای edge تجمیعی، evidence observation با document ID، زمان، effective range و dataset version؛ entity canonical بماند.
- traversal فقط evidence مجاز تا as-of؛ update وزن source/date قدیمی را نابود نکند.
- `rag/hybrid_rag.py`: context واحد برای vector/graph/RRF؛ citation به document/chunk/observation و degradation صریح هنگام Graph ناقص.

### Research tools و agent

- `tools/research_tools/research_server.py`: schema و dispatcher واقعاً `as_of/lookback/version` را منتقل کنند.
- `news_search.py`: جست‌وجوی historical و boundary؛ `sec_edgar.py`: `filed_at <= as_of`؛ `tavily_search.py`: نتیجهٔ وب امروز برای backtest گذشته ممنوع.
- `comprehensive_analysis.py`: catalyst/risk ساختاریافته با timestamp/evidence ID.
- `context_synthesizer.py`: citation/version را حفظ و dedupe کند.
- `agents/research_agent.py`: ForecastContext اجباری و `ResearchFeatures` typed.
- `agents/executor_helpers/search_executor_helpers.py`: انتقال context و validation دوبارهٔ future evidence.
- `agents/prompts/propt_search_agent.py`: rename به `prompt_research_agent.py` و الزام historical boundary/citation.

### Gate

- یک رابطه با دو شاهد در دو تاریخ؛ traversal قدیمی فقط شاهد قدیمی را ببیند. تست vector-only/graph-only/hybrid روی fixture ثابت.

## مرحلهٔ ۵ — Financial نقطه‌زمانی؛ حفظ DCF/WACC

- `tools/financial_tools/sec_edgar.py`: filing با `filed_at <= as_of`، تفکیک filed date و period end، amendment/revision.
- `yahoo_finance.py`: `start/end/as_of`؛ current snapshot در historical mode ممنوع.
- `financial_ratio_calculator.py`: فرمول‌ها حفظ؛ هر metric دارای period/filed/source/eligibility.
- `financial_server.py`: هر ۹ intent و mapper حفظ؛ ForecastContext در schema/dispatcher و response نسخه‌دار.
- `agents/financial_agent.py`: DCF/WACC و sanity حفظ، خروجی کوچک `MonthlyFinancialFeatures` اضافه.
- `finance_executor_helper.py`: انتقال context/version؛ `financial_helper.py` و `agents/executor_helpers/monthly_financial_features.py`: adapter ماهانه، feature contract ثابت و provenance canonical.
- `promt_finance_checker.py`: rename به `prompt_finance_checker.py` و الزام point-in-time.

ویژگی‌ها: valuation gap نسبت به DCF، growth/quality/leverage/liquidity، filing age، availability/missing flags. متن کامل filing از summary پایدار حذف و با document ID reference شود.

### Gate

- filing منتشرشده بعد as-of دیده نشود؛ ۹ mapper و تمام تست‌های DCF/WACC بدون تغییر معنایی سبز باشند.

## مرحلهٔ ۶ — Sentiment یک‌ماهه

- `finbert_analyzer.py`: inference حفظ، model version و determinism ثبت.
- `vader_scorer.py`: baseline حفظ و نسخه‌دار.
- `fear_greed_calculator.py`: فقط component تاریخی موجود؛ نبود داده missing، نه مقدار ساختگی.
- `sentiment_server.py`: پنجره‌های ۷/۱۴/۳۰، aggregate/momentum/dispersion/volume/source diversity.
- `agents/sentiment_agent.py` و `sentiment_helper.py`: retrieval زمانی، dedupe و خروجی `SentimentFeatures` کوچک.
- `promt_sentinemt.py`: rename به `prompt_sentiment.py` و منع شواهد خارج پنجره.
- از persistence حذف شوند: `brain_reasoning`, chain-of-thought، raw chunks، metadata تکراری و attempts کامل؛ فقط aggregate/confidence/evidence IDs/error flags بمانند.

### Gate

- خبر بعد as-of امتیاز گذشته را تغییر ندهد؛ duplicate مقاوم؛ missing data قابل توضیح.

## مرحلهٔ ۷ — Market Data، Feature Builder و مدل عددی

### فایل‌های جدید

- `tools/market_tools/__init__.py`
- `market_data.py`: OHLCV تعدیل‌شده، SPY/sector ETF، تقویم و corporate actions.
- `feature_builder.py`: momentum 5/21/63، volatility، volume surprise، drawdown، beta، relative strength و featureهای چهار بخش دیگر.
- `historical_dataset_builder.py`: ساخت snapshotهای immutable point-in-time و manifest پوشش داده پیش از fit.
- `forecast_model.py`: logistic برای احتمال و ridge/elastic-net برای return؛ scaler/artifact versioned.
- `market_server.py`: API deterministic؛ هیچ LLM در عدد پیش‌بینی.

### قواعد و dependency

- scaler/imputer فقط روی train fold؛ rolling تا as-of؛ label از روز بعد تا target.
- missingness indicator؛ صفر فرض‌کردن دادهٔ مفقود ممنوع.
- `pyproject.toml`, `requirements.txt`, `uv.lock`: `scikit-learn`، calendar و serialization امن با lock بازتولیدشده.

### Data Readiness و Gate

- قبل از fit، manifest شامل universe/version، بازهٔ هر منبع، ticker/cut-off count، missingness، پوشش خبر/filing/Graph، حذف‌ها و hashها ساخته شود.
- feature fixture ثابت، target ۲۱روزه دقیق، overlap/leakage صفر و مدل deterministic.

## مرحلهٔ ۸ — MonthlyForecastAgent و اتصال Manager

### فایل‌های جدید

- `agents/monthly_forecast_agent.py`
- `agents/executor_helpers/monthly_forecast_helper.py`
- `agents/prompts/prompt_monthly_forecast.py`

### قرارداد Agent

- فقط State مشترک را بخواند: ForecastContext، ResearchFeatures، MonthlyFinancialFeatures، SentimentFeatures، MarketFeatures و Graph coverage.
- agentهای دیگر را مستقیم صدا نزند؛ orchestration فقط در Manager باشد.
- خروجی: `expected_return_21d`, `probability_positive`, `expected_excess_return`, `confidence_interval`, `action`, `confidence`, `evidence_ids`, `warnings` و نسخه‌ها.
- در کمبود داده `INSUFFICIENT_DATA` بدهد و confidence نسازد.

### Manager و API isolation

- `agents/manager_agent.py`: node/action/evaluation/result ماهانه و مسیر Research → Financial → Sentiment → Market Features → MonthlyForecast → Canonical Synthesis → Publication.
- سه agent داده‌ای می‌توانند بعد از ساخت context موازی و سپس join شوند؛ failure باید degraded output صریح بدهد.
- prompt نهایی فقط forecast عددی را توضیح دهد و اجازهٔ تغییر عدد مدل را نداشته باشد.
- `manager_executor_helper.py`: context یک‌بار، readiness هر گروه و persistence prediction.
- `promt_manager_agent.py`: rename به `prompt_manager_agent.py`؛ افق/as-of/benchmark و تفاوت valuation با forecast اجباری.
- در Stage 08 فقط isolation دو اجرای هم‌زمان Manager درون‌پردازه‌ای تست شود. تغییر `api/main.py`/`api/dependencies.py` متعلق به Stage 10A و `api/routes/analyze.py` متعلق به Stage 10B است.
- Stage 08 باید constructor و قرارداد `ManagerAgent.run()` مورد استفاده API فعلی را backward-compatible نگه دارد یا adapter سازگار بدهد تا قبل از Stage 10A test collection/API import نشکند.

### Gate

- State کامل/ناقص/future evidence، ترتیب nodeها، schema خروجی و دو درخواست هم‌زمان بدون cross-talk.

## مرحلهٔ ۹ — Walk-forward Backtest و Evaluation

### فایل‌های جدید

- `evaluation/monthly_backtest.py`
- `evaluation/forecast_metrics.py`
- `evaluation/temporal_leakage.py`
- `evaluation/backtest_cli.py`

### روش

1. cut-off دقیقاً close آخرین جلسهٔ معاملاتی هر ماه و target دقیقاً جلسهٔ معاملاتی بیست‌ویکم بعد از آن باشد.
2. در هر cut-off فقط دادهٔ موجود تا همان تاریخ feature شود.
3. expanding/rolling walk-forward: train روی گذشته و predict روی cut-off بعد.
4. پس از target date، outcome واقعی immutable ثبت شود.
5. Full Model با baseline و ablation مقایسه شود.

### baselineها و metricها

- always HOLD/class prior، momentum 21d، market-only، بدون Graph، بدون Sentiment، بدون Financial و Full Model.
- direction: balanced accuracy، precision/recall/F1 و ROC-AUC در صورت کفایت.
- probability: Brier و calibration/ECE.
- amount: MAE/RMSE و Spearman/IC.
- investment: hit rate، excess return، turnover، drawdown و Sharpe با هزینهٔ معاملهٔ اعلام‌شده.
- coverage: درصد cut-offهایی که forecast معتبر دارند.

### evaluationهای فعلی

- `recommendation_gate.py`: data quality/calibration/minimum edge.
- `canonical_synthesis.py`: prediction عددی source of truth.
- `final_publication.py`: forecast box و backtest provenance.
- `publication_policy.py`: insufficient/stale/degraded.
- `claim_contract.py`: horizon/as-of/evidence برای ادعای ماهانه.
- `valid_availability.py`: availability point-in-time.
- `validate_period_consistency.py`: period مالی را با horizon بازار مخلوط نکند.
- `financial_sanity.py`, `earnings_quality.py`, `aspect_sentiment.py`: حفظ و خروجی feature نسخه‌دار.
- `run_ragas.py`: فقط retrieval/answer quality، نه forecast accuracy.
- `rag/evaluation.py`: overlap با RAGAS تعیین تکلیف و `JUDGE_MODEL=''` اصلاح؛ سپس یک مسیر canonical.
- `stage3_compare_cli.py`: به compare backtest تعمیم یا پس از انتقال coverage retire شود.

### Gate

- leakage تمام foldها صفر؛ fixture دستی metricها؛ اجرای deterministic؛ گزارش Full Model در کنار baseline و ablation.

## مرحلهٔ ۱۰A — API Runtime Foundation

- `api/config.py`: forecast/backtest/version settings؛ Sentry/LangSmith واقعاً optional.
- `api/core/exceptions.py`: mapping خطاهای temporal/version بدون جزئیات حساس.
- `api/dependencies.py`: factory امن per request؛ فقط client thread-safe singleton.
- `api/main.py`: lifecycle، ثبت route skeleton و حذف mutable Manager singleton.
- `memory/manager_memory.py`: کلید `(ticker, as_of, horizon, analysis_mode, schema_version)` و عدم merge legacy/monthly.
- `main.py`: حذف auto-seed startup و تبدیل seed به فرمان صریح.

## مرحلهٔ ۱۰B — Route، Persistence، Trace و UI

- `api/routes/analyze.py`: دریافت `as_of/horizon/mode`، ذخیره forecast/version و کوچک‌کردن summaryها.
- `api/routes/progress.py`: stageهای feature/monthly/backtest با run ID.
- `api/routes/backtest.py` جدید: job ایجاد و status/artifact؛ backtest طولانی synchronous نباشد.
- `langsmith_trace.py`: نیاز Stage 10B است ولی writer آن Observability با pass و lock مستقل است؛ run/as-of/horizon/version، latency و evidence count بدون secret/CoT/raw payload.
- `frontend/alpha-agent-app.html`: node و card ماهانه، target date، probability/return/action/uncertainty، Graph status و خلاصهٔ backtest؛ DCF fair value با expected 21-day price یکی نشود.

### Gate

- API contract live/historical، concurrency/memory isolation، persistence idempotency و UI success/degraded/insufficient-data.

## مرحلهٔ ۱۱ — پاک‌سازی نام، dependency و مستندات

### rename با `git mv`

- چهار rename مربوط به promptهای runtime در Stageهای 04B/05/06/08 انجام شده‌اند؛ این Stage فقط با `rg` صفر بودن import/reference نام‌های قدیمی را تأیید می‌کند.
- `tests/unit_tests/test_rag/test_hybried.py` → `test_hybrid.py`
- `tests/unit_tests/test_rag/test_vectore_store.py` → `test_vector_store.py`
- `tests/unit_tests/test_rag/test_pipline_rag.py` → `test_pipeline_rag.py`
- `tests/unit_tests/sentiment_test/test_finbert_anayzer.py` → `test_finbert_analyzer.py`
- `tests/unit_tests/api_test/test_core_eceptions.py` → `test_core_exceptions.py`
- `tests/unit_tests/test_supbase/` → `test_supabase/`
- `tests/unit_tests/test_memory/__ini__.py` → `__init__.py`

### تکراری‌ها و dependencyها

- `tests/integration_tests/` و `tests/integration_test_2/` test-by-test مقایسه؛ coverage منحصربه‌فرد منتقل و فقط بعد از سبزشدن suite canonical، دومی حذف شود.
- live/provider tests marker جدا داشته باشند.
- `pyproject.toml`: `uvicorn` و `pydantic-settings` صریح؛ `asyncio` خارجی در نبود مصرف معتبر حذف؛ test dependencies جدا.
- `requirements.txt` از منبع canonical تولید شود؛ `uv.lock` دستی ویرایش نشود.
- `Dockerfile`: از lock یکسان نصب و migration/seed/backtest را در startup خودکار اجرا نکند.
- `package-lock.json`: چون manifest frontend دیده نشد، منشأ بررسی؛ فقط اگر orphan بود در commit جدا حذف.
- `env.example`: version/provider/backtest و optionalها؛ بدون secret.
- `README.md` و `READ.md`: یکی canonical؛ دیگری پس از انتقال محتوای منحصربه‌فرد redirect یا حذف.
- `infra/ecr-stack.yaml`, `ecs-stack.yaml`, `oidc-role-stack.yaml`: secret/version env، job backtest و timeout/resourceها.

### Gate

- بعد هر batch rename: collection/unit؛ `rg` import نام قدیمی صفر؛ reproducible build و Docker smoke.

---

## ماتریس فایل‌به‌فایل: Agentها و Core

| فایل | تغییر لازم | پاک‌سازی | تست |
|---|---|---|---|
| `agents/state.py` | Context و slotهای typed | legacy فعلاً بماند | serialization |
| `agents/manager_agent.py` | orchestration ماهانه و degradation | mutable shared state | route/concurrency |
| `agents/research_agent.py` | point-in-time ResearchFeatures | متن تکراری | future rejection |
| `agents/financial_agent.py` | monthly adapter کنار DCF | full filing summary | DCF/mapper regression |
| `agents/sentiment_agent.py` | windowed SentimentFeatures | attempts/CoT/chunks | window/dedupe |
| `search_executor_helpers.py` | عبور context/version | fallback بی‌زمان | schema contract |
| `finance_executor_helper.py` | عبور context/version | current fallback | historical financial |
| `financial_helper.py` | feature adapter/provenance | metric duplicate | feature fixture |
| `monthly_financial_features.py` | خروجی typed ثابت متعلق به Financial | منطق مالی در Integration | feature contract |
| `sentiment_helper.py` | retrieval زمانی/aggregate | payload تکراری | temporal sentiment |
| `manager_executor_helper.py` | readiness/build/persist | merge legacy/monthly | end-to-end State |
| `agents/prompts/*.py` | rename و قرارداد as-of/horizon | نام typo | import/snapshot |
| `agents/__init__.py`, `agents/prompts/__init__.py` | exportهای جدید | export مرده بعد migration | import smoke |
| `core/error_handler.py` | خطاهای temporal/version | — | unit |
| `core/observability.py` | metadata نسخه‌ای | raw payload | span test |
| `core/progress_bus.py` | stageهای جدید | event بدون run ID | event test |
| `core/__init__.py` | export contract | — | import |

## ماتریس فایل‌به‌فایل: RAG، Graph و Tools

| فایل/گروه | تغییر لازم | پاک‌سازی | تست |
|---|---|---|---|
| `rag/loader.py` | historical/strict time | fallback اکنون | temporal fixture |
| `rag/processor.py` | revision/effective time | overwrite | chunk/version |
| `rag/embedding_manager.py` | model version/cache | cache بی‌نسخه | determinism |
| `rag/vector_store.py` | V2 temporal RPC/shared ID | historical بی-as-of | boundary/idempotency |
| `rag/retriever.py` | freshness نسبت as-of | `datetime.now()` تاریخی | leakage/ranking |
| `rag/graph_store.py` | temporal observations | edge بی-provenance | traversal |
| `rag/hybrid_rag.py` | context/citation/degradation | merge بی‌نسخه | RRF/ablation |
| `rag/ingestion.py` | manifest/checkpoint/resume | pipeline مبهم | restart consistency |
| `rag/seed.py` | CLI versioned | seed current مبهم | repeat/dry-run |
| `rag/evaluation.py` | یکپارچه‌سازی RAG eval | duplicate ناقص | eval smoke |
| `rag/__init__.py` | export نهایی | export منسوخ بعد window | import |
| `tools/financial_tools/*.py` | point-in-time و ۹ mapper ثابت | current data در backtest | as-of/mappers |
| `tools/research_tools/*.py` | historical providers/evidence IDs | وب امروز برای گذشته | provider fixture |
| `tools/sentiment_tools/*.py` | window/model version | raw persistence | aggregate |
| `tools/market_tools/*.py` | package جدید | — | features/model/leakage |
| `tools/__init__.py` و زیرپکیج‌ها | export package جدید؛ سایر initها بدون رفتار جدید | export مرده فقط بعد rg | imports |

مالکیت فایل‌های shared بالا از `protected_path_schedule` در `.codex/refactor-state.json` خوانده می‌شود و در هر Stage علاوه بر schedule به lock دقیق نیاز دارد؛ schedule جای lock را نمی‌گیرد.

## ماتریس فایل‌به‌فایل: Evaluation، API و Runtime

| فایل/گروه | تغییر لازم |
|---|---|
| `evaluation/aspect_sentiment.py` | feature نسخه‌دار و point-in-time |
| `canonical_synthesis.py` | prediction عددی source of truth |
| `claim_contract.py` | horizon/as-of/evidence |
| `earnings_quality.py` | availability/feature version |
| `final_publication.py` | forecast و backtest box |
| `financial_sanity.py` | حفظ gate و feature snapshot |
| `publication_policy.py` | insufficient/stale/degraded |
| `recommendation_gate.py` | calibration/edge/data gate |
| `run_ragas.py` | فقط retrieval quality |
| `stage3_compare_cli.py` | تعمیم یا retire بعد coverage |
| `valid_availability.py` | point-in-time واقعی |
| `validate_period_consistency.py` | تفکیک period و horizon |
| `api/config.py` | forecast/version و optional observability |
| `api/core/exceptions.py` | خطاهای temporal/version |
| `api/dependencies.py`, `api/main.py` | factory/lifecycle امن |
| `api/routes/analyze.py`, `progress.py` | contract/persistence/stage جدید |
| `memory/manager_memory.py` | کلید ترکیبی/isolation |
| `main.py` | حذف auto-seed startup |
| `langsmith_trace.py` | metadata کافی، payload کم |
| `frontend/alpha-agent-app.html` | forecast/backtest UI |

## پوشش تست‌های موجود

- همهٔ `tests/unit_tests/test_agents/`: قرارداد State/Manager/Agent و regressionهای مراحل قبلی حفظ و ماهانه تکمیل شوند.
- `tests/unit_tests/financial_test/`: DCF/ratio/SEC/Yahoo به point-in-time گسترش یابند.
- `tests/unit_tests/search_test/`: tool schema و historical boundary.
- `tests/unit_tests/sentiment_test/`: window/dedupe/model version.
- `tests/unit_tests/test_rag/`: temporal vector/graph/hybrid/ingestion.
- `tests/unit_tests/api_test/`: schema، isolation، optional config و backtest route.
- تست‌های `test_aspect_sentiment_phase6.py`, `test_canonical_synthesis_phase7_1.py`, `test_claim_contract_phase6.py`, `test_earnings_quality_phase6.py`, `test_final_publication_phase7.py`, `test_financial_sanity.py`, `test_graded_publication_policy.py`, `test_manager_financial_sanity.py`, `test_multi_factor_monte_carlo_phase6.py`, `test_period_provenance_phase5.py`, `test_recommendation_gate_phase6.py`, `test_sentiment_phase4.py`, `test_stage3_narrative_stability.py` و `test_final_synthesis_hardening_phase7_2.py` baselineهای قدیمی‌اند؛ حذف نشوند و تست‌های ماهانه کنار آن‌ها بیایند.
- `tests/fixtures/stage3_ticker_regressions.json` legacy fixture بماند؛ fixture تاریخی ماهانه جدا باشد.
- `tests/integration_tests/` suite canonical پیشنهادی است؛ `integration_test_2/` تا پایان مقایسه حذف نشود.
- `tests/unit_tests/test_root_main.py`: نبود auto-seed و CLI explicit.
- `tests/unit_tests/test_supabase_phase5_migration.py`: migration واقعی و موجود.
- `tests/unit_tests/test_supbase/test_data_supabase.py`: پس از rename، از live DB جدا و marker مناسب بگیرد.
- همهٔ `__init__.py`های tests فقط برای اصلاح نام/ساختار تغییر کنند و رفتار runtime ندارند.

## فایل‌های جدید مورد نیاز

```text
core/forecast_contract.py
agents/executor_helpers/monthly_financial_features.py
agents/monthly_forecast_agent.py
agents/executor_helpers/monthly_forecast_helper.py
agents/prompts/prompt_monthly_forecast.py
tools/market_tools/__init__.py
tools/market_tools/market_data.py
tools/market_tools/feature_builder.py
tools/market_tools/historical_dataset_builder.py
tools/market_tools/forecast_model.py
tools/market_tools/market_server.py
evaluation/point_in_time_dataset.py
evaluation/monthly_backtest.py
evaluation/forecast_metrics.py
evaluation/temporal_leakage.py
evaluation/backtest_cli.py
api/routes/backtest.py
database/supabase/001_alpha_documents_v2.sql
database/supabase/002_forecast_tables.sql
database/supabase/003_analysis_versioning.sql
database/neo4j/001_temporal_evidence.cypher
tests/fixtures/monthly_point_in_time_fixture.json
tests/unit_tests/test_forecast_contract.py
tests/unit_tests/test_monthly_forecast_agent.py
tests/unit_tests/test_monthly_backtest.py
tests/unit_tests/test_temporal_leakage.py
tests/integration_tests/test_monthly_pipeline.py
```

## داده‌هایی که باید اصلاح، کوچک یا در آینده حذف شوند

### فوراً اصلاح شوند، نه حذف

- timestamp نامعتبر که اکنون می‌گیرد → `unknown/quarantine`.
- current snapshot در historical run → error/missing، نه fallback.
- memory key فقط ticker → کلید زمانی/نسخه‌دار.
- Graph edge بدون تاریخچه → evidence observation زمانی.
- payload بدون dataset/model/feature version → نسخهٔ اجباری.

### از payloadهای پایدار حذف یا reference شوند

- `brain_reasoning` و chain-of-thought.
- فهرست کامل `attempts` مگر audit خطای محدود.
- raw article chunks و متن کامل filing در هر analysis.
- raw tool request/response تکراری در trace.
- metadata تکراری موجود در document/evidence table.
- token، header، secret و query حساس.

### فقط پس از migration و تأیید حذف شوند

- tableهای legacy Supabase، edgeهای legacy Neo4j و vector namespace قدیمی.
- suite integration تکراری.
- فایل‌های typo فقط پس از rename و صفرشدن import قدیمی.
- `package-lock.json` یا `READ.md` فقط بعد از اثبات orphan بودن یا انتقال محتوای منحصربه‌فرد.

## ترتیب اجرا و تست؛ هر بار فقط یک مرحله

1. **مرحله ۰:** baseline، backup و test collection. توقف تا تأیید.
2. **مرحله ۱:** ForecastContext و State. فقط unitهای قرارداد. توقف تا تأیید.
3. **مرحله ۲:** DB V2 روی staging و reconciliation. توقف تا تأیید.
4. **مرحله ۳:** ingestion/vector زمانی و leakage fixture. توقف تا تأیید.
5. **مرحله ۴A:** Graph/hybrid زمانی و ablation؛ سپس **۴B** Research point-in-time. توقف پس از هر substage تا تأیید.
6. **مرحله ۵:** Financial point-in-time و DCF/mapper regression. توقف تا تأیید.
7. **مرحله ۶:** Sentiment point-in-time و window/dedupe. توقف تا تأیید.
8. **مرحله ۷:** ابتدا Data Readiness manifest، سپس market features/model baseline و leakage. توقف تا تأیید.
9. **مرحله ۸:** MonthlyForecastAgent/Manager و concurrency. توقف تا تأیید.
10. **مرحله ۹:** walk-forward backtest، metrics، baseline و ablation. توقف تا تأیید.
11. **مرحله ۱۰A:** runtime foundation؛ سپس **۱۰B** route/persistence/UI/E2E و smoke. توقف پس از هر substage تا تأیید.
12. **مرحله ۱۱:** rename/dependency/docs cleanup در commitهای جدا و full regression.

در پایان هر مرحله این artifactها ذخیره شوند: test report، schema/contract diff، latency، data counts، leakage report، known limitations و rollback instruction. بدون این gate وارد مرحلهٔ بعد نشوید.

## تعریف Done برای ارائه به استاد

- چند سال cut-off ماهانه point-in-time فراهم باشد؛ تعداد واقعی backtest تابع تاریخ قابل دسترس **تمام منابع** است، نه فقط قیمت.
- walk-forward backtest با label واقعی ۲۱روزه و benchmark مشخص اجرا شود.
- leakage audit ماشینی صفر future record گزارش کند.
- Full Model با baseline و ablationهای بدون Graph/Sentiment/Financial مقایسه شود.
- metricهای جهت، calibration، خطای مقدار، coverage و عملکرد سرمایه‌گذاری گزارش شوند.
- هر forecast دارای ticker، as-of، target date، horizon، evidence IDs و نسخه‌های data/feature/model/prompt باشد.
- اجرای تاریخی با همان manifest قابل تکرار باشد.
- DCF/WACC صریحاً «ارزش‌گذاری بلندمدت» معرفی و با forecast یک‌ماهه اشتباه نشوند.
- پاک‌سازی legacy فقط پس از دو shadow run موفق، reconciliation و backup قابل restore انجام شود.

## اولین اقدام اجرایی پیشنهادی

فعلاً فقط **مرحلهٔ ۰** اجرا شود: test runner را قابل استفاده کنید، backup و baseline manifest بگیرید و migration گم‌شده را تعیین تکلیف کنید. هنوز Agent یا جدول جدید نسازید و هیچ داده‌ای را پاک نکنید. بعد از سبزشدن این gate، مرحلهٔ ۱ یعنی قرارداد زمانی مرکزی امن‌ترین قدم بعدی است.
