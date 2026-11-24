# Stage 04B revision 1 handoff

Stage 04B revision 1 is CLOSED on accepted Attempt 3. Stage 05 remains PENDING / NOT_STARTED and is not activated by this handoff.

## Accepted evidence

- Primary: research
- Observability: AUDITED; all requested changes resolved
- QA: PASS
- Review: APPROVED
- Checker privacy sentinel: 1 passed
- Focused Stage 04B: 122 passed
- Complete offline search suite: 66 passed
- Graph/RAG regressions: 333 passed
- Collection: 1,314 tests, zero errors
- Integrity: 20/20 current hashes and 11/11 prior-attempt artifact hashes matched

## Delivered contract

Research now carries one mandatory ForecastContext and run identity through schemas, dispatcher, executor, agent and providers. Historical News, SEC and Tavily execution fails closed unless retained versioned evidence is available, never substitutes current web, enforces SEC filed-at eligibility, and rejects future or unknown evidence both before synthesis and before Shared State.

Evidence and citations are deterministic and version-aware. State persists typed compact ResearchFeatures, reference identifiers and counts rather than raw articles, filings, citations or synthesis payloads. Vector-only, Graph-only and Hybrid modes retain explicit degradation behavior and the accepted temporal retrieval contract.

Trace, MCP, lifecycle, log, progress, Sentry and State metadata is compact and allowlisted. All Stage 04B runtime provenance reports revision 1 attempt 3 through `stage04b-r1-a3`; raw query, plan, arguments, exceptions, URLs, evidence, citations, synthesis, checker missing text and checker score are excluded.

## Retry and operational boundary

Attempt 1 failed QA because the evidence package contained one incorrect recorded hash and the direct Research server unit suite had not migrated to mandatory identity. Attempt 2 fixed those issues but Review rejected stale Attempt 1 telemetry provenance. Attempt 3 fixed provenance and a final checker privacy finding, then passed QA and Review.

One legacy integration probe during Attempt 1 accidentally made read-only News and SEC calls. It was stopped immediately, made no external or database write, was not rerun, and was not used as acceptance evidence. No accepted-attempt network access, live provider test, migration, seed, deletion, deploy, commit or push occurred. Live-provider behavior remains unverified until separately authorized.

Stage 05 remains inactive and requires explicit activation.
