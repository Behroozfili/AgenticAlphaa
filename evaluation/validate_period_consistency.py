"""
validate_period_consistency.py

Companion checker to validate_numeric_faithfulness.py.

The faithfulness checker answers: "does every number in final_report trace
back to *some* number in the source data?"

This checker answers a different, more subtle question: "were the numbers
that get narrated together in the same section actually computed from the
*same reporting period*?"

Why this matters (real bug found in NVDA trace_9):
    - gross_margin (74.9%) and operating_margin (65.6%) were narrated
      straight from the Q1 FY2027 10-Q text (quarterly).
    - net_margin (55.6%) was computed by financial_agent.py from
      annual_revenue[0] / annual_net_income[0] (FY2026 *annual* figures),
      because the underlying XBRL extractor
      (sec_edgar.py._extract(..., form="10-K")) only pulls annual filings
      by default -- there is no quarterly income-statement series in the
      pipeline at all.
    - Every individual number was internally correct (net_income/revenue
      *were* same-period with each other), but the report presented all
      three margins side-by-side as if they described the same quarter.
      That's a period-consistency bug, not a faithfulness bug -- and
      validate_numeric_faithfulness.py cannot catch it, because 55.6% DOES
      trace back to real ground-truth data. It's just the wrong period's
      ground truth.

Method
------
1. Build a registry of known "periods" from raw_numerical_data:
     - annual:<year>            -> {revenue, net_income}   (10-K based)
     - quarterly:<period_end>    -> {value, concept}         (10-Q based, if present)
     - ttm                       -> {revenue, net_income}   (yahoo trailing figures)
2. For each computed ratio, recover which period its inputs actually came
   from -- either by explicit provenance tags (preferred; see
   `RatioInput.period_type`) or, if the calling agent didn't tag it, by
   value-matching the input numbers back against the period registry
   (fallback; robust to un-instrumented legacy code).
3. Group ratios by the report section they'll be narrated in together
   (e.g. "Financial Health" / "Profitability & Margins") and flag any
   group whose members don't all share the same period_type + period key.

Usage
-----
    python validate_period_consistency.py --trace state.json

`state.json` should contain (at minimum) the keys the real pipeline
produces: raw_numerical_data (yahoo_ratios, revenue_growth,
xbrl_financials) and calculated_ratios. See `_load_state` for the exact
shape expected, and `_demo_state()` for a worked example reconstructed
from the actual NVDA bug.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any


TOLERANCE = 0.005  # 0.5% relative tolerance when value-matching floats


def _close(a: float | None, b: float | None, tol: float = TOLERANCE) -> bool:
    if a is None or b is None:
        return False
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= tol


# --------------------------------------------------------------------------
# Period registry
# --------------------------------------------------------------------------

@dataclass
class PeriodRegistry:
    """All known (period_type, period_key) -> {field_name: value} facts."""
    annual: dict[str, dict[str, float]] = field(default_factory=dict)      # key = year
    quarterly: dict[str, dict[str, float]] = field(default_factory=dict)  # key = period_end
    ttm: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_raw_numerical_data(cls, raw: dict[str, Any]) -> "PeriodRegistry":
        reg = cls()

        growth = raw.get("revenue_growth", {})
        annual_rev = growth.get("annual_revenue", [])
        annual_ni = growth.get("annual_net_income", [])
        for entry in annual_rev:
            year = str(entry.get("year"))
            reg.annual.setdefault(year, {})["revenue"] = entry.get("revenue")
        for entry in annual_ni:
            year = str(entry.get("year"))
            reg.annual.setdefault(year, {})["net_income"] = entry.get("net_income")

        if "revenue_growth_ttm" in growth:
            reg.ttm["revenue_growth_ttm"] = growth["revenue_growth_ttm"]

        xbrl = raw.get("xbrl_financials", {})
        for concept, entries in xbrl.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                pe = entry.get("period_end")
                val = entry.get("value")
                if pe is None or val is None:
                    continue
                reg.quarterly.setdefault(pe, {})[concept] = val

        yahoo = raw.get("yahoo_ratios", {})
        for k in ("eps_trailing", "current_price", "price_to_sales", "market_cap"):
            if k in yahoo:
                reg.ttm[k] = yahoo[k]

        return reg

    def match_period(self, value: float | None, field_hint: str | None = None) -> list[str]:
        """Return human-readable period labels whose registered value matches `value`."""
        if value is None:
            return []
        hits = []
        for year, fields_ in self.annual.items():
            for fname, fval in fields_.items():
                if field_hint and fname != field_hint:
                    continue
                if _close(value, fval):
                    hits.append(f"annual:{year}")
        for pe, fields_ in self.quarterly.items():
            for fname, fval in fields_.items():
                if field_hint and field_hint not in fname.lower():
                    continue
                if _close(value, fval):
                    hits.append(f"quarterly:{pe}")
        return hits


# --------------------------------------------------------------------------
# Ratio inputs (what actually got passed into each calculator function)
# --------------------------------------------------------------------------

@dataclass
class RatioInput:
    name: str                      # e.g. "net_margin"
    section: str                   # report section it's narrated in, e.g. "Financial Health"
    inputs: dict[str, float | None]  # e.g. {"net_income": 120e9, "revenue": 215.9e9}
    stated_period: str | None = None  # explicit tag from the agent, if it set one
    narrated_period: str | None = None  # what the *report text* claims this describes


def infer_periods(ratio: RatioInput, registry: PeriodRegistry) -> set[str]:
    """Best-effort recovery of which period(s) this ratio's inputs came from."""
    if ratio.stated_period:
        return {ratio.stated_period}

    candidate_sets = []
    for fname, fval in ratio.inputs.items():
        hits = registry.match_period(fval, field_hint=fname)
        if hits:
            candidate_sets.append(set(hits))

    if not candidate_sets:
        return {"unknown"}

    common = candidate_sets[0]
    for s in candidate_sets[1:]:
        common &= s
    return common if common else {"MIXED(" + ",".join(sorted(set().union(*candidate_sets))) + ")"}


# --------------------------------------------------------------------------
# Consistency check
# --------------------------------------------------------------------------

def check_section_consistency(ratios: list[RatioInput], registry: PeriodRegistry) -> list[dict]:
    """Group ratios by section; flag sections whose members disagree on period."""
    findings = []
    by_section: dict[str, list[RatioInput]] = {}
    for r in ratios:
        by_section.setdefault(r.section, []).append(r)

    for section, members in by_section.items():
        resolved = {r.name: infer_periods(r, registry) for r in members}

        # Does any member's own narrated_period conflict with its inferred period?
        for r in members:
            periods = resolved[r.name]
            if r.narrated_period and periods and "unknown" not in periods:
                if not any(r.narrated_period in p or p in r.narrated_period for p in periods):
                    findings.append({
                        "severity": "HIGH",
                        "section": section,
                        "metric": r.name,
                        "issue": "narrated period does not match inferred source period",
                        "narrated_as": r.narrated_period,
                        "actually_from": sorted(periods),
                        "inputs": r.inputs,
                    })

        # Do the metrics in this section disagree with each other on period,
        # even if none of them individually mislabeled themselves?
        flat_periods = set()
        for periods in resolved.values():
            flat_periods |= periods
        flat_periods.discard("unknown")
        distinct_non_mixed = {p for p in flat_periods if not p.startswith("MIXED")}
        if len(distinct_non_mixed) > 1:
            findings.append({
                "severity": "HIGH",
                "section": section,
                "metric": "(section-wide)",
                "issue": "metrics narrated together come from different reporting periods",
                "members": {r.name: sorted(resolved[r.name]) for r in members},
            })

    return findings


# --------------------------------------------------------------------------
# Production entry point — reads the REAL shape of shared_state
# --------------------------------------------------------------------------
# financial_agent.py nests each ratio dict (with its now-present "_period"
# and "_inputs" keys) directly into financial_metrics_summary — see
# financial_agent.py line ~1309: "net_margin": calc.get("net_margin", {}).
# That summary + the LLM-written final_report text are both sitting in
# shared_state by the time _node_finalise() runs in manager_agent.py, which
# is the natural hook point (see the wiring example in the module docstring
# / accompanying integration snippet).

QUARTER_MARKERS = re.compile(
    r"\bQ[1-4]\b|\bquarter(ly)?\b|\bthree[- ]months?\b|\bsequential(ly)?\b|\bQoQ\b",
    re.IGNORECASE,
)
INTERIM_MARKERS = re.compile(
    r"\bnine[- ]months?\b|\bsix[- ]months?\b|\byear[- ]to[- ]date\b|\bYTD\b|"
    r"\bmonths? ended\b|\bfirst (six|nine) months\b",
    re.IGNORECASE,
)
ANNUAL_MARKERS = re.compile(
    r"\bfiscal year\b|\bFY\d{2,4}\s+annual\b|\bfull[- ]year\b|\bannual(ly)?\b|\btwelve months\b",
    re.IGNORECASE,
)

CONTEXT_WINDOW = 160  # chars of surrounding text to inspect for period cues


def _find_value_context(report_text: str, value: float, window: int = CONTEXT_WINDOW) -> list[str]:
    """Locate where a numeric value (e.g. a _pct field) appears in the report
    text and return the surrounding context window(s)."""
    if value is None:
        return []
    # NOTE: deliberately NOT including round(value) as a candidate — for a
    # value like 0.8, round() gives the single digit "1", which then
    # matches virtually anywhere in the text (list numbering "1.", "Q1",
    # "18%", any number containing a "1"...) and produces garbage context.
    # 2-decimal and 1-decimal formatted strings are specific enough to be
    # meaningful on their own; if a report rounds to a different precision
    # than either of these, that's a separate (rarer) faithfulness issue,
    # not something this period-consistency check needs to chase.
    candidates = {f"{value:.1f}", f"{value:.2f}"}
    contexts = []
    for cand in candidates:
        # (?<![\d.]) / (?![\d.]) : the match must not be immediately
        # preceded or followed by a digit or '.', so "38" cannot match
        # inside "386.74" or inside "138.5", and "0.8" cannot match inside
        # "10.80" or "0.85". This is stricter than \b, which does NOT
        # exclude digit-adjacent matches (both sides of a digit run are
        # "word" characters with no boundary between them, but a boundary
        # DOES appear right before a following '.', which \b alone can't
        # rule out).
        pattern = re.compile(r"(?<![\d.])" + re.escape(cand) + r"(?![\d.])")
        for m in pattern.finditer(report_text):
            start = max(0, m.start() - window)
            end = min(len(report_text), m.end() + window)
            contexts.append(report_text[start:end])
    return contexts


def check_narration_vs_period(final_report: str, financial_metrics_summary: dict) -> list[dict]:
    """
    For every ratio in financial_metrics_summary that carries a '_period'
    tag, find where its value is narrated in final_report and check whether
    the surrounding prose's period language (quarterly / interim-YTD /
    annual markers) matches the tag. This is what catches "a number from
    one period narrated as if it described a different one" automatically,
    directly against the real report text — covers both the original NVDA
    case (annual net_margin narrated as "Q1 FY2027") and the MSFT case (a
    standalone-quarter net_margin narrated next to "nine-month period"
    cumulative YTD revenue/income figures — a real, distinct bug: MSFT's
    10-Q reports both a 3-month-standalone and a 9-month-YTD column, and
    the report mixed the former's margin with the latter's raw numbers).
    """
    findings = []
    value_field_by_metric = {
        "net_margin": "net_margin_pct", "roe": "roe_pct", "pe_ratio": "pe_ratio",
        "de_ratio": "de_ratio", "revenue_cagr": "cagr_pct",
    }
    tag_prefix_to_category = {"quarterly:": "quarterly", "annual:": "annual"}
    category_markers = {
        "quarterly": QUARTER_MARKERS,
        "interim":   INTERIM_MARKERS,
        "annual":    ANNUAL_MARKERS,
    }

    for metric_key, value_field in value_field_by_metric.items():
        entry = financial_metrics_summary.get(metric_key, {})
        if not isinstance(entry, dict):
            continue
        value = entry.get(value_field)
        period = entry.get("_period")
        if value is None or not period:
            continue

        tagged_category = next(
            (cat for prefix, cat in tag_prefix_to_category.items() if period.startswith(prefix)),
            None,
        )
        if tagged_category is None:
            continue  # "ttm"/"mrq"/"mixed"/"derived:..." aren't quarter/interim/annual claims

        for ctx in _find_value_context(final_report, value):
            present = {cat for cat, pattern in category_markers.items() if pattern.search(ctx)}
            if not present or tagged_category in present:
                continue  # no period language nearby, or it agrees with the tag

            # Text mentions a DIFFERENT period category than the one this
            # value is actually tagged with, and does not also mention the
            # correct one.
            other = sorted(present)
            severity = "HIGH" if "interim" in present or tagged_category == "annual" else "MEDIUM"
            findings.append({
                "severity": severity,
                "metric": metric_key,
                "issue": (
                    f"{tagged_category}-period figure narrated with "
                    f"{'/'.join(other)} language"
                ),
                "value": value,
                "tagged_period": period,
                "report_context": ctx.strip(),
            })
    return findings


# --------------------------------------------------------------------------
# Loading a real trace
# --------------------------------------------------------------------------

def _load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_ratio_inputs_from_state(state: dict, section_map: dict[str, str],
                                   narration_map: dict[str, str] | None = None) -> list[RatioInput]:
    """
    section_map:   {ratio_key: report_section_name}, e.g.
                   {"gross_margin": "Financial Health", "net_margin": "Financial Health"}
    narration_map: {ratio_key: period the *report text* claims}, e.g.
                   {"net_margin": "quarterly"}  # report said "Q1 FY2027 net margin"
    """
    narration_map = narration_map or {}
    calc = state.get("calculated_ratios", {})
    out = []
    for key, section in section_map.items():
        entry = calc.get(key, {})
        inputs = entry.get("_inputs", {})  # agents should log/store these; see note below
        out.append(RatioInput(
            name=key,
            section=section,
            inputs=inputs,
            stated_period=entry.get("_period"),
            narrated_period=narration_map.get(key),
        ))
    return out


# --------------------------------------------------------------------------
# Worked example -- reconstructed from the real NVDA trace_9 bug
# --------------------------------------------------------------------------

def _demo_state() -> dict:
    """Reconstructs the actual NVDA numbers we traced by hand, so the checker
    is demonstrably correct against a real, already-diagnosed bug."""
    return {
        "raw_numerical_data": {
            "revenue_growth": {
                "annual_revenue": [
                    {"year": 2026, "revenue": 215938000000, "yoy_growth": 0.6547},
                    {"year": 2025, "revenue": 130497000000, "yoy_growth": 1.142},
                ],
                "annual_net_income": [
                    # FY2026 annual net income implied by 55.6% margin claim:
                    # 215.938e9 * 0.556 ~= 120.06e9 (not directly visible in the
                    # trace export we received -- reconstructed for the demo).
                    {"year": 2026, "net_income": 120061528000},
                    {"year": 2025, "net_income": 72880000000},
                ],
                "revenue_growth_ttm": 0.852,
            },
            "xbrl_financials": {
                "Revenues": [
                    {"period_end": "2026-04-26", "value": 81615000000},
                ],
                "NetIncomeLoss": [
                    {"period_end": "2026-04-26", "value": 58321000000},
                ],
                "OperatingIncomeLoss": [
                    {"period_end": "2026-04-26", "value": 53536000000},
                ],
            },
            "yahoo_ratios": {
                "current_price": 195.55,
                "eps_trailing": 6.53,
            },
        },
        "calculated_ratios": {
            "gross_margin": {
                "gross_margin_pct": 74.9,
                "_inputs": {"revenue": 81615000000, "cogs": 20485566000},  # 25.1% of revenue
                "_period": "quarterly:2026-04-26",
            },
            "operating_margin": {
                "operating_margin_pct": 65.6,
                "_inputs": {"operating_income": 53536000000, "revenue": 81615000000},
                "_period": "quarterly:2026-04-26",
            },
            "net_margin": {
                "net_margin_pct": 55.6,
                # THE BUG: these are annual_net_income[0] / annual_revenue[0],
                # not the quarterly XBRL figures used by the other two ratios.
                "_inputs": {"net_income": 120061528000, "revenue": 215938000000},
                # deliberately NOT tagging _period, to demonstrate the
                # value-matching fallback recovering it independently
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", help="Path to a JSON state dump. Omit to run the built-in demo.")
    args = parser.parse_args()

    state = _load_state(args.trace) if args.trace else _demo_state()
    registry = PeriodRegistry.from_raw_numerical_data(state["raw_numerical_data"])

    section_map = {
        "gross_margin": "Financial Health / Profitability & Margins",
        "operating_margin": "Financial Health / Profitability & Margins",
        "net_margin": "Financial Health / Profitability & Margins",
    }
    # What the final_report text actually claimed about each metric's period.
    # In the real NVDA report, all three were narrated as "Q1 FY2027".
    narration_map = {
        "gross_margin": "quarterly",
        "operating_margin": "quarterly",
        "net_margin": "quarterly",   # <- report said "Q1 FY2027", inputs say annual
    }

    ratios = build_ratio_inputs_from_state(state, section_map, narration_map)
    findings = check_section_consistency(ratios, registry)

    print(f"Checked {len(ratios)} ratios across "
          f"{len(set(section_map.values()))} report section(s).\n")

    if not findings:
        print("No period-consistency issues found.")
        return

    for f in findings:
        print(f"[{f['severity']}] section={f['section']!r} metric={f['metric']!r}")
        print(f"    issue: {f['issue']}")
        for k, v in f.items():
            if k in ("severity", "section", "metric", "issue"):
                continue
            print(f"    {k}: {v}")
        print()


if __name__ == "__main__":
    main()