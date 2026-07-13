"""The golden evaluation set: NL questions + reference SQL + measured answers.

Every `expected_value` here was measured from the live database (SimFin US,
FY2020-2025) — nothing is invented. The cases are chosen to each stress ONE
financial semantic that currently lives as prose in `tools.schema_tool`
(_TABLE_NOTES) rather than as machine-readable data. `semantic_rule` names the
rule a governed semantic layer would encode; a case the agent fails is direct
evidence that rule should be formalized first.

Reference SQL convention: every `reference_sql` (and `trap_sql`) returns a
single row with a numeric column aliased `answer`, so the harness can extract it
generically. All queries are plain `SELECT` / `WITH … SELECT` so they pass the
read-only SQL tool unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """One text-to-SQL evaluation case.

    Attributes:
        id: Stable identifier.
        category: The semantic category under test (also the scorecard bucket).
        question: Natural-language question fed to the agent in agent mode.
        reference_sql: Golden SQL returning the correct answer as column `answer`.
        expected_value: Ground-truth value measured from the database.
        unit: 'USD' | 'ratio' | '%' — used for display and answer grading.
        rel_tolerance: Relative tolerance for the ground-truth numeric check.
        semantic_rule: The machine-readable rule that would guard this case —
            i.e. what to formalize if the agent fails it.
        trap_sql: Optional naive/wrong query a semantics-blind agent might run.
        trap_label: Human description of that naive path.
        trap_value: The value `trap_sql` yields (None = it returns no rows).
        notes: Free-text context.
    """

    id: str
    category: str
    question: str
    reference_sql: str
    expected_value: float
    semantic_rule: str
    unit: str = "USD"
    rel_tolerance: float = 0.005
    trap_sql: str | None = None
    trap_label: str = ""
    trap_value: float | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# The cases. Values measured 2026-07 from the SimFin US dataset in Postgres.
# ---------------------------------------------------------------------------

CASES: list[EvalCase] = [
    EvalCase(
        id="flow-agg-revenue-quarters",
        category="flow-aggregation",
        question=(
            "Using Apple's quarterly income statements, what was Apple's total "
            "revenue for fiscal year 2023 (sum of Q1-Q4)?"
        ),
        reference_sql="""
            SELECT sum(f.value) AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'AAPL' AND f.statement = 'income'
              AND f.metric = 'revenue' AND f.fiscal_year = 2023
              AND f.fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
        """,
        expected_value=383_285_000_000,
        semantic_rule="aggregation=SUM (revenue is a flow; quarters sum to the year)",
        notes=(
            "The summed quarters equal the reported FY revenue (383.285B) exactly "
            "— the invariant that makes SUM the correct aggregation for flows."
        ),
    ),
    EvalCase(
        id="stock-agg-cash-period-end",
        category="stock-aggregation",
        question=(
            "What was Apple's cash, cash equivalents and short-term investments "
            "at the end of fiscal year 2023?"
        ),
        reference_sql="""
            SELECT f.value AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'AAPL' AND f.statement = 'balance'
              AND f.metric = 'cash_cash_equivalents_short_term_investments'
              AND f.fiscal_year = 2023 AND f.fiscal_period = 'Q4'
        """,
        expected_value=61_555_000_000,
        semantic_rule=(
            "aggregation=PERIOD_END (cash is a balance-sheet stock; summing "
            "quarters is invalid — the year-end value is the Q4 snapshot)"
        ),
        trap_sql="""
            SELECT sum(f.value) AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'AAPL' AND f.statement = 'balance'
              AND f.metric = 'cash_cash_equivalents_short_term_investments'
              AND f.fiscal_year = 2023 AND f.fiscal_period IN ('Q1','Q2','Q3','Q4')
        """,
        trap_label="Summing the four quarterly snapshots (treats a stock like a flow)",
        trap_value=231_264_000_000,
        notes="The flagship flow-vs-stock case: correct 61.6B vs naive-sum 231.3B.",
    ),
    EvalCase(
        id="period-map-balance-fy",
        category="period-mapping",
        question="What were Apple's total assets for fiscal year 2023?",
        reference_sql="""
            SELECT f.value AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'AAPL' AND f.statement = 'balance'
              AND f.metric = 'total_assets'
              AND f.fiscal_year = 2023 AND f.fiscal_period = 'Q4'
        """,
        expected_value=352_583_000_000,
        semantic_rule=(
            "period_semantics: balance sheets have no 'FY' row; the Q4 snapshot "
            "IS the fiscal-year-end balance"
        ),
        trap_sql="""
            SELECT f.value AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'AAPL' AND f.statement = 'balance'
              AND f.metric = 'total_assets'
              AND f.fiscal_year = 2023 AND f.fiscal_period = 'FY'
        """,
        trap_label="Filtering balance for fiscal_period='FY' (returns nothing)",
        trap_value=None,
        notes="A literal FY filter on a balance metric yields zero rows.",
    ),
    EvalCase(
        id="derived-eps",
        category="derived-metric",
        question="What was Apple's diluted earnings per share (EPS) for fiscal year 2023?",
        reference_sql="""
            SELECT ni.value / NULLIF(sd.value, 0) AS answer
            FROM (
                SELECT f.value
                FROM financials f JOIN companies c ON c.id = f.company_id
                WHERE c.ticker = 'AAPL' AND f.statement = 'income'
                  AND f.metric = 'net_income'
                  AND f.fiscal_year = 2023 AND f.fiscal_period = 'FY'
            ) ni,
            (
                SELECT f.value
                FROM financials f JOIN companies c ON c.id = f.company_id
                WHERE c.ticker = 'AAPL' AND f.statement = 'income'
                  AND f.metric = 'shares_diluted'
                  AND f.fiscal_year = 2023 AND f.fiscal_period = 'FY'
            ) sd
        """,
        expected_value=6.1341,
        unit="ratio",
        rel_tolerance=0.01,
        semantic_rule=(
            "derived_metric: eps = net_income / shares_diluted (there is NO "
            "stored eps metric — it must be computed)"
        ),
        notes="net_income 96.995B / shares_diluted 15.8125B ≈ 6.13.",
    ),
    EvalCase(
        id="entity-resolution-ford",
        category="entity-resolution",
        question="What was Ford's total revenue in fiscal year 2023?",
        reference_sql="""
            SELECT f.value AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'F' AND f.statement = 'income'
              AND f.metric = 'revenue'
              AND f.fiscal_year = 2023 AND f.fiscal_period = 'FY'
        """,
        expected_value=176_191_000_000,
        semantic_rule=(
            "entity_resolution: 'Ford' -> Ford Motor Co (ticker F), NOT Forward "
            "Industries (ticker FORD)"
        ),
        trap_sql="""
            SELECT f.value AS answer
            FROM financials f JOIN companies c ON c.id = f.company_id
            WHERE c.ticker = 'FORD' AND f.statement = 'income'
              AND f.metric = 'revenue'
              AND f.fiscal_year = 2023 AND f.fiscal_period = 'FY'
        """,
        trap_label="Resolving 'Ford' to ticker FORD (Forward Industries)",
        trap_value=36_688_307,
        notes="Correct 176.2B (Ford Motor) vs trap 36.7M (Forward Industries).",
    ),
    EvalCase(
        id="computation-revenue-growth",
        category="computation",
        question=(
            "What was the percentage change in Apple's annual revenue from fiscal "
            "year 2022 to fiscal year 2023?"
        ),
        reference_sql="""
            WITH r AS (
                SELECT f.fiscal_year, f.value
                FROM financials f JOIN companies c ON c.id = f.company_id
                WHERE c.ticker = 'AAPL' AND f.statement = 'income'
                  AND f.metric = 'revenue' AND f.fiscal_period = 'FY'
                  AND f.fiscal_year IN (2022, 2023)
            )
            SELECT round(
                100.0 * (
                    max(value) FILTER (WHERE fiscal_year = 2023)
                    - max(value) FILTER (WHERE fiscal_year = 2022)
                ) / max(value) FILTER (WHERE fiscal_year = 2022), 2
            ) AS answer
            FROM r
        """,
        expected_value=-2.80,
        unit="%",
        rel_tolerance=0.02,
        semantic_rule=(
            "computation: growth = (current - prior) / prior on FY-aligned rows "
            "(revenue fell 394.3B -> 383.3B)"
        ),
        notes="Grading is magnitude-based, so 'declined 2.8%' and '-2.8%' both pass.",
    ),
]
