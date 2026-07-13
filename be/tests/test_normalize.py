"""Unit tests for SimFin normalization (no network or database required)."""

from __future__ import annotations

import pandas as pd

from ingestion.simfin_ingest import normalize_simfin_data


def _income_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "SimFinId": 111,
                "Currency": "USD",
                "Fiscal Year": 2023,
                "Fiscal Period": "FY",
                "Report Date": "2023-09-30",
                "Revenue": 383285.0,
                "Net Income": 96995.0,
            },
            {
                "Ticker": "MSFT",
                "SimFinId": 222,
                "Currency": "USD",
                "Fiscal Year": 2023,
                "Fiscal Period": "FY",
                "Report Date": "2023-06-30",
                "Revenue": 211915.0,
                "Net Income": 72361.0,
            },
        ]
    )


def test_normalizes_to_long_format() -> None:
    records = normalize_simfin_data(_income_frame(), statement="income")

    # 2 companies x 2 metrics = 4 long rows.
    assert len(records) == 4
    assert {r.metric for r in records} == {"revenue", "net_income"}

    aapl_revenue = next(
        r for r in records if r.ticker == "AAPL" and r.metric == "revenue"
    )
    assert aapl_revenue.statement == "income"
    assert aapl_revenue.fiscal_year == 2023
    assert aapl_revenue.fiscal_period == "FY"
    assert aapl_revenue.currency == "USD"
    assert aapl_revenue.value == 383285.0
    assert str(aapl_revenue.report_date) == "2023-09-30"
    assert aapl_revenue.simfin_id == "111"  # coerced to a clean string, not '111.0'


def test_snake_cases_metric_names() -> None:
    df = pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "Fiscal Year": 2023,
                "Fiscal Period": "Q1",
                "Shares (Diluted)": 15_500.0,
            }
        ]
    )
    records = normalize_simfin_data(df, statement="income")
    assert [r.metric for r in records] == ["shares_diluted"]
    assert records[0].fiscal_period == "Q1"  # quarterly granularity preserved


def test_skips_null_metric_values() -> None:
    df = pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "Fiscal Year": 2023,
                "Fiscal Period": "FY",
                "Revenue": 100.0,
                "Net Income": None,
            }
        ]
    )
    records = normalize_simfin_data(df, statement="income")
    assert len(records) == 1
    assert records[0].metric == "revenue"
