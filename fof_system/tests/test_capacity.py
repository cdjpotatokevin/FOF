import pandas as pd

from fof_system.engine.capacity import etf_capacity_report, etf_capacity_weight_caps


def test_capacity_report_calculates_participation_and_portfolio_ceiling():
    market = pd.DataFrame({
        "code": ["159915", "159915", "510300"],
        "trading_date": ["2026-06-20", "2026-06-23", "2026-06-23"],
        "amount": [100_000_000, 300_000_000, 500_000_000],
    })
    report = etf_capacity_report(
        market, pd.Series({"159915": 0.25}), participation_rate=0.10,
        portfolio_aum_yi=4.0,
    )
    assert report.loc["159915", "adv_yi"] == 2.0
    assert report.loc["159915", "max_portfolio_yi"] == 0.8
    assert bool(report.loc["159915", "within_limit"]) is False


def test_capacity_weight_cap_uses_adv_and_blocks_missing_market_data():
    market = pd.DataFrame({
        "code": ["159915", "159915"],
        "trading_date": ["2026-06-20", "2026-06-23"],
        "amount": [100_000_000, 300_000_000],
    })
    caps = etf_capacity_weight_caps(
        market, ["159915", "510300"], portfolio_aum_yi=4.0,
        participation_rate=0.10,
    )
    assert caps["159915"] == 0.05  # 10% × 2亿元ADV / 4亿元组合
    assert caps["510300"] == 0.0
