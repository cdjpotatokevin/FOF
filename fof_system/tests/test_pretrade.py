import pandas as pd

from fof_system.engine.pretrade import evaluate_pretrade_status, normalize_fund_code


def test_pretrade_fails_closed_when_limit_evidence_is_missing():
    orders = pd.DataFrame({"code": ["22003"], "name": ["博道大盘成长股票A"]})
    live = pd.DataFrame({"code": ["022003"], "name": ["博道大盘成长股票A"], "subscription_status": ["开放申购"]})

    report, summary = evaluate_pretrade_status(
        orders, live, asof="2026-06-25", source="csv", require_limit_evidence=True,
    )

    assert normalize_fund_code("22003.OF") == "022003"
    assert summary.ok is False
    assert report.loc[0, "reason"] == "missing_limit_evidence_columns"


def test_pretrade_blocks_status_and_explicit_daily_limit():
    orders = pd.DataFrame({"code": ["000001", "000002", "000003"]})
    live = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "asset_type": ["fund", "fund", "fund"],
        "subscription_status": ["开放申购", "暂停大额申购", "开放申购"],
        "daily_subscription_limit_yi": [pd.NA, pd.NA, 0.1],
        "has_daily_subscription_limit": [False, False, False],
    })

    report, summary = evaluate_pretrade_status(
        orders, live, asof="2026-06-25", source="csv", require_limit_evidence=True,
    )

    assert summary.failed == 2
    reasons = dict(zip(report["code"], report["reason"]))
    assert reasons["000001"] == "open_no_limit"
    assert reasons["000002"] == "blocked_subscription_status"
    assert reasons["000003"] == "daily_subscription_limit_present"


def test_pretrade_marks_etf_as_secondary_market_subscription_not_checked():
    orders = pd.DataFrame({"code": ["159307"], "type": ["ETF"]})
    live = pd.DataFrame({"code": ["159307"], "asset_type": ["etf"], "subscription_status": [""]})

    report, summary = evaluate_pretrade_status(
        orders, live, asof="2026-06-25", source="csv", require_limit_evidence=True,
    )

    assert summary.ok is True
    assert report.loc[0, "reason"] == "etf_secondary_market_not_subscription_checked"
