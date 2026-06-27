"""第①层PIT数据合同测试。"""
from dataclasses import replace

import pandas as pd
import pytest

from fof_system.config import DEFAULT_FILTER
from fof_system.data import PITDataError, PITDataStore
from fof_system.data import get_provider
from fof_system.engine.universe import filter_universe, management_company_allowed
from fof_system.pipeline import score_universe
from fof_system.run_data import _stock_etf_pool_from_pit


def test_universe_reads_only_records_available_at_asof(tmp_path):
    store = PITDataStore(tmp_path / "pit")
    first = pd.DataFrame([
        {"code": "000001", "name": "国内主动基金", "asset_type": "fund", "fund_type": "偏股混合型"},
        {"code": "510300", "name": "沪深300ETF", "asset_type": "etf", "fund_type": "ETF", "is_stock_etf": True},
    ])
    store.write_universe_snapshot(first, source="vendor", available_at="2024-01-02")

    # 该状态在3月1日生效，但3月20日才可获得；在3月10日的历史回测中不能使用。
    second = first.copy()
    second.loc[second["code"] == "000001", "status"] = "suspended"
    store.write_universe_snapshot(
        second, source="vendor", effective_date="2024-03-01", available_at="2024-03-20",
    )

    before_disclosure = store.read_universe_asof("2024-03-10", active_only=False).set_index("code")
    after_disclosure = store.read_universe_asof("2024-03-20", active_only=False).set_index("code")
    assert before_disclosure.loc["000001", "status"] == "active"
    assert after_disclosure.loc["000001", "status"] == "suspended"
    assert "000001" not in set(store.read_universe_asof("2024-03-20")["code"])


def test_market_data_uses_availability_lag(tmp_path):
    store = PITDataStore(tmp_path / "pit")
    market = pd.DataFrame({
        "trading_date": ["2024-01-02", "2024-01-03"],
        "code": ["510300", "510300"],
        "close": [3.50, 3.55],
        "amount": [1_000_000, 1_100_000],
    })
    write = store.write_market_data(market, source="vendor:etf", availability_lag_days=1)
    assert write.path.exists() and write.manifest_path.exists()
    assert store.read_market_asof("510300", "2024-01-02").empty
    visible = store.read_market_asof("510300", "2024-01-03")
    assert len(visible) == 1
    assert visible.loc[0, "close"] == 3.50


def test_market_data_rejects_duplicate_or_invalid_prices(tmp_path):
    store = PITDataStore(tmp_path / "pit")
    bad = pd.DataFrame({
        "trading_date": ["2024-01-02", "2024-01-02"],
        "code": ["510300", "510300"], "close": [3.50, -1.0],
    })
    with pytest.raises(PITDataError, match="数据质量检查失败"):
        store.write_market_data(bad, source="vendor:etf")


def test_holdings_reads_only_latest_disclosed_report_and_preserves_codes(tmp_path):
    store = PITDataStore(tmp_path / "pit")
    q1 = pd.DataFrame({
        "fund_code": ["000001"], "security_code": ["000001"],
        "security_name": ["平安银行"], "weight": [0.08], "industry": ["银行"],
    })
    q2 = pd.DataFrame({
        "fund_code": ["000001"], "security_code": ["000002"],
        "security_name": ["万科A"], "weight": [0.09], "industry": ["地产"],
    })
    store.write_holdings_snapshot(q1, source="vendor", report_period="2024-03-31", available_at="2024-04-25")
    store.write_holdings_snapshot(q2, source="vendor", report_period="2024-06-30", available_at="2024-07-25")

    before_q2 = store.read_holdings_asof("2024-07-01", "000001")
    after_q2 = store.read_holdings_asof("2024-07-25", "000001")
    assert before_q2.loc[0, "security_code"] == "000001"
    assert after_q2.loc[0, "security_code"] == "000002"


def test_universe_filter_excludes_qdii_from_type_field():
    universe = pd.DataFrame([
        {"code": "A", "name": "国内ETF", "fund_type": "ETF"},
        {"code": "B", "name": "美股指数", "fund_type": "QDII-股票型"},
        {"code": "C", "name": "海外ETF", "fund_type": "ETF QDII"},
    ])
    filtered = filter_universe(universe)
    assert set(filtered["code"]) == {"A"}


def test_strict_investment_universe_applies_fund_and_stock_etf_thresholds():
    universe = pd.DataFrame([
        {"code": "F_OK", "name": "主动权益", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 2.0, "inception": "2023-01-01", "subscription_status": "开放申购"},
        {"code": "F_SMALL", "name": "小规模主动", "asset_type": "fund", "fund_type": "偏股混合型",
         "aum_yi": 1.99, "inception": "2022-01-01", "subscription_status": "开放申购"},
        {"code": "F_NEW", "name": "新基金", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2024-08-01", "subscription_status": "开放申购"},
        {"code": "F_PAUSED", "name": "暂停申购主动", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "暂停大额申购"},
        {"code": "F_LIMITED", "name": "限大额主动", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "限大额"},
        {"code": "F_DAILY", "name": "单日限额主动", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "开放申购",
         "daily_subscription_limit_yi": 0.1},
        {"code": "E_OK", "name": "股票ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": True, "is_qdii": False, "aum_yi": 5.0, "inception": "2024-01-01"},
        {"code": "E_SMALL", "name": "小ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": True, "is_qdii": False, "aum_yi": 4.99, "inception": "2020-01-01"},
        {"code": "E_QDII", "name": "海外ETF", "asset_type": "etf", "fund_type": "ETF QDII",
         "is_stock_etf": True, "is_qdii": True, "aum_yi": 20.0, "inception": "2020-01-01"},
    ])
    filtered = filter_universe(
        universe, asof="2025-07-31", strict_eligibility=True,
        flt=replace(DEFAULT_FILTER, allowed_management_companies=[], require_manager_tenure=False),
    )
    assert set(filtered["code"]) == {"F_OK", "E_OK"}


def test_management_company_whitelist():
    assert management_company_allowed("易方达基金管理有限公司", ["易方达基金"])
    assert not management_company_allowed("财通基金管理有限公司", ["易方达基金"])
    assert management_company_allowed("中银国际证券股份有限公司", ["中银国际证券", "中银基金"])
    assert management_company_allowed("中银基金管理有限公司", ["中银国际证券", "中银基金"])
    assert not management_company_allowed("上海国泰海通证券资产管理有限公司", ["国泰基金"])

    universe = pd.DataFrame([
        {"code": "F_YFD", "name": "易方达产品", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "开放申购",
         "management_company": "易方达基金管理有限公司", "manager_start": "2020-01-01"},
        {"code": "F_CT", "name": "财通产品", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "开放申购",
         "management_company": "财通基金管理有限公司", "manager_start": "2020-01-01"},
        {"code": "F_NEW_MGR", "name": "新经理产品", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "开放申购",
         "management_company": "易方达基金管理有限公司", "manager_start": "2025-01-01"},
        {"code": "E_YFD", "name": "易方达ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": True, "is_qdii": False, "aum_yi": 10.0, "inception": "2020-01-01",
         "management_company": "易方达基金管理有限公司"},
    ])
    filtered = filter_universe(universe, asof="2025-07-31", strict_eligibility=True)
    assert set(filtered["code"]) == {"F_YFD", "E_YFD"}


def test_manager_tenure_requires_one_year_on_fund():
    universe = pd.DataFrame([
        {"code": "F_OK", "name": "老经理", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "开放申购",
         "management_company": "易方达基金管理有限公司", "manager_start": "2020-01-01"},
        {"code": "F_NEW", "name": "新经理", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 10.0, "inception": "2020-01-01", "subscription_status": "开放申购",
         "management_company": "易方达基金管理有限公司", "manager_start": "2025-01-01"},
    ])
    filtered = filter_universe(universe, asof="2025-07-31", strict_eligibility=True)
    assert set(filtered["code"]) == {"F_OK"}


def test_universe_excludes_passive_index_fund_and_stock_fof_but_keeps_labeled_etf():
    universe = pd.DataFrame([
        {"code": "PASSIVE", "name": "普通指数", "asset_type": "fund", "fund_type": "被动指数型股票基金"},
        {"code": "FOF", "name": "股票FOF", "asset_type": "fund", "fund_type": "股票型FOF"},
        {"code": "ACTIVE", "name": "增强指数", "asset_type": "fund", "fund_type": "增强指数型股票基金"},
        {"code": "BOND_ENHANCED", "name": "增强债基", "asset_type": "fund", "fund_type": "增强指数型债券基金"},
        {"code": "ETF", "name": "股票ETF", "asset_type": "etf", "fund_type": "被动指数型股票基金",
         "is_stock_etf": True, "is_qdii": False},
    ])
    filtered = filter_universe(universe, strict_eligibility=False)
    assert set(filtered["code"]) == {"ACTIVE", "ETF"}


def test_scoring_uses_pit_aum_instead_of_current_provider_metadata(tmp_path):
    provider = get_provider("mock")
    store = PITDataStore(tmp_path / "pit")
    snapshot = provider.list_funds().assign(
        asset_type="fund", aum_yi=12.3, manager_start="2020-01-01", subscription_status="开放申购",
        management_company="易方达基金管理有限公司",
    )
    store.write_universe_snapshot(snapshot, source="mock", available_at="2024-09-30")
    scored = score_universe(
        provider, start="2019-01-01", end="2024-09-30", group_by_type=False,
        pit_store=store, universe_asof="2024-09-30",
    )
    assert set(scored["size_yi"]) == {12.3}


def test_stock_etf_pool_from_pit_requires_stock_label_non_qdii_and_size(tmp_path):
    store = PITDataStore(tmp_path / "pit")
    snapshot = pd.DataFrame([
        {"code": "159001", "name": "股票ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": True, "is_qdii": False, "aum_yi": 5.0},
        {"code": "159002", "name": "小ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": True, "is_qdii": False, "aum_yi": 4.99},
        {"code": "159003", "name": "QDII ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": True, "is_qdii": True, "aum_yi": 20.0},
        {"code": "159004", "name": "非股票ETF", "asset_type": "etf", "fund_type": "ETF",
         "is_stock_etf": False, "is_qdii": False, "aum_yi": 20.0},
    ])
    store.write_universe_snapshot(snapshot, source="vendor", available_at="2026-06-25")

    pool = _stock_etf_pool_from_pit(store, "2026-06-25")

    assert pool["code"].tolist() == ["159001"]
