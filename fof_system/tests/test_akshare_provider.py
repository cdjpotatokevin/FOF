"""AkShare适配器的离线字段合同测试，不依赖网络。"""
import pandas as pd

from fof_system.data.akshare_provider import AkshareProvider


class _FakeAk:
    def fund_open_fund_info_em(self, **kwargs):
        raise RuntimeError("not an open-end fund")

    def fund_etf_hist_em(self, **kwargs):
        return pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"], "收盘": [3.50, 3.55],
            "成交量": [100, 120], "成交额": [350, 426], "换手率": [1.2, 1.3],
        })

    def fund_etf_spot_em(self):
        return pd.DataFrame({"代码": ["510300"], "名称": ["沪深300ETF"]})


def _provider() -> AkshareProvider:
    # 绕过 __init__ 的真实 akshare import，验证适配逻辑本身。
    provider = object.__new__(AkshareProvider)
    provider.ak = _FakeAk()
    provider.retry = 1
    provider.sleep = 0
    provider._fund_list_cache = None
    provider.cache_dir = None
    return provider


def test_etf_market_normalizes_required_columns_and_open_fund_falls_back():
    provider = _provider()
    market = provider.get_etf_market("510300", "2024-01-01", "2024-01-31")
    assert {"trading_date", "code", "close", "volume", "amount", "source"}.issubset(market.columns)
    assert market["code"].eq("510300").all()
    close = provider.get_fund_nav("510300", "2024-01-01", "2024-01-31")
    assert close.iloc[-1] == 3.55


def test_etf_snapshot_has_asset_contract():
    etfs = _provider().list_etfs()
    assert etfs.loc[0, "code"] == "510300"
    assert etfs.loc[0, "asset_type"] == "etf"


def test_fund_nav_cache_reuses_standardized_series(tmp_path):
    provider = _provider()
    provider.cache_dir = tmp_path
    first = provider.get_fund_nav("510300", "2024-01-01", "2024-01-31")
    provider.ak.fund_etf_hist_em = lambda **_: (_ for _ in ()).throw(RuntimeError("should use cache"))

    second = provider.get_fund_nav("510300", "2024-01-01", "2024-01-31")

    assert first.equals(second)
    assert list((tmp_path / "fund_nav").glob("510300_*.csv"))
