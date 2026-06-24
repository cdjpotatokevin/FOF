"""第⑤层归因测试：恒等式、符号、与端到端链路。"""
import numpy as np
import pandas as pd

from fof_system.data import get_provider
from fof_system.engine.attribution import style_selection_attribution, synth_etf_returns


def _make_style_returns(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "growth": rng.normal(0.001, 0.02, n),
        "value": rng.normal(0.0005, 0.015, n),
    }, index=idx)


def test_period_identity():
    """每期 excess 必须恒等于 风格择时 + 选股。"""
    sr = _make_style_returns()
    rng = np.random.default_rng(2)
    # 两只基金：一只偏成长有alpha，一只偏价值
    g_load = pd.Series({"F1": 0.8, "F2": 0.2})
    v_load = 1 - g_load
    R = pd.DataFrame({
        "F1": 0.8 * sr["growth"] + 0.2 * sr["value"] + 0.0005 + rng.normal(0, 0.003, len(sr)),
        "F2": 0.2 * sr["growth"] + 0.8 * sr["value"] - 0.0002 + rng.normal(0, 0.003, len(sr)),
    }, index=sr.index)
    w = pd.Series({"F1": 0.6, "F2": 0.4})
    attr = style_selection_attribution(w, R, g_load, v_load, sr, ppy=252)
    pr = attr.periods
    recon = pr["style_timing"] + pr["selection"]
    assert np.allclose(recon.values, pr["excess"].values, atol=1e-12)


def test_pure_style_no_selection():
    """持有纯风格ETF（无alpha），选股贡献应≈0，超额全来自风格择时。"""
    sr = _make_style_returns(seed=5)
    etf = synth_etf_returns(sr, {"ETF_G": 1.0, "ETF_V": 0.0})
    g_load = pd.Series({"ETF_G": 1.0, "ETF_V": 0.0})
    v_load = 1 - g_load
    w = pd.Series({"ETF_G": 0.7, "ETF_V": 0.3})  # 偏成长 70/30
    attr = style_selection_attribution(w, etf, g_load, v_load, sr, ppy=252)
    assert abs(attr.total_selection) < 1e-9          # 无选股
    assert abs(attr.avg_growth_exposure - 0.7) < 1e-9
    # 超额≈风格择时
    assert abs(attr.cum_excess - attr.total_style_timing - attr.linking_residual) < 1e-9


def test_selection_sign():
    """跑赢自身风格的基金，其选股贡献应为正。"""
    sr = _make_style_returns(seed=7)
    g_load = pd.Series({"WIN": 0.5})
    v_load = 1 - g_load
    # WIN 每期在风格复制上额外 +20bp
    R = pd.DataFrame({"WIN": 0.5 * sr["growth"] + 0.5 * sr["value"] + 0.002}, index=sr.index)
    w = pd.Series({"WIN": 1.0})
    attr = style_selection_attribution(w, R, g_load, v_load, sr, ppy=252)
    assert attr.total_selection > 0
    assert attr.fund_selection["WIN"] > 0


def test_end_to_end_mock_attribution():
    from fof_system.portfolio import build_portfolio
    from fof_system.pipeline import build_benchmark
    from fof_system import config
    mp = get_provider("mock")
    codes = list(mp.truth.keys())
    res, detail = build_portfolio(mp, codes, target_growth=0.55,
                                  start="2019-01-01", end="2023-12-31")
    factor_df, _ = build_benchmark(mp, "2019-01-01", "2023-12-31")
    weights = detail.set_index("code")["weight"]
    gl = detail.set_index("code")["growth_load"]
    vl = 1 - gl
    fund_codes = detail.loc[detail["type"] == "主动基金", "code"].tolist()
    etf_codes = detail.loc[detail["type"] == "ETF补全", "code"].tolist()
    cols = {c: mp.to_returns(mp.get_fund_nav(c, "2019-01-01", "2023-12-31"), "W") for c in fund_codes}
    rdf = pd.DataFrame(cols)
    if etf_codes:
        rdf = pd.concat([rdf, synth_etf_returns(factor_df, {e: float(gl[e]) for e in etf_codes})], axis=1)
    attr = style_selection_attribution(weights, rdf, gl, vl, factor_df, ppy=52)
    recon = attr.periods["style_timing"] + attr.periods["selection"]
    assert np.allclose(recon.values, attr.periods["excess"].values, atol=1e-10)
    assert attr.diagnostics["n_periods"] > 50
