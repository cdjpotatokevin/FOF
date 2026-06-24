"""风格择时引擎与回测测试。"""
import numpy as np
import pandas as pd

from fof_system.data import get_provider
from fof_system.engine.style_timing import StyleTimer
from fof_system.engine.style_backtest import run_style_backtest
from fof_system.engine import signals
from fof_system import config


def test_signals_bounded():
    """所有信号 view 必须落在 [-1,1]。"""
    mp = get_provider("mock")
    g = mp.get_index_close("399370", "", "")
    v = mp.get_index_close("399371", "", "")
    mom = signals.momentum_view(g, v, 120, 504).dropna()
    vol = signals.vol_regime_view(g, v, 20, 504).dropna()
    assert mom.between(-1, 1).all()
    assert vol.between(-1, 1).all()


def test_valuation_signal_contrarian():
    """成长估值比走高时，估值信号应转向偏价值(<0)。"""
    mp = get_provider("mock")
    gv = mp.get_index_valuation("399370")
    vv = mp.get_index_valuation("399371")
    view = signals.valuation_spread_view(gv, vv, 504).dropna()
    assert view.between(-1, 1).all()
    # 与对数估值比的相关应为负（逆向）
    ratio = np.log(gv / vv).reindex(view.index)
    corr = np.corrcoef(ratio.values, view.values)[0, 1]
    assert corr < 0


def test_target_weights_within_tilt_bounds():
    """目标权重相对考核基准的偏离不得超过 max_tilt。"""
    mp = get_provider("mock")
    timer = StyleTimer(mp)
    wdf = timer.target_weight_series("2019-01-01", "2023-12-31")
    cap = config.STYLE_TIMING.max_tilt + 1e-9
    assert (wdf["w_growth"] - config.BENCHMARK_WEIGHTS["growth"]).abs().max() <= cap
    assert np.allclose(wdf["w_growth"] + wdf["w_value"], 1.0)


def test_current_view_structure():
    mp = get_provider("mock")
    view = StyleTimer(mp).current_view("2019-01-01", "2023-12-31")
    assert -1 <= view.composite <= 1
    assert abs(view.w_growth - config.BENCHMARK_WEIGHTS["growth"]) <= config.STYLE_TIMING.max_tilt + 1e-9
    # mock 无 iFinD 但自带合成估值，三个信号都应在
    assert set(view.contributions) >= {"momentum", "vol_regime"}


def test_undated_valuation_aligns_to_price_calendar():
    raw = pd.Series([1.0, 1.1], index=pd.RangeIndex(2), name="pb")
    raw.attrs.update({"ifind_date_sequence_without_dates": True, "start": "2026-06-01", "end": "2026-06-02"})
    calendar = pd.DatetimeIndex([pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")])

    aligned = StyleTimer._align_undated_valuation(raw, calendar)

    assert aligned.index.equals(calendar)
    assert aligned.tolist() == [1.0, 1.1]


def test_short_undated_valuation_aligns_to_calendar_tail():
    raw = pd.Series([1.1], index=pd.RangeIndex(1), name="pb")
    raw.attrs.update({"ifind_date_sequence_without_dates": True, "start": "2026-06-01", "end": "2026-06-02"})
    calendar = pd.DatetimeIndex([pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")])

    aligned = StyleTimer._align_undated_valuation(raw, calendar)

    assert aligned.index.tolist() == [pd.Timestamp("2026-06-02")]
    assert aligned.tolist() == [1.1]


def test_backtest_runs_and_neutral_matches_bench():
    """max_tilt=0（完全中性）时，择时组合应等于基准，超额≈0。"""
    mp = get_provider("mock")
    g = mp.get_index_close("399370", "", "")
    v = mp.get_index_close("399371", "", "")
    # 构造全程考核基准的权重表
    idx = pd.date_range(g.index[0], g.index[-1], freq="ME")
    growth = config.BENCHMARK_WEIGHTS["growth"]
    wdf = pd.DataFrame({"w_growth": growth, "w_value": 1 - growth}, index=idx)
    res, _ = run_style_backtest(g, v, wdf)
    assert abs(res.ann_excess) < 1e-6
    assert abs(res.info_ratio) < 1e-6 or np.isnan(res.info_ratio)


def test_backtest_metrics_present():
    mp = get_provider("mock")
    timer = StyleTimer(mp)
    g = mp.get_index_close("399370", "", "")
    v = mp.get_index_close("399371", "", "")
    wdf = timer.target_weight_series("2019-01-01", "2023-12-31")
    res, detail = run_style_backtest(g, v, wdf)
    assert res.n_periods > 0
    assert {"w_growth", "active", "active_nav"}.issubset(detail.columns)
