"""rebalance_at 与生产路径一致性测试。"""
from pathlib import Path

import pandas as pd

from fof_system import config
from fof_system.data import get_provider
from fof_system.data.pit import PITDataStore
from fof_system.portfolio import build_portfolio, resolve_target_growth, select_full_universe_candidates
from fof_system.pipeline import score_universe
from fof_system.rebalance import rebalance_at, score_cache_path


def test_rebalance_at_matches_manual_portfolio_build(tmp_path):
    provider = get_provider("mock")
    codes = list(provider.truth.keys())  # type: ignore[attr-defined]
    asof = "2024-06-28"
    target_g = resolve_target_growth(provider, None, "2019-01-01", asof)
    scored = score_universe(provider, codes=codes, start="2019-01-01", end=asof)
    pick = select_full_universe_candidates(scored, n_active=config.OPTIMIZER.n_candidates, target_growth=target_g)
    manual, _ = build_portfolio(provider, pick, target_g, "2019-01-01", asof)

    aligned = rebalance_at(
        provider, asof=asof, eval_start="2019-01-01", pit_store=None,
        target_growth=target_g, strict_eligibility=False,
    )
    assert aligned.success
    assert manual.success
    merged = pd.concat([manual.weights.rename("manual"), aligned.weights.rename("aligned")], axis=1).fillna(0.0)
    assert (merged["manual"] - merged["aligned"]).abs().max() < 1e-6


def test_score_cache_roundtrip(tmp_path):
    provider = get_provider("mock")
    cache_dir = tmp_path / "scores"
    asof = "2024-03-29"
    first = rebalance_at(
        provider, asof=asof, eval_start="2019-01-01", pit_store=None,
        score_cache_dir=cache_dir, strict_eligibility=False,
    )
    assert first.success
    cache_file = score_cache_path(cache_dir, asof)
    assert cache_file is not None and cache_file.exists()
    second = rebalance_at(
        provider, asof=asof, eval_start="2019-01-01", pit_store=None,
        score_cache_dir=cache_dir, strict_eligibility=False,
    )
    assert second.success
    assert (first.weights - second.weights).abs().max() < 1e-9


def test_active_capacity_caps_reduce_heavy_weight(tmp_path):
    store = PITDataStore(tmp_path / "pit")
    universe = pd.DataFrame([
        {"code": "000001", "name": "小基金", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 5.0, "inception": "2018-01-01", "subscription_status": "开放申购"},
        {"code": "000002", "name": "大基金", "asset_type": "fund", "fund_type": "股票型",
         "aum_yi": 100.0, "inception": "2018-01-01", "subscription_status": "开放申购"},
    ])
    store.write_universe_snapshot(universe, source="test", available_at="2024-06-28")

    from fof_system.rebalance import compute_capacity_caps
    caps = compute_capacity_caps(
        ["000001", "000002"],
        {row["code"]: row for row in universe.to_dict("records")},
        pit_store=store,
        capacity_asof="2024-06-28",
        portfolio_aum_yi=14.0,
        max_order_to_fund_aum=0.20,
        etf_participation_rate=0.10,
        etf_adv_lookback=20,
        enforce_active_fund_capacity=True,
        enforce_etf_capacity=False,
    )
    assert caps["000001"] < config.OPTIMIZER.max_weight_fund
