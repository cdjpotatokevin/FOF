"""指标与综合打分测试。"""
import numpy as np
import pandas as pd

from fof_system.data import get_provider
from fof_system.pipeline import score_universe
from fof_system.engine.metrics import max_drawdown_from_returns, _size_score
from fof_system.engine.scoring import robust_zscore


def test_max_drawdown_basic():
    # 连续 -10% 两期再回升，回撤应约 19%
    r = pd.Series([0.0, -0.1, -0.1, 0.05])
    dd = max_drawdown_from_returns(r)
    assert 0.18 < dd < 0.20


def test_size_score_sweet_spot():
    assert _size_score(10, (2, 80)) == 1.0
    assert _size_score(1, (2, 80)) < 1.0       # 太小衰减
    assert _size_score(160, (2, 80)) < 0.6     # 太大衰减
    assert np.isnan(_size_score(float("nan"), (2, 80)))


def test_robust_zscore_constant_series():
    z = robust_zscore(pd.Series([3.0, 3.0, 3.0]))
    assert (z == 0).all()


def test_end_to_end_mock_ranking():
    """整条链路跑通，且优质基金排名应高于落后基金。"""
    mp = get_provider("mock")
    scored = score_universe(mp, codes=list(mp.truth.keys()),
                            start="2019-01-01", end="2023-12-31",
                            group_by_type=False)
    assert {"composite_score", "rank", "style_alpha_ann"}.issubset(scored.columns)
    rank = scored.set_index("code")["rank"]
    # 成长之星(高alpha) 应排在 成长落后(负alpha) 之前
    assert rank["F_GROWTH_STAR"] < rank["F_GROWTH_LAG"]
    # 还原的风格偏离方向正确
    row = scored.set_index("code").loc["F_VALUE_STAR"]
    assert row["value_load"] > row["growth_load"]
