"""综合打分：把多维指标合成单一可排序的分数。

做法：
1. 在"同业组"内对每个指标做稳健 z-score（用中位数 + MAD，抗异常值）。
2. 按方向（越大越好/越小越好）对齐符号。
3. 加权求和得综合分；再映射到 0~100 便于阅读。

为什么组内标准化：不同指标量纲不同（alpha 是小数、回撤是小数、任期是年），
直接相加无意义；z-score 让每个指标贡献可比，权重才真正决定话语权。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import MetricSpec, SCORE_METRICS


def robust_zscore(s: pd.Series) -> pd.Series:
    """用中位数/MAD 的稳健 z-score；MAD 退化时回退到标准差。"""
    x = s.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad and np.isfinite(mad) and mad > 1e-12:
        z = (x - med) / (1.4826 * mad)
    else:
        std = x.std(ddof=0)
        z = (x - x.mean()) / std if std and std > 1e-12 else pd.Series(0.0, index=x.index)
    # 截断极端值，避免单指标主导
    return z.clip(-3, 3)


def score_funds(
    metrics_df: pd.DataFrame,
    specs: list[MetricSpec] = SCORE_METRICS,
    group_col: str | None = None,
) -> pd.DataFrame:
    """对一组基金打分。

    metrics_df : 每行一只基金，列含各 MetricSpec.key
    group_col  : 若给定（如 'fund_type'），在组内分别标准化（更公平的同业比较）
    返回：原表 + 各指标 z 列 + composite_score(0~100) + rank，按分数降序。
    """
    df = metrics_df.copy()
    specs = [s for s in specs if s.key in df.columns]
    total_w = sum(abs(s.weight) for s in specs) or 1.0

    def _score_block(block: pd.DataFrame) -> pd.DataFrame:
        contrib = pd.Series(0.0, index=block.index)
        for sp in specs:
            col = block[sp.key]
            # 缺失值以组内中位数填补，避免缺数据被极端惩罚/奖励
            col = col.fillna(col.median())
            z = robust_zscore(col) * sp.direction
            block[f"z_{sp.key}"] = z
            contrib = contrib + z * (sp.weight / total_w)
        block["composite_raw"] = contrib
        return block

    if group_col and group_col in df.columns:
        df = df.groupby(group_col, group_keys=False).apply(_score_block)
    else:
        df = _score_block(df)

    # 映射到 0~100：用全样本的稳健分位映射，均值约 50
    raw = df["composite_raw"]
    z_all = robust_zscore(raw)
    df["composite_score"] = (50 + 15 * z_all).clip(0, 100).round(1)
    df = df.sort_values("composite_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def explain_top(scored_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """给前 n 名做一个精简展示表（核心列）。"""
    cols = [
        "rank", "code", "name", "composite_score",
        "style_alpha_ann", "info_ratio", "excess_win_rate", "alpha_consistency",
        "growth_load", "value_load", "style_vs_bench",
        "max_drawdown", "calmar", "style_r2",
        "excess_vs_bench_ann", "te_vs_bench_ann",
        "size_yi", "tenure_years", "n_obs",
    ]
    cols = [c for c in cols if c in scored_df.columns]
    return scored_df[cols].head(n).reset_index(drop=True)
