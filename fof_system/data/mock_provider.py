"""合成数据源：离线、确定性，用于单元测试和无网络环境下验证整条链路。

生成机制（已知真值，便于校验 RBSA）：
- 两个风格指数（成长/价值）各为带漂移的几何随机游走。
- 每只基金 = 设定的真实风格载荷 w 对两指数的组合 + 设定的真实 alpha + 特质噪声。
这样 RBSA 还原出的载荷应接近设定值，alpha 应接近设定值。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import DataProvider, FundMeta


class MockProvider(DataProvider):
    def __init__(self, seed: int = 7, n_days: int = 1500):
        self.rng = np.random.default_rng(seed)
        self.dates = pd.bdate_range("2019-01-01", periods=n_days)
        # 风格指数日收益
        g = self.rng.normal(0.0006, 0.014, n_days)   # 成长：高收益高波动
        v = self.rng.normal(0.0004, 0.010, n_days)   # 价值：低收益低波动
        self._idx_ret = {"399370": pd.Series(g, index=self.dates),
                         "399371": pd.Series(v, index=self.dates)}
        self._idx_close = {c: (1 + r).cumprod() * 1000 for c, r in self._idx_ret.items()}

        # 预设若干基金的“真实”风格载荷与年化 alpha
        # noise 为日频特质波动；过大会淹没 alpha（RBSA 小样本固有现象），
        # 这里设成较小值以便 demo 清晰还原已知真值。
        self._specs = {
            "F_GROWTH_STAR": dict(w_g=0.85, w_v=0.15, alpha=0.06, noise=0.0025, name="成长之星", typ="股票型"),
            "F_VALUE_STAR":  dict(w_g=0.15, w_v=0.85, alpha=0.05, noise=0.0025, name="价值标杆", typ="偏股混合型"),
            "F_BALANCED_OK": dict(w_g=0.50, w_v=0.50, alpha=0.03, noise=0.0025, name="均衡稳健", typ="偏股混合型"),
            "F_GROWTH_LAG":  dict(w_g=0.80, w_v=0.20, alpha=-0.02, noise=0.0035, name="成长落后", typ="股票型"),
            "F_CLOSET_IDX":  dict(w_g=0.50, w_v=0.50, alpha=0.00, noise=0.0010, name="影子指数", typ="指数增强"),
            # 10% 单基金上限下，离线整链验证也需要足够多的可投主动基金才能满仓。
            "F_GROWTH_QUALITY": dict(w_g=0.70, w_v=0.30, alpha=0.04, noise=0.0028, name="成长质量", typ="股票型"),
            "F_VALUE_DEFENSIVE": dict(w_g=0.25, w_v=0.75, alpha=0.02, noise=0.0024, name="价值防御", typ="偏股混合型"),
            "F_BALANCED_ALPHA": dict(w_g=0.60, w_v=0.40, alpha=0.025, noise=0.0026, name="均衡优选", typ="偏股混合型"),
            "F_VALUE_MOMENTUM": dict(w_g=0.35, w_v=0.65, alpha=0.015, noise=0.0027, name="价值动量", typ="股票型"),
            "F_STYLE_DIVERSIFIER": dict(w_g=0.45, w_v=0.55, alpha=0.01, noise=0.0025, name="风格分散", typ="偏股混合型"),
        }
        self._fund_ret = {}
        rf_daily = 0.018 / 252
        for code, sp in self._specs.items():
            base = sp["w_g"] * g + sp["w_v"] * v
            alpha_daily = sp["alpha"] / 252
            noise = self.rng.normal(0, sp["noise"], n_days)
            self._fund_ret[code] = pd.Series(base + alpha_daily + noise, index=self.dates)
        self._rf_daily = rf_daily

    @staticmethod
    def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
        if start:
            s = s.loc[start:]
        if end:
            s = s.loc[:end]
        return s

    def get_index_close(self, code: str, start: str = "", end: str = "") -> pd.Series:
        return self._slice(self._idx_close[code], start, end)

    def get_index_valuation(self, code: str, metric: str = "pe_ttm",
                            start: str = "", end: str = "") -> pd.Series:
        """合成 PE：基线 + 价格相对 1 年均线的偏离（涨多了估值高），便于测试估值信号。"""
        close = self._idx_close[code]
        base = {"399370": 35.0, "399371": 12.0}.get(code, 20.0)  # 成长估值更高
        ma = close.rolling(252, min_periods=20).mean()
        pe = (base * (close / ma)).bfill()
        return self._slice(pe, start, end)

    def get_fund_nav(self, code: str, start: str = "", end: str = "") -> pd.Series:
        nav = (1 + self._fund_ret[code]).cumprod()
        return self._slice(nav, start, end)

    def list_funds(self) -> pd.DataFrame:
        rows = [dict(code=c, name=sp["name"], fund_type=sp["typ"], aum_yi=20.0,
                     inception="2019-01-01")
                for c, sp in self._specs.items()]
        return pd.DataFrame(rows)

    def get_fund_meta(self, code: str) -> FundMeta:
        sp = self._specs[code]
        return FundMeta(
            code=code, name=sp["name"], fund_type=sp["typ"],
            size_yi=20.0, inception=pd.Timestamp("2019-01-01"),
            manager="模拟经理", manager_start=pd.Timestamp("2020-01-01"),
        )

    # 暴露真值，供测试断言
    @property
    def truth(self) -> dict:
        return self._specs
