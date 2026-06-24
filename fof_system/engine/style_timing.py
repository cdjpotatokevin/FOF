"""风格择时引擎：把信号合成为"成长/价值目标权重"时间序列与当期建议。

输出的目标权重相对考核基准做有界主动偏离：
    w_growth = benchmark_growth + max_tilt · composite_view
    w_value  = 1 - w_growth
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

from .. import config
from ..data.base import DataProvider
from . import signals

log = logging.getLogger("fof.style")


@dataclass
class StyleView:
    asof: pd.Timestamp
    composite: float                  # 合成观点 [-1,1]
    w_growth: float
    w_value: float
    contributions: dict[str, float]   # 各信号当期 view
    active_growth: float              # 相对考核基准的成长净偏离


class StyleTimer:
    def __init__(self, provider: DataProvider, cfg: config.StyleTimingConfig = config.STYLE_TIMING,
                 valuation_provider: DataProvider | None = None):
        """provider 取价格；valuation_provider 取指数估值（默认与 provider 同源）。

        实务常见组合：价格用 akshare（快、全），估值用 iFinD（akshare 拿不到）。
        """
        self.p = provider
        self.vp = valuation_provider or provider
        self.cfg = cfg

    # -- 取数 --------------------------------------------------------------
    def _load_prices(self, start: str, end: str) -> tuple[pd.Series, pd.Series]:
        g = self.p.get_index_close(config.STYLE_FACTORS["growth"], start, end)
        v = self.p.get_index_close(config.STYLE_FACTORS["value"], start, end)
        return g.sort_index(), v.sort_index()

    @staticmethod
    def _align_undated_valuation(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
        """将iFinD无日期的交易日序列映射到价格主交易日历。"""
        if not series.attrs.get("ifind_date_sequence_without_dates"):
            return series
        start, end = series.attrs.get("start"), series.attrs.get("end")
        dates = calendar
        if start:
            dates = dates[dates >= pd.Timestamp(start)]
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        if len(series) > len(dates):
            raise ValueError(
                f"iFinD无日期估值长度 {len(series)} 超过价格交易日历长度 {len(dates)}，拒绝猜测日期。"
            )
        if len(series) < len(dates):
            # iFinD会在长历史边界处只返回请求区间尾部的可用观测；原始响应不带日期，
            # 因此只接受这种“开头截断”情形，并把数值映射到交易日历尾部。
            log.warning(
                "iFinD估值仅返回请求区间尾部 %d/%d 个交易日观测，早期不可得部分不参与信号。",
                len(series), len(dates),
            )
            dates = dates[-len(series):]
        return pd.Series(series.to_numpy(), index=dates, name=series.name)

    def _load_valuation(self, start: str, end: str,
                        calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series] | None:
        try:
            gv = self.vp.get_index_valuation(config.STYLE_FACTORS["growth"], self.cfg.valuation_metric, start, end)
            vv = self.vp.get_index_valuation(config.STYLE_FACTORS["value"], self.cfg.valuation_metric, start, end)
            return self._align_undated_valuation(gv, calendar).sort_index(), self._align_undated_valuation(vv, calendar).sort_index()
        except NotImplementedError:
            log.info("数据源无指数估值，跳过估值价差信号。")
            return None
        except Exception as e:  # noqa: BLE001
            # 当前THS_DS下PE可能返回空值；在明确记录日志的前提下用PB继续构造同一
            # 相对估值信号，避免把本可用的估值维度整体丢弃。
            if self.cfg.valuation_metric != "pe_ttm":
                log.warning("估值取数失败，跳过估值信号：%s", e)
                return None
            try:
                gv = self.vp.get_index_valuation(config.STYLE_FACTORS["growth"], "pb", start, end)
                vv = self.vp.get_index_valuation(config.STYLE_FACTORS["value"], "pb", start, end)
                log.warning("PE估值不可用，已回退使用PB估值信号：%s", e)
                return self._align_undated_valuation(gv, calendar).sort_index(), self._align_undated_valuation(vv, calendar).sort_index()
            except Exception as fallback_error:  # noqa: BLE001
                log.warning("估值取数失败，跳过估值信号：PE=%s；PB=%s", e, fallback_error)
                return None

    # -- 信号 → 合成观点（daily）-----------------------------------------
    def build_views(self, start: str = "2015-01-01", end: str = "") -> pd.DataFrame:
        """返回 daily DataFrame：各信号 view + composite + w_growth/w_value。"""
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        g, v = self._load_prices(start, end)
        # 主交易日历 = 价格网格（成长∩价值）。所有信号对齐到这条网格，
        # 否则估值(iFinD)与价格(akshare)日历不同会产生"只有单信号"的错位行。
        master_idx = g.index.intersection(v.index)

        views: dict[str, pd.Series] = {}
        w = dict(self.cfg.signal_weights)

        views["momentum"] = signals.momentum_view(g, v, self.cfg.momentum_lookback, self.cfg.zscore_window)
        views["vol_regime"] = signals.vol_regime_view(
            g, v, self.cfg.vol_window, self.cfg.zscore_window,
            growth_weight=config.BENCHMARK_WEIGHTS["growth"],
        )

        val = self._load_valuation(start, end, master_idx)
        if val is not None:
            # 估值在自己的日历上算信号，再对齐到价格网格（ffill 补非交易日差异）
            vsig = signals.valuation_spread_view(val[0], val[1], self.cfg.zscore_window)
            views["valuation"] = vsig.reindex(master_idx.union(vsig.index)).ffill().reindex(master_idx)
        else:
            w.pop("valuation", None)   # 无估值数据则从权重里剔除

        # 统一对齐到主交易日历
        vdf = pd.DataFrame({k: s.reindex(master_idx) for k, s in views.items()}).sort_index()

        # 在"可用且非缺失"的信号上按权重归一合成
        wser = pd.Series({k: w[k] for k in vdf.columns if k in w}, dtype=float)
        comp = self._weighted_combine(vdf, wser)
        vdf["composite"] = comp.clip(-1, 1)
        benchmark_growth = config.BENCHMARK_WEIGHTS["growth"]
        vdf["w_growth"] = (benchmark_growth + self.cfg.max_tilt * vdf["composite"]).clip(0, 1)
        vdf["w_value"] = 1 - vdf["w_growth"]
        return vdf

    @staticmethod
    def _weighted_combine(vdf: pd.DataFrame, wser: pd.Series) -> pd.Series:
        """逐期按"当期有值的信号"重新归一加权，避免缺失信号把观点拉向 0。"""
        sub = vdf[wser.index]
        mask = sub.notna()
        wmat = mask.mul(wser, axis=1)
        denom = wmat.sum(axis=1).replace(0, np.nan)
        comp = (sub.fillna(0) * wmat).sum(axis=1) / denom
        return comp.fillna(0.0)

    # -- 当期建议 ----------------------------------------------------------
    def current_view(self, start: str = "2015-01-01", end: str = "") -> StyleView:
        vdf = self.build_views(start, end).dropna(subset=["composite"])
        row = vdf.iloc[-1]
        contrib = {k: float(row[k]) for k in self.cfg.signal_weights if k in vdf.columns and pd.notna(row.get(k))}
        return StyleView(
            asof=vdf.index[-1],
            composite=float(row["composite"]),
            w_growth=float(row["w_growth"]),
            w_value=float(row["w_value"]),
            contributions=contrib,
            active_growth=float(row["w_growth"] - config.BENCHMARK_WEIGHTS["growth"]),
        )

    # -- 调仓权重序列（供回测）-------------------------------------------
    def target_weight_series(self, start: str = "2015-01-01", end: str = "") -> pd.DataFrame:
        """按调仓频率采样目标权重（月末/周末），权重在下一期生效。"""
        vdf = self.build_views(start, end)
        rule = {"M": "ME", "W": "W-FRI"}.get(self.cfg.rebalance, "ME")
        sampled = vdf[["w_growth", "w_value", "composite"]].resample(rule).last().dropna()
        return sampled
