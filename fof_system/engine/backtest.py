"""第⑥层：整链滚动回测（walk-forward）。

每个调仓日 t（月末），**只用 ≤t 的数据**完成：
  ② 对候选基金做 RBSA → 风格调整后 alpha / IR → 选 top-N
  ③ 取当期风格目标（StyleTimer 的 w_growth，信号本身因果）
  ④ 由主动收益协方差 + alpha×R²，在风格目标约束下优化出基金/ETF 权重
然后持有到下一个调仓日，用**之后**的真实收益结算，杜绝前视。

为效率：数据一次性预取，walk-forward 全程在内存里点时切片，复用各层引擎函数。
回测期内基金打分用"单次 RBSA"的轻量版（不做滚动一致性），权衡速度。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Mapping, Any
import numpy as np
import pandas as pd

from .. import config
from ..data.base import DataProvider
from ..data.pit import PITDataStore
from .rbsa import run_rbsa
from . import risk_model
from .optimizer import optimize_portfolio
from .attribution import synth_etf_returns

log = logging.getLogger("fof.backtest")


@dataclass
class BacktestOutput:
    nav: pd.DataFrame                 # 列：strat / bench（净值，起点1）
    weekly: pd.DataFrame              # 含毛收益、交易成本、净收益、基准与超额
    weights_history: pd.DataFrame     # 各调仓日权重（行=日期，列=资产）
    metrics: dict
    target_growth_history: pd.Series
    turnover_history: pd.Series
    cost_history: pd.Series
    rebalances: int = 0
    notes: list[str] = field(default_factory=list)


def _ann_return(ret: pd.Series, ppy: int) -> float:
    if len(ret) == 0:
        return float("nan")
    return float((1 + ret).prod() ** (ppy / len(ret)) - 1)


def _max_dd(ret: pd.Series) -> float:
    nav = (1 + ret).cumprod()
    return float(-((nav - nav.cummax()) / nav.cummax()).min()) if len(nav) else float("nan")


class ChainBacktester:
    def __init__(self, provider: DataProvider, codes: list[str],
                 start: str, end: str, val_provider: DataProvider | None = None,
                 opt_cfg: config.OptimizerConfig = config.OPTIMIZER,
                 style_cfg: config.StyleTimingConfig = config.STYLE_TIMING,
                 backtest_cfg: config.BacktestConfig = config.BACKTEST,
                 pit_store: PITDataStore | None = None,
                 asset_metadata: Mapping[str, Mapping[str, Any]] | None = None):
        self.p = provider
        self.vp = val_provider
        self.codes = codes
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.opt = opt_cfg
        self.scfg = style_cfg
        self.bcfg = backtest_cfg
        self.pit_store = pit_store
        self.asset_metadata = {str(code): dict(meta) for code, meta in (asset_metadata or {}).items()}
        self.freq = config.RETURN_FREQ
        self.ppy = config.PERIODS_PER_YEAR[self.freq]
        self._prepared = False

    # -- 一次性预取数据 ----------------------------------------------------
    def _prepare(self):
        from ..pipeline import build_benchmark
        from .style_timing import StyleTimer

        self.factor_df, self.bench_ret = build_benchmark(self.p, self.start, self.end)
        # 基金周收益矩阵
        cols = {}
        for c in self.codes:
            try:
                nav = self.p.get_fund_nav(c, self.start, self.end)
                cols[c] = self.p.to_returns(nav, self.freq)
            except Exception as e:  # noqa: BLE001
                log.warning("回测跳过 %s：%s", c, e)
        self.fund_ret = pd.DataFrame(cols)
        # 第③层风格目标（因果，预先整段算好，按调仓日 asof 取）
        timer = StyleTimer(self.p, self.scfg, valuation_provider=self.vp)
        self.style_views = timer.build_views(self.start, self.end)  # daily w_growth
        self._prepared = True

    # -- 单个调仓日：点时建组合 -------------------------------------------
    @staticmethod
    def _truthy(value: Any) -> bool:
        return str(value).strip().lower() in ("1", "true", "yes", "y", "是")

    def _eligible_records(self, t: pd.Timestamp) -> dict[str, dict]:
        """返回调仓日当时可投的候选元数据；无PIT时使用显式静态元数据。"""
        if self.pit_store is None:
            return {str(code): self.asset_metadata.get(str(code), {}) for code in self.fund_ret.columns}
        universe = self.pit_store.read_universe_asof(t, asset_types=("fund", "etf"))
        if universe.empty:
            return {}
        if "is_qdii" in universe:
            qdii = universe["is_qdii"].astype(str).str.lower().isin(("1", "true", "yes"))
            universe = universe[~qdii]
        from .universe import filter_universe
        universe = filter_universe(universe, asof=t, strict_eligibility=True)
        records = {str(row["code"]): row.to_dict() for _, row in universe.iterrows()}
        return {str(code): records[str(code)] for code in self.fund_ret.columns if str(code) in records}

    def _eligible_codes(self, t: pd.Timestamp) -> list[str]:
        """兼容旧调用方的候选代码视图。"""
        return list(self._eligible_records(t))

    def _rebalance(self, t: pd.Timestamp):
        rf = config.RISK_FREE_ANNUAL / self.ppy
        fac_hist = self.factor_df.loc[:t]

        rows = []
        active_map = {}
        eligible = self._eligible_records(t)
        for c, meta in eligible.items():
            fr = self.fund_ret[c].loc[:t].dropna()
            aligned = pd.concat([fr, fac_hist], axis=1, join="inner").dropna()
            if len(aligned) < config.MIN_OBS:
                continue
            aligned = aligned.iloc[-config.EVAL_WINDOW:]
            try:
                rb = run_rbsa(aligned.iloc[:, 0], aligned[fac_hist.columns], rf_per_period=rf)
            except Exception:  # noqa: BLE001
                continue
            active = rb.active_returns
            avol = active.std(ddof=1) * np.sqrt(self.ppy)
            alpha = rb.alpha_annual(self.ppy)
            ir = alpha / avol if avol > 0 else 0.0
            rows.append({"code": c, "alpha": alpha,
                         "alpha_r2": alpha * np.clip(rb.style_r2, 0.0, 1.0),
                         "style_r2": np.clip(rb.style_r2, 0.0, 1.0), "ir": ir,
                         "growth_load": rb.weights.get("growth", config.BENCHMARK_WEIGHTS["growth"]),
                         "asset_type": "etf" if (str(meta.get("asset_type", "")).lower() == "etf"
                                                    or self._truthy(meta.get("is_stock_etf"))) else "fund"})
            active_map[c] = active

        if len(rows) < 2:
            return None, None, None
        cand = pd.DataFrame(rows).set_index("code")

        # 轻量打分：alpha 与 IR 的组内 z 合成，选 top-N
        def _z(s):
            sd = s.std(ddof=0)
            return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0
        cand["score"] = 0.6 * _z(cand["alpha"]) + 0.4 * _z(cand["ir"])
        # 主动基金按alpha/IR选优；实际ETF作为已验证的风格工具进入优化器，但alpha固定为0。
        active_top = cand[cand["asset_type"] != "etf"].sort_values("score", ascending=False).head(self.opt.n_candidates)
        actual_etf = cand[cand["asset_type"] == "etf"]
        top = pd.concat([active_top, actual_etf]).loc[lambda x: ~x.index.duplicated(keep="first")]
        top_codes = top.index.tolist()

        # 第③层风格目标（asof t）
        wg = self.style_views["w_growth"]
        target_g = (float(wg.loc[:t].iloc[-1]) if len(wg.loc[:t])
                    else config.BENCHMARK_WEIGHTS["growth"])

        # 协方差 + alpha（风格调整后alpha × R²后再收缩）
        active_df = pd.DataFrame({c: active_map[c] for c in top_codes})
        cov = risk_model.active_cov(active_df, shrink=self.opt.cov_shrink, ppy=self.ppy)
        alpha = risk_model.shrink_alpha(top["alpha_r2"], self.opt.alpha_shrink)
        growth_load = top["growth_load"].copy()
        actual_etf_codes = top.index[top["asset_type"].eq("etf")].tolist()
        alpha.loc[actual_etf_codes] = 0.0

        etf_codes = list(actual_etf_codes)
        synthetic_etf_codes = []
        # 有PIT元数据的回测应只交易历史实际存在的ETF；合成ETF只保留给无PIT的mock验证。
        if self.opt.use_style_etf and self.pit_store is None and not self.asset_metadata:
            for ecode, meta in config.STYLE_ETF_ASSETS.items():
                synthetic_etf_codes.append(ecode)
                etf_codes.append(ecode)
                alpha[ecode] = 0.0
                growth_load[ecode] = meta["growth_load"]
            cov = risk_model.add_etf_assets(cov, synthetic_etf_codes)

        res = optimize_portfolio(
            alpha=alpha, cov=cov, growth_load=growth_load, target_growth=target_g,
            etf_codes=etf_codes, max_weight_fund=self.opt.max_weight_fund,
            min_weight_fund=self.opt.min_weight_fund, risk_aversion=self.opt.risk_aversion,
            etf_total_cap=self.opt.etf_total_cap, te_budget_annual=self.opt.te_budget_annual,
        )
        return res.weights, target_g, set(synthetic_etf_codes), set(etf_codes)

    # -- 主循环 ------------------------------------------------------------
    def _transaction_cost(self, weights: pd.Series, previous: pd.Series | None,
                          etf_codes: set[str]) -> float:
        """按目标权重变化估算单边交易成本。

        首次建仓按从零建到目标组合计费；基金和 ETF 使用不同成本假设。
        正式投资委员会材料应按产品费率、持有期和实际成交冲击成本覆写该简化模型。
        """
        old = previous if previous is not None else pd.Series(dtype=float)
        codes = weights.index.union(old.index)
        delta = weights.reindex(codes, fill_value=0.0) - old.reindex(codes, fill_value=0.0)
        is_etf = codes.isin(etf_codes)
        rates = np.where(is_etf, self.bcfg.etf_one_way_cost, self.bcfg.fund_one_way_cost)
        return float((delta.abs().to_numpy() * rates).sum())

    def run(self, warmup_obs: int | None = None) -> BacktestOutput:
        if not self._prepared:
            self._prepare()
        warmup = warmup_obs if warmup_obs is not None else self.bcfg.warmup_obs
        warmup = max(warmup, config.MIN_OBS)
        widx = self.factor_df.index
        if len(widx) <= warmup + 4:
            raise RuntimeError("样本太短，无法回测。")

        # 调仓日 = 每月最后一个周频日期，且已过 warmup
        month_end = pd.Series(widx, index=widx).resample("ME").last().dropna()
        rebal_dates = [d for d in month_end.values if d >= widx[warmup]]
        rebal_dates = pd.to_datetime(rebal_dates)
        if len(rebal_dates) < 2:
            raise RuntimeError("可调仓次数不足。")

        # ETF 合成收益（整段）
        etf_ret_all = synth_etf_returns(
            self.factor_df, {e: config.STYLE_ETF_ASSETS[e]["growth_load"]
                             for e in config.STYLE_ETF_ASSETS})

        weekly_rows = []
        w_hist, tg_hist, turn_hist, cost_hist = {}, {}, {}, {}
        prev_w = None
        prev_etf_codes: set[str] = set()
        notes = [
            f"单边交易成本：主动基金 {self.bcfg.fund_one_way_cost:.2%}，ETF {self.bcfg.etf_one_way_cost:.2%}",
            "首次建仓与每次调仓均已从策略收益中扣除估算交易成本。",
        ]
        if self.pit_store is None:
            notes.append("未提供PIT数据仓：候选池为静态输入，可能含幸存者偏差。")
        else:
            notes.append("候选池按每个调仓日的PIT主数据快照筛选。")

        for i in range(len(rebal_dates) - 1):
            t0, t1 = rebal_dates[i], rebal_dates[i + 1]
            weights, target_g, synthetic_etf_set, all_etf_codes = self._rebalance(t0)
            if weights is None:
                continue
            w_hist[t0] = weights
            tg_hist[t0] = target_g
            old_w = prev_w
            # 换手（单边）
            if old_w is not None:
                allc = weights.index.union(old_w.index)
                turn_hist[t0] = float((weights.reindex(allc, fill_value=0)
                                       - old_w.reindex(allc, fill_value=0)).abs().sum() / 2)
            trade_cost = self._transaction_cost(weights, old_w, all_etf_codes | prev_etf_codes)
            cost_hist[t0] = trade_cost
            prev_w = weights
            prev_etf_codes = all_etf_codes

            # 持有 (t0, t1]：用之后的周收益结算（防前视）
            sub = self.factor_df.loc[(self.factor_df.index > t0) & (self.factor_df.index <= t1)].index
            for j, d in enumerate(sub):
                gross_r = 0.0
                for code, wt in weights.items():
                    if code in synthetic_etf_set:
                        gross_r += wt * etf_ret_all.loc[d, code]
                    elif code in self.fund_ret.columns and d in self.fund_ret.index:
                        rv = self.fund_ret.loc[d, code]
                        gross_r += wt * (rv if pd.notna(rv) else 0.0)
                cost = trade_cost if j == 0 else 0.0
                weekly_rows.append({
                    "date": d, "gross_port_ret": gross_r,
                    "transaction_cost": cost, "port_ret": gross_r - cost,
                })

        if not weekly_rows:
            raise RuntimeError("回测无任何持有期收益。")

        weekly = pd.DataFrame(weekly_rows).set_index("date").sort_index()
        gross_pser = weekly["gross_port_ret"]
        pser = weekly["port_ret"]
        bser = self.bench_ret.reindex(pser.index).fillna(0.0)
        active = pser - bser

        nav = pd.DataFrame({
            "strat": (1 + pser).cumprod(),
            "bench": (1 + bser).cumprod(),
        })
        weekly["bench_ret"] = bser
        weekly["active"] = active

        te = active.std(ddof=1) * np.sqrt(self.ppy)
        ann_excess = active.mean() * self.ppy
        metrics = {
            "ann_return_strat": _ann_return(pser, self.ppy),
            "ann_return_gross": _ann_return(gross_pser, self.ppy),
            "ann_return_bench": _ann_return(bser, self.ppy),
            "ann_excess": float(ann_excess),
            "ann_excess_gross": float((gross_pser - bser).mean() * self.ppy),
            "tracking_error": float(te),
            "info_ratio": float(ann_excess / te) if te > 0 else float("nan"),
            "max_dd_strat": _max_dd(pser),
            "max_dd_bench": _max_dd(bser),
            "active_max_dd": _max_dd(active),
            "monthly_hit": float(((1 + active).resample("ME").prod() - 1 > 0).mean()),
            "avg_turnover": float(np.mean(list(turn_hist.values()))) if turn_hist else 0.0,
            "annual_turnover": float(sum(turn_hist.values()) / max(len(pser) / self.ppy, 1e-9)),
            "total_transaction_cost": float(weekly["transaction_cost"].sum()),
            "n_weeks": len(pser),
        }

        return BacktestOutput(
            nav=nav, weekly=weekly,
            weights_history=pd.DataFrame(w_hist).T.fillna(0.0),
            metrics=metrics,
            target_growth_history=pd.Series(tg_hist),
            turnover_history=pd.Series(turn_hist),
            cost_history=pd.Series(cost_hist),
            rebalances=len(w_hist), notes=notes,
        )
