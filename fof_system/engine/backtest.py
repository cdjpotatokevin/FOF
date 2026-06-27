"""第⑥层：整链滚动回测（walk-forward）。

每个调仓日 t（默认季度末），**只用 ≤t 的数据**调用 ``rebalance_at``：
  ② 全池评分（按 PIT 快照键缓存）→ 候选筛选 → ③ 风格目标 → ④ 优化权重
基金净值按需加载，不预取全区间 PIT 并集。
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
from ..rebalance import collect_pit_codes, rebalance_at, pit_snapshot_label
from .attribution import synth_etf_returns

log = logging.getLogger("fof.backtest")


@dataclass
class BacktestOutput:
    nav: pd.DataFrame
    weekly: pd.DataFrame
    weights_history: pd.DataFrame
    metrics: dict
    target_growth_history: pd.Series
    turnover_history: pd.Series
    cost_history: pd.Series
    rebalances: int = 0
    notes: list[str] = field(default_factory=list)
    rebalance_log: pd.DataFrame = field(default_factory=pd.DataFrame)


def _ann_return(ret: pd.Series, ppy: int) -> float:
    if len(ret) == 0:
        return float("nan")
    return float((1 + ret).prod() ** (ppy / len(ret)) - 1)


def _max_dd(ret: pd.Series) -> float:
    nav = (1 + ret).cumprod()
    return float(-((nav - nav.cummax()) / nav.cummax()).min()) if len(nav) else float("nan")


class ChainBacktester:
    def __init__(self, provider: DataProvider, codes: list[str] | None,
                 start: str, end: str, val_provider: DataProvider | None = None,
                 opt_cfg: config.OptimizerConfig = config.OPTIMIZER,
                 style_cfg: config.StyleTimingConfig = config.STYLE_TIMING,
                 backtest_cfg: config.BacktestConfig = config.BACKTEST,
                 pit_store: PITDataStore | None = None,
                 asset_metadata: Mapping[str, Mapping[str, Any]] | None = None):
        self.p = provider
        self.vp = val_provider
        self.codes = codes or []
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
        self._score_mem_cache: dict[str, pd.DataFrame] = {}
        self._provider_kwargs: dict = {}
        if getattr(self.p, "cache_dir", None):
            self._provider_kwargs["cache_dir"] = str(self.p.cache_dir)

    def _prepare(self):
        from ..pipeline import build_benchmark

        self.factor_df, self.bench_ret = build_benchmark(self.p, self.start, self.end)
        self.fund_ret = pd.DataFrame()

        if self.bcfg.prefetch_all_pit_codes and self.bcfg.full_universe_scoring and self.pit_store is not None:
            prefetch_codes = collect_pit_codes(
                self.pit_store, self.start, self.end,
                strict_eligibility=self.bcfg.strict_eligibility,
                freq=self.bcfg.rebalance_freq,
            )
            cols: dict[str, pd.Series] = {}
            for code in prefetch_codes:
                try:
                    nav = self.p.get_fund_nav(code, self.start, self.end)
                    cols[str(code)] = self.p.to_returns(nav, self.freq)
                except Exception as exc:  # noqa: BLE001
                    log.warning("回测跳过 %s：%s", code, exc)
            self.fund_ret = pd.DataFrame(cols)
            log.info("预取 %d 只基金收益（prefetch_all_pit_codes=True）", len(self.fund_ret.columns))
        elif not self.bcfg.full_universe_scoring and self.codes:
            self._prefetch_codes([str(c) for c in self.codes])
        self._prepared = True

    def _ensure_return_column(self, code: str) -> None:
        code = str(code)
        if code in self.fund_ret.columns:
            return
        try:
            nav = self.p.get_fund_nav(code, self.start, self.end)
            self.fund_ret[code] = self.p.to_returns(nav, self.freq)
        except Exception as exc:  # noqa: BLE001
            log.warning("动态加载收益失败 %s：%s", code, exc)

    def _prefetch_codes(self, codes: list[str]) -> None:
        for code in codes:
            self._ensure_return_column(str(code))

    def _rebalance(self, t: pd.Timestamp):
        if self.bcfg.full_universe_scoring and self.pit_store is not None:
            cache_key = f"{pit_snapshot_label(self.pit_store, t)}|{t.strftime('%Y-%m-%d')}"
            result = rebalance_at(
                self.p,
                asof=t,
                eval_start=self.bcfg.eval_start,
                pit_store=self.pit_store,
                val_provider=self.vp,
                opt_cfg=self.opt,
                strict_eligibility=self.bcfg.strict_eligibility,
                portfolio_aum_yi=self.bcfg.portfolio_aum_yi,
                max_order_to_fund_aum=self.bcfg.max_order_to_fund_aum,
                etf_participation_rate=self.bcfg.etf_participation_rate,
                etf_adv_lookback=self.bcfg.etf_adv_lookback,
                enforce_active_fund_capacity=self.bcfg.enforce_active_fund_capacity,
                enforce_etf_capacity=self.bcfg.enforce_etf_capacity,
                score_cache_dir=self.bcfg.score_cache_dir or None,
                capacity_asof=t.strftime("%Y-%m-%d"),
                skip_rolling=self.bcfg.backtest_skip_rolling,
                preselect_pool=(
                    self.bcfg.backtest_preselect_pool
                    if self.bcfg.backtest_preselect_pool > 0 else None
                ),
                workers=self.bcfg.backtest_workers,
                provider_kwargs=self._provider_kwargs or None,
            )
            if not result.success or result.weights is None:
                return None, None, None, None, result
            if result.scored is not None:
                self._score_mem_cache[cache_key] = result.scored
            # 预取：当期持仓 + 优化候选（持有期结算与下期调仓 RBSA 更快）
            prefetch = set(result.weights.index.astype(str))
            if result.scored is not None and "code" in result.scored.columns:
                from ..portfolio import select_full_universe_candidates
                try:
                    prefetch.update(select_full_universe_candidates(
                        result.scored, n_active=self.opt.n_candidates,
                        target_growth=result.target_growth,
                    ))
                except Exception:  # noqa: BLE001
                    pass
            self._prefetch_codes(sorted(prefetch))
            all_etf = result.etf_codes | result.synthetic_etf_codes
            return (
                result.weights, result.target_growth,
                result.synthetic_etf_codes, all_etf, result,
            )

        # 兼容 mock / 显式 codes 的小样本路径
        from .universe import filter_universe
        from ..portfolio import build_portfolio, resolve_target_growth, select_full_universe_candidates
        from ..pipeline import score_universe

        asof_str = t.strftime("%Y-%m-%d")
        eligible_codes = list(self.codes) or list(self.fund_ret.columns)
        asset_metadata = self.asset_metadata
        if self.pit_store is not None:
            universe = self.pit_store.read_universe_asof(t, asset_types=("fund", "etf"))
            if "is_qdii" in universe.columns:
                qdii = universe["is_qdii"].astype(str).str.lower().isin(("1", "true", "yes"))
                universe = universe[~qdii]
            universe = filter_universe(universe, asof=t, strict_eligibility=self.bcfg.strict_eligibility)
            eligible_set = set(universe["code"].astype(str))
            eligible_codes = [str(c) for c in eligible_codes if str(c) in eligible_set]
            asset_metadata = {str(row["code"]): row.to_dict() for _, row in universe.iterrows()}

        scored = score_universe(
            self.p, codes=eligible_codes, start=self.bcfg.eval_start, end=asof_str,
            pit_store=self.pit_store, universe_asof=asof_str,
            strict_eligibility=self.bcfg.strict_eligibility,
        )
        target_g = resolve_target_growth(self.p, self.vp, self.bcfg.eval_start, asof_str)
        pick = select_full_universe_candidates(scored, n_active=self.opt.n_candidates, target_growth=target_g)
        opt_result, detail = build_portfolio(
            self.p, pick, target_g, self.bcfg.eval_start, asof_str,
            cfg=self.opt, asset_metadata=asset_metadata or None,
        )
        if not opt_result.success:
            return None, None, None, None, None
        for code in opt_result.weights.index:
            self._ensure_return_column(str(code))
        synth = set(detail.loc[detail["type"] == "ETF补全", "code"].astype(str))
        etf = set(detail.loc[detail["type"].isin(("ETF", "ETF补全")), "code"].astype(str))
        return opt_result.weights, target_g, synth, etf, None

    def _transaction_cost(self, weights: pd.Series, previous: pd.Series | None,
                          etf_codes: set[str]) -> float:
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

        month_end = pd.Series(widx, index=widx).resample(self.bcfg.rebalance_freq).last().dropna()
        rebal_dates = [d for d in month_end.values if d >= widx[warmup]]
        rebal_dates = pd.to_datetime(rebal_dates)
        if len(rebal_dates) < 2:
            raise RuntimeError("可调仓次数不足。")

        freq_label = {"QE": "季度末", "ME": "月末"}.get(self.bcfg.rebalance_freq, self.bcfg.rebalance_freq)

        etf_ret_all = synth_etf_returns(
            self.factor_df, {e: config.STYLE_ETF_ASSETS[e]["growth_load"]
                             for e in config.STYLE_ETF_ASSETS})

        weekly_rows: list[dict] = []
        w_hist, tg_hist, turn_hist, cost_hist = {}, {}, {}, {}
        log_rows: list[dict] = []
        prev_w = None
        prev_etf_codes: set[str] = set()
        skipped = 0
        notes = [
            f"调仓频率：{freq_label}（{len(rebal_dates)} 个调仓日）",
            f"单边交易成本：主动基金 {self.bcfg.fund_one_way_cost:.2%}，ETF {self.bcfg.etf_one_way_cost:.2%}",
            "净值加载：按需（持仓+候选），不预取全 PIT 并集"
            if not self.bcfg.prefetch_all_pit_codes else "净值加载：全 PIT 并集预取",
            "调仓路径：rebalance_at（全池评分 + select_full_universe_candidates + build_portfolio）"
            if self.bcfg.full_universe_scoring and self.pit_store is not None
            else "调仓路径：显式候选 codes + 生产优化器",
            f"主动基金容量：{'启用' if self.bcfg.enforce_active_fund_capacity else '关闭'}",
            f"ETF成交额容量：{'启用' if self.bcfg.enforce_etf_capacity else '关闭'}",
            f"评分加速：skip_rolling={self.bcfg.backtest_skip_rolling}，"
            f"preselect={self.bcfg.backtest_preselect_pool or '全池'}，"
            f"workers={self.bcfg.backtest_workers}",
        ]
        if self.pit_store is None:
            notes.append("未提供PIT数据仓：候选池为静态输入，可能含幸存者偏差。")
        else:
            notes.append("候选池按每个调仓日的PIT主数据快照筛选。")

        for i in range(len(rebal_dates) - 1):
            t0, t1 = rebal_dates[i], rebal_dates[i + 1]
            out = self._rebalance(t0)
            if out[0] is None:
                skipped += 1
                reason = ""
                scored_n = candidate_n = ""
                if out[4] is not None:
                    reason = getattr(out[4], "skip_reason", "")
                    scored_n = getattr(out[4], "scored_count", "")
                    candidate_n = getattr(out[4], "candidate_count", "")
                log_rows.append({
                    "asof": t0, "success": False, "skip_reason": reason or "rebalance_failed",
                    "scored_count": scored_n, "candidate_count": candidate_n,
                })
                if prev_w is None:
                    continue
                weights = prev_w
                target_g = tg_hist.get(rebal_dates[i - 1] if i > 0 else t0, config.BENCHMARK_WEIGHTS["growth"])
                synthetic_etf_set = set()
                all_etf_codes = prev_etf_codes
                trade_cost = 0.0
            else:
                weights, target_g, synthetic_etf_set, all_etf_codes, reb_res = out
                w_hist[t0] = weights
                tg_hist[t0] = target_g
                log_rows.append({
                    "asof": t0, "success": True, "skip_reason": "",
                    "scored_count": getattr(reb_res, "scored_count", ""),
                    "candidate_count": getattr(reb_res, "candidate_count", ""),
                    "n_holdings": int((weights > 1e-6).sum()),
                })
                old_w = prev_w
                if old_w is not None:
                    allc = weights.index.union(old_w.index)
                    turn_hist[t0] = float((weights.reindex(allc, fill_value=0)
                                           - old_w.reindex(allc, fill_value=0)).abs().sum() / 2)
                trade_cost = self._transaction_cost(weights, old_w, all_etf_codes | prev_etf_codes)
                cost_hist[t0] = trade_cost
                prev_w = weights
                prev_etf_codes = all_etf_codes

            sub = self.factor_df.loc[(self.factor_df.index > t0) & (self.factor_df.index <= t1)].index
            for j, d in enumerate(sub):
                gross_r = 0.0
                missing_weight = 0.0
                for code, wt in weights.items():
                    if code in synthetic_etf_set:
                        gross_r += wt * etf_ret_all.loc[d, code]
                    elif code in self.fund_ret.columns and d in self.fund_ret.index:
                        rv = self.fund_ret.loc[d, code]
                        if pd.isna(rv):
                            missing_weight += wt
                            continue
                        gross_r += wt * rv
                if missing_weight > 1e-8:
                    log.debug("%s 缺失收益权重 %.2f%%，未按 0 计入", d.date(), missing_weight * 100)
                cost = trade_cost if j == 0 else 0.0
                weekly_rows.append({
                    "date": d, "gross_port_ret": gross_r,
                    "transaction_cost": cost, "port_ret": gross_r - cost,
                })

        if not weekly_rows:
            raise RuntimeError("回测无任何持有期收益。")
        if skipped:
            notes.append(f"调仓失败/跳过 {skipped} 次；失败期持有上一期权重或空仓。")

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
            "skipped_rebalances": skipped,
        }

        return BacktestOutput(
            nav=nav, weekly=weekly,
            weights_history=pd.DataFrame(w_hist).T.fillna(0.0),
            metrics=metrics,
            target_growth_history=pd.Series(tg_hist),
            turnover_history=pd.Series(turn_hist),
            cost_history=pd.Series(cost_hist),
            rebalances=len(w_hist), notes=notes,
            rebalance_log=pd.DataFrame(log_rows),
        )
