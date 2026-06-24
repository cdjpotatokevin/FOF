"""策略稳健性压力测试：参数约束与交易成本的walk-forward敏感性。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Any

import pandas as pd

from .. import config
from ..data.base import DataProvider
from ..data.pit import PITDataStore
from .backtest import ChainBacktester


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    max_weight_fund: float
    risk_aversion: float
    fund_one_way_cost: float
    etf_one_way_cost: float


def default_scenarios(base_opt: config.OptimizerConfig = config.OPTIMIZER,
                      base_bt: config.BacktestConfig = config.BACKTEST) -> list[RobustnessScenario]:
    """一组方向明确的敏感性情景；并不等于全量参数寻优。"""
    return [
        RobustnessScenario("base", base_opt.max_weight_fund, base_opt.risk_aversion,
                           base_bt.fund_one_way_cost, base_bt.etf_one_way_cost),
        RobustnessScenario("high_cost", base_opt.max_weight_fund, base_opt.risk_aversion,
                           base_bt.fund_one_way_cost * 2, base_bt.etf_one_way_cost * 2),
        RobustnessScenario("conservative", base_opt.max_weight_fund, base_opt.risk_aversion * 2,
                           base_bt.fund_one_way_cost, base_bt.etf_one_way_cost),
        RobustnessScenario("loose", max(base_opt.max_weight_fund, 0.20), base_opt.risk_aversion / 2,
                           base_bt.fund_one_way_cost, base_bt.etf_one_way_cost),
    ]


def run_robustness(
    provider: DataProvider,
    codes: list[str],
    start: str,
    end: str,
    *,
    val_provider: DataProvider | None = None,
    pit_store: PITDataStore | None = None,
    asset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    scenarios: list[RobustnessScenario] | None = None,
) -> pd.DataFrame:
    """在同一候选池和决策顺序下比较不同约束/成本情景。"""
    scenarios = scenarios or default_scenarios()
    rows: list[dict] = []
    for scenario in scenarios:
        opt = replace(config.OPTIMIZER,
                      max_weight_fund=scenario.max_weight_fund,
                      risk_aversion=scenario.risk_aversion)
        bt = replace(config.BACKTEST,
                     fund_one_way_cost=scenario.fund_one_way_cost,
                     etf_one_way_cost=scenario.etf_one_way_cost)
        try:
            output = ChainBacktester(
                provider, codes, start, end, val_provider=val_provider,
                opt_cfg=opt, backtest_cfg=bt, pit_store=pit_store,
                asset_metadata=asset_metadata,
            ).run()
            rows.append({
                "scenario": scenario.name,
                "max_weight_fund": scenario.max_weight_fund,
                "risk_aversion": scenario.risk_aversion,
                "fund_one_way_cost": scenario.fund_one_way_cost,
                "etf_one_way_cost": scenario.etf_one_way_cost,
                "rebalances": output.rebalances,
                **output.metrics,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "scenario": scenario.name,
                "max_weight_fund": scenario.max_weight_fund,
                "risk_aversion": scenario.risk_aversion,
                "fund_one_way_cost": scenario.fund_one_way_cost,
                "etf_one_way_cost": scenario.etf_one_way_cost,
                "error": str(exc),
            })
    return pd.DataFrame(rows)
