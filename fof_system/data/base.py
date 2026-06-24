"""数据源抽象接口。

新增数据源只需继承 DataProvider 并实现下面三个方法，引擎其余部分无需改动。
所有"收益序列"统一约定：pandas.Series，index 为 DatetimeIndex（升序），值为
简单收益率（非百分比，0.01 = 1%）。所有"净值"统一为复权/累计净值。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class FundMeta:
    code: str
    name: str = ""
    fund_type: str = ""           # 基金类型（股票型/偏股混合/指数增强...）
    size_yi: float = float("nan")  # 最新规模（亿元）
    inception: Optional[pd.Timestamp] = None  # 成立日
    manager: str = ""
    manager_start: Optional[pd.Timestamp] = None  # 现任经理任职日


ETF_MARKET_COLUMNS = [
    "trading_date", "code", "close", "volume", "amount", "turnover_rate",
    "premium_discount", "source",
]


class DataProvider(ABC):
    """统一数据接口。"""

    # -- 指数（风格基准）---------------------------------------------------
    @abstractmethod
    def get_index_close(self, code: str, start: str, end: str) -> pd.Series:
        """返回指数收盘价 Series（DatetimeIndex, 升序）。code 如 '399370'。"""

    # -- 基金净值 ----------------------------------------------------------
    @abstractmethod
    def get_fund_nav(self, code: str, start: str, end: str) -> pd.Series:
        """返回基金复权/累计净值 Series（DatetimeIndex, 升序）。"""

    # -- 基金清单与元数据 --------------------------------------------------
    @abstractmethod
    def list_funds(self) -> pd.DataFrame:
        """返回候选基金清单 DataFrame，至少含列: code, name, fund_type。"""

    def get_fund_meta(self, code: str) -> FundMeta:
        """单只基金元数据。默认空实现，子类可覆盖以补全规模/任期等。"""
        return FundMeta(code=code)

    # -- 股票 ETF（可选；用于真实 ETF 行情而非风格合成资产）---------------
    def list_etfs(self) -> pd.DataFrame:
        """返回当前 ETF 快照。

        标准列至少应包含 code/name/asset_type；生产研究应额外维护 is_stock_etf、
        is_qdii、status、fund_type 与 available_at，不能只凭名称做范围判断。
        默认空表，使只支持开放式基金的数据源仍可工作。
        """
        return pd.DataFrame(columns=["code", "name", "asset_type", "fund_type"])

    def get_etf_market(self, code: str, start: str = "", end: str = "") -> pd.DataFrame:
        """返回 ETF 日行情，列遵循 ETF_MARKET_COLUMNS。"""
        raise NotImplementedError(f"{type(self).__name__} 不提供 ETF 行情。")

    def get_etf_close(self, code: str, start: str = "", end: str = "") -> pd.Series:
        """从 ETF 日行情取复权/研究口径的收盘价。"""
        df = self.get_etf_market(code, start, end)
        if "trading_date" not in df or "close" not in df:
            raise ValueError("ETF 行情缺少 trading_date 或 close 列")
        return pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df["trading_date"]), name=str(code)).sort_index()

    # -- 指数估值（用于风格择时的估值价差信号；可选）----------------------
    def get_index_valuation(self, code: str, metric: str = "pe_ttm",
                            start: str = "", end: str = "") -> pd.Series:
        """返回指数估值序列（如 PE-TTM / PB）。

        默认未实现：数据源不支持估值时，风格择时会自动跳过估值信号。
        iFinD 支持；akshare 对国证指数估值覆盖有限，故默认不实现。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不提供指数估值；估值价差信号将被跳过。"
        )

    # -- 通用工具：净值/收盘价 -> 收益率 ------------------------------------
    @staticmethod
    def to_returns(price: pd.Series, freq: str = "W") -> pd.Series:
        """价格序列 -> 指定频率简单收益率。freq: 'W'(周) / 'D'(日) / 'M'(月)。"""
        s = price.dropna().sort_index()
        if freq.upper() == "D":
            return s.pct_change(fill_method=None).dropna()

        # 'W' / 'M'：按周期末值重采样后求收益。若研究截止日落在周期中间，pandas
        # 会把该不完整周期标记为未来的周五/月末；在PIT研究里不能让这种标签混入
        # 截止日后的结果，因此只保留已经完整结束的周期。
        rule = {"W": "W-FRI", "M": "ME"}.get(freq.upper(), "W-FRI")
        period_close = s.resample(rule).last()
        if len(period_close) and period_close.index[-1] > s.index[-1]:
            period_close = period_close.iloc[:-1]

        # 缺失期不能被默认前向填充为“零收益”；这既掩盖数据缺口，也会随 pandas
        # 默认行为变更而改变回测结果。保留缺失并由上层在对齐时显式处理。
        return period_close.pct_change(fill_method=None).dropna()
