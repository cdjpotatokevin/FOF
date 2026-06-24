"""基于iFinD HTTP超级命令协议的估值与ETF成交额数据源。

实现的协议：
* THS_DS -> ``date_sequence``：国证成长/价值的PE、PB时间序列；
* THS_HQ -> ``cmd_history_quotation``：ETF的收盘价和成交额。

refresh token 仅由 :class:`IFindHTTPClient` 从进程环境读取，绝不写入缓存或代码。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .base import DataProvider, ETF_MARKET_COLUMNS, FundMeta
from .ifind_http import IFindHTTPClient, IFindHTTPError, response_to_frame


_VALUATION_SPEC = {
    "pe_ttm": ("ths_pe_ttm_sr_index", ["101", "100"]),
    "pb": ("ths_pb_latest_index", ["100"]),
}


def _code_without_market(code: str) -> str:
    return str(code).strip().split(".", 1)[0].zfill(6)


def _ifind_index_code(code: str) -> str:
    """把项目内裸指数代码规范为iFinD日期序列所需的交易所代码。"""
    value = str(code).strip().upper()
    if "." in value:
        return value
    raw = _code_without_market(value)
    return f"{raw}.SZ" if raw.startswith("399") else f"{raw}.SH"


def _date_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).lower() in {"time", "date", "trading_date", "日期", "时间"}:
            return str(column)
    raise IFindHTTPError(f"iFinD响应缺少日期列，实际字段: {list(frame.columns)}")


def _metric_column(frame: pd.DataFrame, indicator: str) -> str:
    exact = [column for column in frame.columns if str(column).lower() == indicator.lower()]
    if exact:
        return str(exact[0])
    candidates = [column for column in frame.columns if indicator.lower() in str(column).lower()]
    if len(candidates) == 1:
        return str(candidates[0])
    raise IFindHTTPError(f"iFinD响应未找到指标 {indicator}，实际字段: {list(frame.columns)}")


def _annual_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """将日期序列拆成不超过一年的请求，规避服务端长区间静默截断。"""
    left, right = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if left > right:
        raise ValueError("start 不能晚于 end")
    chunks: list[tuple[str, str]] = []
    while left <= right:
        chunk_end = min(left + pd.DateOffset(years=1) - pd.Timedelta(days=1), right)
        chunks.append((left.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        left = chunk_end + pd.Timedelta(days=1)
    return chunks


class IFindHTTPProvider(DataProvider):
    """iFinD HTTP适配器；适合作为 ``--val-source ifind_http`` 与ETF流动性来源。"""

    def __init__(self, timeout: int = 45, client: IFindHTTPClient | None = None,
                 cache_dir: str | Path | None = None,
                 calendar_provider: DataProvider | None = None):
        self.client = client or IFindHTTPClient(timeout=timeout, cache_dir=cache_dir)
        self.calendar_provider = calendar_provider
        self._akshare_cache_dir = Path(cache_dir).parent / "akshare" if cache_dir else None

    def _etf_trading_dates(self, code: str, start: str, end: str, expected_len: int) -> pd.DatetimeIndex:
        """无日期THS_HQ响应时，用同一ETF的AkShare净值交易日历严格恢复日期。"""
        provider = self.calendar_provider
        if provider is None:
            from .akshare_provider import AkshareProvider
            provider = AkshareProvider(cache_dir=self._akshare_cache_dir)
        dates = pd.DatetimeIndex(provider.get_fund_nav(_code_without_market(code), start, end).index).sort_values()
        if len(dates) != expected_len:
            raise IFindHTTPError(
                f"iFinD无日期ETF行情长度 {expected_len} 与AkShare交易日历长度 {len(dates)} 不一致，拒绝写入PIT。"
            )
        return dates

    def get_index_close(self, code: str, start: str, end: str) -> pd.Series:
        raise NotImplementedError("iFinD HTTP适配器目前仅提供估值；指数价格请使用 akshare 数据源")

    def get_index_valuation(self, code: str, metric: str = "pe_ttm",
                            start: str = "", end: str = "") -> pd.Series:
        if metric not in _VALUATION_SPEC:
            raise ValueError(f"不支持的指数估值指标: {metric}")
        if not start or not end:
            raise ValueError("iFinD日期序列估值必须明确指定 start 和 end")
        indicator, params = _VALUATION_SPEC[metric]
        # 用户提供的 THS_DS 中 block:history 是指标的服务端参数，映射为 otherparams。
        request_code = _ifind_index_code(code)
        dated_parts: list[pd.Series] = []
        undated_values: list[pd.Series] = []
        # 实测服务端对超过约五年的长区间会静默截断。年度分段既限制单次数据量，
        # 又确保每个无日期返回数组能与价格交易日历严格一一对应。
        for chunk_start, chunk_end in _annual_chunks(start, end):
            response = self.client.date_sequence(
                request_code,
                [{"indicator": indicator, "indiparams": params, "otherparams": {"block": "history"}}],
                chunk_start, chunk_end,
                functionpara={"Days": "Tradedays", "Fill": "Blank", "Interval": "D"},
            )
            frame = response_to_frame(response)
            if "thscode" in frame:
                frame = frame[frame["thscode"].astype(str).str.upper().eq(request_code)]
            value_col = _metric_column(frame, indicator)
            values = pd.to_numeric(frame[value_col], errors="coerce")
            try:
                date_col = _date_column(frame)
            except IFindHTTPError:
                undated_values.append(values.reset_index(drop=True))
            else:
                dated_parts.append(pd.Series(
                    values.to_numpy(), index=pd.to_datetime(frame[date_col], errors="coerce"), name=metric,
                ).dropna())

        if dated_parts and undated_values:
            raise IFindHTTPError("iFinD日期序列同时出现有日期和无日期响应，拒绝混合拼接")
        if undated_values:
            # 调用方StyleTimer会用同一请求区间的价格主交易日历恢复日期；此处不猜测节假日。
            series = pd.concat(undated_values, ignore_index=True).dropna()
            series.name = metric
            series.attrs["ifind_date_sequence_without_dates"] = True
            series.attrs["start"] = start
            series.attrs["end"] = end
        else:
            series = pd.concat(dated_parts).dropna().sort_index() if dated_parts else pd.Series(dtype=float, name=metric)
        if series.empty:
            raise IFindHTTPError(f"iFinD未返回 {request_code} 的 {metric} 有效数据")
        return series[~series.index.duplicated(keep="last")]

    def get_etf_market(self, code: str, start: str, end: str) -> pd.DataFrame:
        # 用户给出的THS_HQ(amount)同端点；额外请求close以满足PIT行情主键/价格合同。
        response = self.client.history_quotation(str(code), "close,amount", start, end)
        frame = response_to_frame(response)
        if "thscode" in frame:
            frame = frame[frame["thscode"].astype(str).eq(str(code))]
        close_col = _metric_column(frame, "close")
        amount_col = _metric_column(frame, "amount")
        try:
            dates = pd.to_datetime(frame[_date_column(frame)], errors="coerce")
        except IFindHTTPError:
            dates = self._etf_trading_dates(code, start, end, len(frame))
        out = pd.DataFrame({
            "trading_date": dates,
            "code": _code_without_market(code),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
            "amount": pd.to_numeric(frame[amount_col], errors="coerce"),
            "source": "ifind_http:cmd_history_quotation",
        }).dropna(subset=["trading_date", "close"])
        for column in ETF_MARKET_COLUMNS:
            if column not in out:
                out[column] = float("nan")
        out = out[ETF_MARKET_COLUMNS]
        out.attrs["ifind_request"] = dict(getattr(self.client, "last_request_metadata", {}))
        return out

    def get_fund_nav(self, code: str, start: str, end: str) -> pd.Series:
        raise NotImplementedError("iFinD HTTP适配器尚未配置基金净值指标；请使用 akshare")

    def list_funds(self) -> pd.DataFrame:
        raise NotImplementedError("基金池请使用 p04955 PIT快照")

    def get_fund_meta(self, code: str) -> FundMeta:
        return FundMeta(code=str(code))
