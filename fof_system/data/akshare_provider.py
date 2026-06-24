"""akshare 数据源（免费、无需 token）。

注意：akshare 接口名/字段偶有版本变动。这里对常见接口做了容错与多路兜底；
若某接口在你的 akshare 版本上报错，按下方注释切换备选接口即可。
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from .base import DataProvider, ETF_MARKET_COLUMNS, FundMeta


def _sz_sh_prefix(code: str) -> str:
    """国证指数在深交所发布，东财接口需带市场前缀。399xxx -> sz399xxx。"""
    code = str(code)
    if code.startswith("399"):
        return "sz" + code
    if code.startswith(("000", "0")):  # 上证系列
        return "sh" + code
    return "sz" + code


class AkshareProvider(DataProvider):
    def __init__(self, retry: int = 3, sleep: float = 0.6, cache_dir: str | Path | None = None):
        import akshare as ak  # 延迟导入，未安装时不影响其他数据源
        self.ak = ak
        self.retry = retry
        self.sleep = sleep
        self._fund_list_cache: pd.DataFrame | None = None
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

    def _series_cache_path(self, kind: str, code: str, start: str, end: str) -> Path | None:
        cache_dir = getattr(self, "cache_dir", None)
        if cache_dir is None:
            return None
        start_key = pd.Timestamp(start).strftime("%Y%m%d") if start else "begin"
        end_key = pd.Timestamp(end).strftime("%Y%m%d") if end else "latest"
        return cache_dir / kind / f"{str(code)}_{start_key}_{end_key}.csv"

    def _load_series_cache(self, kind: str, code: str, start: str, end: str) -> pd.Series | None:
        path = self._series_cache_path(kind, code, start, end)
        if path is None or not path.exists():
            return None
        try:
            frame = pd.read_csv(path, parse_dates=["date"])
            if not {"date", "value"}.issubset(frame.columns):
                return None
            return pd.Series(frame["value"].to_numpy(), index=frame["date"], name=str(code)).sort_index()
        except (OSError, ValueError, pd.errors.ParserError):
            return None

    def _store_series_cache(self, kind: str, code: str, start: str, end: str, series: pd.Series) -> None:
        path = self._series_cache_path(kind, code, start, end)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": series.index, "value": series.to_numpy()}).to_csv(
            path, index=False, encoding="utf-8-sig",
        )

    def _with_retry(self, fn, *a, **k):
        last = None
        for _ in range(self.retry):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(self.sleep)
        raise RuntimeError(f"akshare 调用失败: {fn.__name__}: {last}")

    # -- 指数 --------------------------------------------------------------
    def get_index_close(self, code: str, start: str = "", end: str = "") -> pd.Series:
        """多路兜底取指数收盘价：东财 -> 新浪。任一成功即返回。"""
        cached = self._load_series_cache("index_close", code, start, end)
        if cached is not None:
            return cached
        sym = _sz_sh_prefix(code)
        errors = []
        # 路径1：东方财富
        try:
            df = self._with_retry(self.ak.stock_zh_index_daily_em, symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["close"].astype(float).sort_index()
            result = self._slice(s, start, end)
            self._store_series_cache("index_close", code, start, end, result)
            return result
        except Exception as e:  # noqa: BLE001
            errors.append(f"em:{e}")
        # 路径2：新浪
        try:
            df = self._with_retry(self.ak.stock_zh_index_daily, symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["close"].astype(float).sort_index()
            result = self._slice(s, start, end)
            self._store_series_cache("index_close", code, start, end, result)
            return result
        except Exception as e:  # noqa: BLE001
            errors.append(f"sina:{e}")
        raise RuntimeError(f"指数 {code} 取数失败（已试东财/新浪）: {' | '.join(errors)[:300]}")

    @staticmethod
    def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
        if start:
            s = s.loc[start:]
        if end:
            s = s.loc[:end]
        return s

    # -- 基金净值 ----------------------------------------------------------
    def get_fund_nav(self, code: str, start: str = "", end: str = "") -> pd.Series:
        """开放式基金取累计净值；若接口不支持则回退到交易所 ETF 复权收盘价。"""
        cached = self._load_series_cache("fund_nav", code, start, end)
        if cached is not None:
            return cached
        try:
            df = self._with_retry(
                self.ak.fund_open_fund_info_em, symbol=str(code), indicator="累计净值走势"
            )
            date_col = "净值日期" if "净值日期" in df.columns else df.columns[0]
            nav_col = "累计净值" if "累计净值" in df.columns else df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col])
            s = df.set_index(date_col)[nav_col].astype(float).sort_index()
            result = self._slice(s, start, end)
            self._store_series_cache("fund_nav", code, start, end, result)
            return result
        except Exception as open_fund_error:  # noqa: BLE001
            try:
                result = self.get_etf_close(code, start, end)
                self._store_series_cache("fund_nav", code, start, end, result)
                return result
            except Exception as etf_error:  # noqa: BLE001
                raise RuntimeError(
                    f"基金/ETF {code} 取数失败：开放式基金接口={open_fund_error}; ETF接口={etf_error}"
                ) from etf_error

    # -- 股票 ETF ----------------------------------------------------------
    @staticmethod
    def _col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        return next((column for column in candidates if column in df.columns), None)

    def get_etf_market(self, code: str, start: str = "", end: str = "") -> pd.DataFrame:
        """取交易所 ETF 日行情。

        使用前复权收盘价用于收益计算，避免分红除权把 ETF 收益低估；成交量和成交额保留
        原始口径，供流动性与容量约束使用。NAV、折溢价和份额仍须由正式数据供应商补齐。
        """
        start_date = pd.Timestamp(start).strftime("%Y%m%d") if start else "19900101"
        end_date = pd.Timestamp(end).strftime("%Y%m%d") if end else pd.Timestamp.today().strftime("%Y%m%d")
        df = self._with_retry(
            self.ak.fund_etf_hist_em, symbol=str(code), period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq",
        )
        date_col = self._col(df, ("日期", "date", "交易日期"))
        close_col = self._col(df, ("收盘", "close", "收盘价"))
        if date_col is None or close_col is None:
            raise ValueError(f"ETF {code} 行情字段不完整：{list(df.columns)}")

        out = pd.DataFrame({
            "trading_date": pd.to_datetime(df[date_col]),
            "code": str(code),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "source": "akshare:fund_etf_hist_em:qfq",
        })
        for target, candidates in {
            "volume": ("成交量", "volume"),
            "amount": ("成交额", "amount"),
            "turnover_rate": ("换手率", "turnover_rate"),
            "premium_discount": ("折溢价率", "基金折价率", "折价率"),
        }.items():
            source_col = self._col(df, candidates)
            out[target] = pd.to_numeric(df[source_col], errors="coerce") if source_col else float("nan")
        out = out.dropna(subset=["trading_date", "close"]).sort_values("trading_date")
        return out[ETF_MARKET_COLUMNS]

    def list_etfs(self) -> pd.DataFrame:
        """返回东财当前 ETF 快照；分类字段需落盘后以 PIT 主数据维护。"""
        df = self._with_retry(self.ak.fund_etf_spot_em)
        code_col = self._col(df, ("代码", "基金代码", "code"))
        name_col = self._col(df, ("名称", "基金简称", "name"))
        if code_col is None or name_col is None:
            raise ValueError(f"ETF 快照字段不完整：{list(df.columns)}")
        return pd.DataFrame({
            "code": df[code_col].astype(str).str.zfill(6),
            "name": df[name_col].astype(str),
            "asset_type": "etf",
            "fund_type": "ETF",
        })

    # -- 基金清单 ----------------------------------------------------------
    def list_funds(self) -> pd.DataFrame:
        if self._fund_list_cache is None:
            df = self._with_retry(self.ak.fund_name_em)  # 全部公募
            df = df.rename(columns={"基金代码": "code", "基金简称": "name", "基金类型": "fund_type"})
            self._fund_list_cache = df[["code", "name", "fund_type"]].copy()
        return self._fund_list_cache.copy()

    # -- 元数据（规模/经理）-----------------------------------------------
    def get_fund_meta(self, code: str) -> FundMeta:
        meta = FundMeta(code=str(code))
        try:
            info = self._with_retry(self.ak.fund_individual_basic_info_xq, symbol=str(code))
            kv = dict(zip(info["item"], info["value"]))
            meta.name = kv.get("基金名称", "")
            meta.fund_type = kv.get("基金类型", "")
            size = str(kv.get("最新规模", "")).replace("亿", "").strip()
            meta.size_yi = float(size) if size and size.replace(".", "").isdigit() else float("nan")
            est = kv.get("成立时间", "")
            meta.inception = pd.to_datetime(est, errors="coerce") if est else None
            meta.manager = kv.get("基金经理", "")
        except Exception:  # noqa: BLE001 元数据缺失不应中断打分
            pass
        return meta
