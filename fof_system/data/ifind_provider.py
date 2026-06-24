"""同花顺 iFinD 数据源（经 ifind-finance-data skill 的 MCP 通道）。

设计取舍
--------
该 skill 是"自然语言 query"接口，并提示长区间会被截断。因此：
  • 长历史的指数收盘价 / 基金净值 —— 建议仍用 akshare（稳、免费、全序列），
    iFinD 这里也实现了，但更适合较短区间或快照。
  • iFinD 的独特价值 —— 指数估值(PE/PB)价差、宏观/行业 EDB、基金持仓穿透，
    这些 akshare 拿不全，正是风格择时(第③层)需要的。

返回解析
--------
MCP 工具返回的是文本块（可能是 JSON 字符串或 markdown 表）。_parse_timeseries
做了双路解析（JSON 优先，回退表格）。**首次拿到真实密钥后，建议用
scripts/probe_ifind.py 打一条真实返回核对字段名，必要时微调解析。**
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd

from .base import DataProvider, FundMeta
from .ifind_client import IFindClient, extract_text_blocks


_INDEX_NAME = {"399370": "国证成长", "399371": "国证价值"}

# iFinD 经 skill 的 NL 查询单次最多返回约 100 行；超长区间会被截断甚至退化成快照。
# 因此长历史按 ~90 天窗口分块抓取再拼接。
_CHUNK_DAYS = 90


def _unwrap(text: str) -> object:
    """剥掉 iFinD 外层信封，返回内层可解析对象（markdown 表字符串 或 list[dict]）。

    实测返回形如：{"code":1,"msg":"success","data":{"text":"|列1|列2|...markdown表..."}}
    也兼容 data 直接是 list[dict] 的情况。
    """
    t = text.strip()
    try:
        obj = json.loads(t)
    except Exception:  # noqa: BLE001
        return t  # 已是纯文本/markdown
    if isinstance(obj, dict):
        d = obj.get("data", obj)
        if isinstance(d, dict) and "text" in d:
            return d["text"]           # 最常见：markdown 表在 data.text
        if isinstance(d, (list, str)):
            return d
        return obj
    return obj


def _parse_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if re.fullmatch(r"\d{8}", s):
        return pd.to_datetime(s, format="%Y%m%d")
    return pd.to_datetime(s.replace("/", "-"))


def _pick_col(header: list[str], hints: tuple[str, ...]) -> int | None:
    for i, h in enumerate(header):
        hl = str(h).lower()
        if any(t.lower() in hl for t in hints):
            return i
    return None


def _parse_md_table(text: str, value_hint: tuple[str, ...]) -> pd.Series:
    """解析 markdown 管道表（| a | b |）为 [日期, 数值] 序列。"""
    lines = [ln for ln in text.splitlines() if "|" in ln]
    header: list[str] | None = None
    rows: dict[pd.Timestamp, float] = {}
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not any(cells):
            continue
        if all(set(c) <= set("-: ") for c in cells if c):  # 分隔行 |---|---|
            continue
        if header is None:
            header = cells
            continue
        if len(cells) < len(header):
            continue
        dcol = _pick_col(header, ("日期", "date", "时间", "time"))
        vcol = _pick_col(header, value_hint)
        if dcol is None:
            continue
        if vcol is None:  # 退而取最后一列
            vcol = len(header) - 1
        try:
            rows[_parse_date(cells[dcol])] = float(cells[vcol])
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        raise ValueError(f"markdown 表无法解析为序列：{text[:200]}")
    return pd.Series(rows).sort_index()


def _parse_timeseries(text: str, value_hint: tuple[str, ...]) -> pd.Series:
    """把 iFinD 文本返回解析成 [日期, 数值] 序列。

    value_hint：在多列里挑目标数值列的关键词（如 'close','收盘','pe','市盈'）。
    """
    inner = _unwrap(text)
    # 路1：list[dict]
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        cols = list(inner[0].keys())
        dcol = next((c for c in cols if any(t in str(c).lower() for t in ("date", "time", "日期", "时间"))), None)
        vcol = next((c for c in cols if any(t.lower() in str(c).lower() for t in value_hint)), None) or \
            [c for c in cols if c != dcol][-1]
        rows = {_parse_date(r[dcol]): float(r[vcol]) for r in inner
                if r.get(dcol) and r.get(vcol) not in (None, "")}
        return pd.Series(rows).sort_index()
    # 路2：markdown / 纯文本表
    return _parse_md_table(str(inner), value_hint)


def _chunk_windows(start: str, end: str, days: int = _CHUNK_DAYS):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end or pd.Timestamp.today())
    cur = s
    while cur <= e:
        nxt = min(cur + pd.Timedelta(days=days - 1), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + pd.Timedelta(days=1)


class IFinDProvider(DataProvider):
    def __init__(self, skill_dir: str | None = None, timeout: int = 90,
                 cache_dir: str | None = None, verbose: bool = False):
        self.cli = IFindClient(skill_dir=skill_dir, timeout=timeout)
        self.verbose = verbose
        # 分块 NL 取数很慢，磁盘缓存每个窗口，避免重复抓取
        self.cache_dir = Path(cache_dir or (Path(__file__).resolve().parent.parent / ".cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- 通用：分块 + 缓存抓取一段时间序列 --------------------------------
    def _fetch_series(self, kind: str, code: str, metric: str,
                      start: str, end: str, hints: tuple[str, ...]) -> pd.Series:
        parts: list[pd.Series] = []
        for ws, we in _chunk_windows(start, end):
            ck = self.cache_dir / f"ifind_{kind}_{code}_{metric}_{ws}_{we}.csv"
            if ck.exists():
                s = pd.read_csv(ck, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                s = self._fetch_window(kind, code, metric, ws, we, hints)
                s.to_frame("v").to_csv(ck)
                if self.verbose:
                    print(f"  iFinD {kind} {code} {ws}~{we}: {len(s)} 行")
            parts.append(s)
        if not parts:
            raise RuntimeError(f"iFinD {kind} {code} 无数据")
        full = pd.concat(parts).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        if start:
            full = full.loc[start:]
        if end:
            full = full.loc[:end]
        return full

    def _fetch_window(self, kind: str, code: str, metric: str,
                      ws: str, we: str, hints: tuple[str, ...], retries: int = 4) -> pd.Series:
        """抓单个窗口。iFinD 的 NL 接口偶发把区间退化成"最新快照"（单行无日期），
        故检测到 <2 行即视为失败、换措辞重试。"""
        name = _INDEX_NAME.get(str(code), str(code))
        if kind == "valuation":
            mname = {"pe_ttm": "市盈率PE(TTM)", "pb": "市净率PB", "pe": "市盈率"}.get(metric, metric)
            phrasings = [
                f"{name}指数 {ws} 到 {we} 的每日{mname}",
                f"{name}指数在{ws}至{we}期间每个交易日的{mname}历史序列",
            ]
        else:
            phrasings = [
                f"{name}指数 {ws} 到 {we} 的每日收盘点数",
                f"{name}指数在{ws}至{we}期间每个交易日的收盘点数历史序列",
            ]
        last = ""
        for attempt in range(retries):
            res = self.cli.call("index", "index_data", {"query": phrasings[attempt % len(phrasings)]})
            if not res.get("ok", False):
                last = f"接口错误 {res.get('error')}"
                continue
            try:
                s = _parse_timeseries(extract_text_blocks(res), hints)
                if len(s) >= 2:    # 拒绝单行快照
                    return s
                last = "退化为快照(单行)"
            except Exception as e:  # noqa: BLE001
                last = str(e)[:80]
        raise RuntimeError(f"iFinD 取数失败({kind} {code} {ws}~{we})，{retries}次重试后仍失败: {last}")

    # -- 指数收盘价（分块）------------------------------------------------
    def get_index_close(self, code: str, start: str, end: str) -> pd.Series:
        return self._fetch_series("close", str(code), "close", start, end,
                                  ("close", "收盘", "点数"))

    # -- 指数估值（iFinD 的核心价值，分块）-------------------------------
    def get_index_valuation(self, code: str, metric: str = "pe_ttm",
                            start: str = "", end: str = "") -> pd.Series:
        start = start or "2015-01-01"
        return self._fetch_series("valuation", str(code), metric, start, end,
                                  ("pe", "pb", "市盈", "市净", "估值"))

    # -- 基金净值 ----------------------------------------------------------
    def get_fund_nav(self, code: str, start: str, end: str) -> pd.Series:
        q = f"基金{code} {start} 到 {end} 的每日累计净值"
        res = self.cli.call("fund", "get_fund_market_performance", {"query": q})
        if not res.get("ok", False):
            raise RuntimeError(f"iFinD 净值取数失败: {res.get('error')}")
        return _parse_timeseries(extract_text_blocks(res), ("nav", "净值", "累计"))

    # -- 基金清单 ----------------------------------------------------------
    def list_funds(self) -> pd.DataFrame:
        raise NotImplementedError(
            "iFinD 经 skill 取全市场清单不便；请用 search_funds 选基或传入自有清单 CSV。"
            "长历史净值建议用 akshare。"
        )

    # -- 基金元数据 --------------------------------------------------------
    def get_fund_meta(self, code: str) -> FundMeta:
        meta = FundMeta(code=str(code))
        try:
            res = self.cli.call("fund", "get_fund_profile",
                                {"query": f"基金{code}的最新规模、成立日期、基金经理及任职日期"})
            txt = extract_text_blocks(res)
            meta.name = self._grab(txt, ("基金名称", "基金简称")) or ""
            size = self._grab_num(txt, ("最新规模", "资产规模", "规模"))
            if size is not None:
                meta.size_yi = size / 1e8 if size > 1e6 else size  # 元→亿 容错
            meta.manager = self._grab(txt, ("基金经理",)) or ""
            inc = self._grab(txt, ("成立日期", "成立时间"))
            meta.inception = pd.to_datetime(inc, errors="coerce") if inc else None
            ms = self._grab(txt, ("任职日期", "现任经理任职"))
            meta.manager_start = pd.to_datetime(ms, errors="coerce") if ms else None
        except Exception:  # noqa: BLE001
            pass
        return meta

    @staticmethod
    def _grab(text: str, keys: tuple[str, ...]) -> str | None:
        for k in keys:
            m = re.search(rf"{k}[：:\s]*([^\s，,；;|]+)", text)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _grab_num(text: str, keys: tuple[str, ...]) -> float | None:
        for k in keys:
            m = re.search(rf"{k}[：:\s]*([\d.]+)", text)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None
