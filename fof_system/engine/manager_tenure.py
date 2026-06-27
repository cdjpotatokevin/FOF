"""现任基金经理在本基金的任职时长（用于 strict 合规池）。"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

_MANAGER_JS_RE = re.compile(r"currentFundManager\s*=\s*(\[.*?\])\s*;", re.S)
_F10_TENURE_ROW_RE = re.compile(
    r"<tr[^>]*>\s*"
    r"<td[^>]*>(\d{4}-\d{2}-\d{2})</td>\s*"
    r"<td[^>]*>(至今|--|----)</td>\s*"
    r"<td[^>]*>(?:<a[^>]*>)?([^<]+?)(?:</a>)?\s*</td>",
    re.I | re.S,
)
_WORK_TIME_RE = re.compile(r"(\d+)年")
_WORK_DAYS_RE = re.compile(r"(\d+)天")


def parse_eastmoney_work_time(work_time: object, asof: str | pd.Timestamp) -> pd.Timestamp | pd.NaT:
    """把东方财富 ``workTime``（如“1年又312天”）换算为任职起始日。"""
    text = str(work_time or "").strip()
    if not text:
        return pd.NaT
    years = int(_WORK_TIME_RE.search(text).group(1)) if _WORK_TIME_RE.search(text) else 0
    days = int(_WORK_DAYS_RE.search(text).group(1)) if _WORK_DAYS_RE.search(text) else 0
    if years == 0 and days == 0:
        return pd.NaT
    point = pd.Timestamp(asof).normalize()
    return point - pd.Timedelta(days=years * 365.25 + days)


def parse_f10_current_manager_starts(html: str) -> tuple[list[pd.Timestamp], list[str]]:
    """解析天天基金 F10「基金经理变动」表中现任经理在本基金的任职起始日。"""
    starts: list[pd.Timestamp] = []
    names: list[str] = []
    for start_text, end_text, manager_text in _F10_TENURE_ROW_RE.findall(html):
        if str(end_text).strip() != "至今":
            continue
        start = pd.to_datetime(start_text, errors="coerce")
        if pd.isna(start):
            continue
        starts.append(pd.Timestamp(start).normalize())
        names.append(str(manager_text).strip())
    return starts, names


def _fetch_one_manager_start(code: str, asof: str) -> dict:
    """拉取现任经理在本基金的任职起始日（F10 变动表为准）。"""
    f10_url = f"https://fundf10.eastmoney.com/jjjl_{str(code).zfill(6)}.html"
    try:
        response = requests.get(f10_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        starts, names = parse_f10_current_manager_starts(response.text)
        if starts:
            # 共管基金取最短任职起点：任何现任经理未满一年均不放行。
            return {
                "code": str(code).zfill(6),
                "manager_start": max(starts),
                "manager": ",".join(name for name in names if name),
                "work_time_raw": "|".join(
                    f"{name}:{start.strftime('%Y-%m-%d')}" for name, start in zip(names, starts)
                ),
                "tenure_source": "f10",
            }
    except Exception:  # noqa: BLE001 单只失败不阻断全池
        pass

    # F10 失败时回退 pingzhongdata；其 workTime 为经理累计从业年限，仅作兜底。
    js_url = f"https://fund.eastmoney.com/pingzhongdata/{str(code).zfill(6)}.js"
    try:
        response = requests.get(js_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        match = _MANAGER_JS_RE.search(response.text)
        if not match:
            return {
                "code": str(code).zfill(6),
                "manager_start": pd.NaT,
                "manager": "",
                "work_time_raw": "",
                "tenure_source": "",
            }
        managers = json.loads(match.group(1))
        if not managers:
            return {
                "code": str(code).zfill(6),
                "manager_start": pd.NaT,
                "manager": "",
                "work_time_raw": "",
                "tenure_source": "",
            }
        starts = []
        names = []
        raw_times = []
        for mgr in managers:
            raw = str(mgr.get("workTime", "") or "").strip()
            start = parse_eastmoney_work_time(raw, asof)
            if pd.notna(start):
                starts.append(start)
            names.append(str(mgr.get("name", "") or "").strip())
            raw_times.append(raw)
        if not starts:
            return {
                "code": str(code).zfill(6),
                "manager_start": pd.NaT,
                "manager": ",".join(name for name in names if name),
                "work_time_raw": "|".join(raw_times),
                "tenure_source": "pingzhongdata_fallback",
            }
        return {
            "code": str(code).zfill(6),
            "manager_start": max(starts),
            "manager": ",".join(name for name in names if name),
            "work_time_raw": "|".join(raw_times),
            "tenure_source": "pingzhongdata_fallback",
        }
    except Exception:  # noqa: BLE001
        return {
            "code": str(code).zfill(6),
            "manager_start": pd.NaT,
            "manager": "",
            "work_time_raw": "",
            "tenure_source": "",
        }


def fetch_eastmoney_manager_starts(
    codes: list[str],
    asof: str | pd.Timestamp,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    """批量拉取现任基金经理在本基金的任职起始日。"""
    asof_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
    unique_codes = sorted({str(code).zfill(6) for code in codes if str(code).strip()})
    if not unique_codes:
        return pd.DataFrame(columns=["code", "manager_start", "manager", "work_time_raw", "tenure_source"])
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_manager_start, code, asof_str): code for code in unique_codes}
        for future in as_completed(futures):
            rows.append(future.result())
    frame = pd.DataFrame(rows)
    frame["code"] = frame["code"].astype(str)
    frame["manager_start"] = pd.to_datetime(frame["manager_start"], errors="coerce")
    return frame


def _cache_path(cache_dir: Path, asof: str) -> Path:
    return cache_dir / f"manager_fund_tenure_asof={asof}.csv"


def enrich_manager_start(
    universe: pd.DataFrame,
    asof: str | pd.Timestamp,
    *,
    cache_dir: Path | str | None = None,
    max_workers: int = 8,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """为缺失 ``manager_start`` 的主动基金补全本基金任职起始日。"""
    frame = universe.copy()
    if "manager_start" not in frame.columns:
        frame["manager_start"] = pd.NaT
    frame["manager_start"] = pd.to_datetime(frame["manager_start"], errors="coerce")
    asof_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
    asset_type = frame.get("asset_type", pd.Series("fund", index=frame.index)).fillna("fund").astype(str).str.lower()
    is_etf = asset_type.eq("etf")
    need = frame.loc[~is_etf & frame["manager_start"].isna(), "code"].astype(str).tolist()
    if not need:
        return frame

    fetched = pd.DataFrame(columns=["code", "manager_start", "manager", "work_time_raw", "tenure_source"])
    cache_file: Path | None = None
    if cache_dir:
        cache_file = _cache_path(Path(cache_dir), asof_str)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists() and not force_refresh:
            cached = pd.read_csv(cache_file, dtype={"code": "string"})
            cached["manager_start"] = pd.to_datetime(cached["manager_start"], errors="coerce")
            if "tenure_source" not in cached.columns:
                cached["tenure_source"] = ""
            # 旧缓存或 pingzhongdata 兜底数据不可靠，强制重拉。
            cached = cached[cached["tenure_source"].astype(str).eq("f10")].copy()
            fetched = cached

    cached_codes = set(fetched["code"].astype(str)) if not fetched.empty else set()
    missing = [code for code in need if code not in cached_codes]
    if missing:
        fresh = fetch_eastmoney_manager_starts(missing, asof_str, max_workers=max_workers)
        if fetched.empty:
            fetched = fresh
        else:
            fetched = pd.concat([fetched, fresh], ignore_index=True).drop_duplicates("code", keep="last")
        if cache_file is not None:
            fetched.to_csv(cache_file, index=False, encoding="utf-8-sig")

    if fetched.empty:
        return frame
    by_code = fetched.set_index(fetched["code"].astype(str))
    for idx, row in frame.loc[~is_etf & frame["manager_start"].isna()].iterrows():
        code = str(row["code"])
        if code not in by_code.index:
            continue
        record = by_code.loc[code]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[-1]
        if pd.notna(record.get("manager_start", pd.NaT)):
            frame.at[idx, "manager_start"] = record["manager_start"]
        if "manager" in frame.columns and not str(frame.at[idx, "manager"]).strip():
            frame.at[idx, "manager"] = record.get("manager", "")
    return frame


def filter_strict_universe(
    universe: pd.DataFrame,
    asof: str | pd.Timestamp,
    *,
    flt=None,
    enrich: bool = True,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """strict 合规池：先放宽经理任期做初筛，必要时补全任职日后再次过滤。"""
    from dataclasses import replace

    from ..config import DEFAULT_FILTER
    from .universe import filter_universe

    use_flt = flt or DEFAULT_FILTER
    if not use_flt.require_manager_tenure or use_flt.min_manager_tenure_years <= 0:
        return filter_universe(universe, flt=use_flt, asof=asof, strict_eligibility=True)

    prelim = filter_universe(
        universe,
        flt=replace(use_flt, min_manager_tenure_years=0),
        asof=asof,
        strict_eligibility=True,
    )
    if enrich and not prelim.empty:
        prelim = enrich_manager_start(prelim, asof=asof, cache_dir=cache_dir)
    return filter_universe(prelim, flt=use_flt, asof=asof, strict_eligibility=True)
