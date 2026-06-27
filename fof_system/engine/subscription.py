"""申购状态判定与实时状态拉取（PIT 与交易日口径共用）。"""
from __future__ import annotations

import pandas as pd

# 仅接受“开放申购 / 开放大额申购”；暂停、限大额、限额等均不放行。
SUBSCRIPTION_OPEN_RE = r"开放(?:大额)?申购"
SUBSCRIPTION_BLOCK_RE = r"暂停|封闭|限大额|限额|限购|限制"


def subscription_status_open(status: object) -> bool:
    """单条申购状态是否视为可建仓的开放申购。"""
    text = str(status or "").strip()
    if not text:
        return False
    series = pd.Series([text])
    opening = series.str.contains(SUBSCRIPTION_OPEN_RE, regex=True, na=False).iloc[0]
    blocked = series.str.contains(SUBSCRIPTION_BLOCK_RE, regex=True, na=False).iloc[0]
    return bool(opening and not blocked)


def subscription_open_mask(df: pd.DataFrame) -> pd.Series:
    """与 ``filter_universe`` 中主动基金开放申购判定一致。"""
    status = df.get("subscription_status", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    opening = status.str.contains(SUBSCRIPTION_OPEN_RE, regex=True, na=False)
    blocked = status.str.contains(SUBSCRIPTION_BLOCK_RE, regex=True, na=False)
    explicit_limit = _bool_col(df, "has_daily_subscription_limit")
    if "daily_subscription_limit_yi" in df:
        limit_yi = pd.to_numeric(df["daily_subscription_limit_yi"], errors="coerce")
        explicit_limit = explicit_limit | limit_yi.notna() & (limit_yi > 0)
    return opening & ~blocked & ~explicit_limit


def _bool_col(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index)
    return df[column].astype(str).str.lower().isin(("1", "true", "yes"))


def fetch_akshare_subscription_status() -> pd.DataFrame:
    """拉取东方财富开放式基金当日申购/赎回状态。"""
    import akshare as ak

    raw = ak.fund_purchase_em()
    status = raw["申购状态"].astype(str).str.strip()
    frame = pd.DataFrame({
        "code": raw["基金代码"].astype(str).str.zfill(6),
        "name": raw.get("基金简称", pd.Series("", index=raw.index)).astype(str),
        "subscription_status": status,
        "redemption_status": raw.get("赎回状态", pd.Series("", index=raw.index)).astype(str).str.strip(),
    })
    if "日累计限定金额" in raw.columns:
        daily_limit_yuan = pd.to_numeric(raw["日累计限定金额"], errors="coerce")
        # 东方财富对“开放申购”常用 1e11 元作“无单日限额”占位，不能当作真实限额。
        meaningful_limit = daily_limit_yuan.notna() & (daily_limit_yuan > 0) & (daily_limit_yuan < 10_000_000_000)
        frame["daily_subscription_limit_yi"] = daily_limit_yuan.where(meaningful_limit) / 1e8
        frame["has_daily_subscription_limit"] = meaningful_limit
    else:
        frame["daily_subscription_limit_yi"] = pd.NA
        frame["has_daily_subscription_limit"] = False
    frame["asset_type"] = "fund"
    return frame.drop_duplicates("code", keep="last").reset_index(drop=True)


def open_subscription_codes_from_frame(status_frame: pd.DataFrame) -> set[str]:
    """从含 subscription_status 的状态表提取可申购代码集合。"""
    if status_frame.empty or "code" not in status_frame:
        return set()
    funds = status_frame.copy()
    funds["code"] = funds["code"].astype(str)
    asset_type = funds.get("asset_type", pd.Series("fund", index=funds.index)).fillna("fund").astype(str).str.lower()
    active = funds[asset_type.ne("etf")].copy()
    if active.empty:
        return set()
    mask = subscription_open_mask(active)
    return set(active.loc[mask, "code"].astype(str))
