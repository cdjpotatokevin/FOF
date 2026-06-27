"""候选基金池筛选：把全市场清单收敛到"主动权益 + 股票型ETF/指增"。"""
from __future__ import annotations
import pandas as pd
from ..config import UniverseFilter, DEFAULT_FILTER
from .subscription import subscription_open_mask as _subscription_open


def management_company_allowed(company: object, whitelist: list[str]) -> bool:
    """判断 PIT ``management_company`` 是否落在允许的管理人白名单内。"""
    text = str(company or "").strip()
    if not text or not whitelist:
        return not bool(whitelist)
    for allowed in sorted(whitelist, key=len, reverse=True):
        if allowed.endswith("证券"):
            stem = allowed.removesuffix("证券")
            if text.startswith(stem) and "证券" in text:
                return True
            continue
        stem = allowed.removesuffix("基金")
        if not text.startswith(stem):
            continue
        if "基金" not in text or "管理" not in text:
            continue
        if "证券资产" in text:
            continue
        return True
    return False


def _management_company_mask(companies: pd.Series, whitelist: list[str]) -> pd.Series:
    if not whitelist:
        return pd.Series(True, index=companies.index)
    return companies.fillna("").astype(str).apply(lambda c: management_company_allowed(c, whitelist))


def _bool_col(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index)
    return df[column].astype(str).str.lower().isin(("1", "true", "yes"))


def filter_universe(
    fund_list: pd.DataFrame,
    flt: UniverseFilter = DEFAULT_FILTER,
    asof: str | pd.Timestamp | None = None,
    strict_eligibility: bool = False,
) -> pd.DataFrame:
    """按投资范围筛选基金池。

    ``strict_eligibility`` 仅用于有PIT元数据的场景：主动权益需成立满一年且 AUM≥2亿，
    股票ETF需非QDII、AUM≥5亿。缺少这些字段时不猜测、不放行。
    """
    df = fund_list.copy()
    ftype = df["fund_type"].fillna("").astype(str)
    name = df["name"].fillna("").astype(str)

    type_ok = ftype.apply(lambda t: any(k in t for k in flt.allowed_type_keywords))
    asset_type = df.get("asset_type", pd.Series("fund", index=df.index)).fillna("fund").astype(str).str.lower()
    is_etf = asset_type.eq("etf") | ftype.str.contains("ETF", case=False, na=False)
    is_stock_etf = _bool_col(df, "is_stock_etf")
    is_qdii = _bool_col(df, "is_qdii")
    passive_fund_type = ftype.apply(
        lambda t: any(k in t for k in getattr(flt, "passive_fund_type_keywords", []))
    )
    # ETF必须有明确股票型标签；不能仅因名称里有ETF而推断其在投资范围内。
    # 非PIT探索模式保留非QDII ETF，方便旧数据源的初筛；严格回测/实盘模式则要求
    # 明确的 is_stock_etf 标签，缺失标签一律不放行。
    etf_in_scope = is_etf & ~is_qdii & (is_stock_etf | (not strict_eligibility))
    in_scope = ((~is_etf) & type_ok & ~passive_fund_type) | etf_in_scope
    df = df[in_scope]

    if flt.exclude_keywords:
        # QDII 等范围标签常在 fund_type 而非名称中，两个字段都要检查。
        scope_text = name.loc[df.index] + " " + ftype.loc[df.index]
        excl = scope_text.apply(lambda text: any(k in text for k in flt.exclude_keywords))
    else:
        excl = pd.Series(False, index=df.index)
    suffixes = tuple(flt.exclude_share_class_suffixes)
    share_class = name.loc[df.index].str.strip().str.endswith(suffixes) if suffixes else False
    df = df[~(excl | share_class)]

    if strict_eligibility:
        required = {"aum_yi", "inception"}
        missing = required - set(df.columns)
        if missing:
            return df.iloc[0:0].copy()
        aum = pd.to_numeric(df["aum_yi"], errors="coerce")
        inception = pd.to_datetime(df["inception"], errors="coerce")
        point = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
        minimum_inception = point - pd.Timedelta(days=365.25 * flt.min_track_record_years)
        etf = is_etf.loc[df.index]
        active_fund_ok = (~etf) & (aum >= flt.min_active_equity_size_yi) & (inception <= minimum_inception)
        if flt.require_open_subscription:
            active_fund_ok = active_fund_ok & _subscription_open(df)
        if flt.require_manager_tenure and flt.min_manager_tenure_years > 0:
            if "manager_start" not in df.columns:
                active_fund_ok = active_fund_ok & False
            else:
                manager_start = pd.to_datetime(df["manager_start"], errors="coerce")
                minimum_manager = point - pd.Timedelta(days=365.25 * flt.min_manager_tenure_years)
                active_fund_ok = active_fund_ok & manager_start.notna() & (manager_start <= minimum_manager)
        stock_etf_ok = etf & (aum >= flt.min_stock_etf_size_yi)
        if flt.allowed_management_companies:
            if "management_company" not in df.columns:
                return df.iloc[0:0].copy()
            company_ok = _management_company_mask(df["management_company"], flt.allowed_management_companies)
            active_fund_ok = active_fund_ok & company_ok
            stock_etf_ok = stock_etf_ok & company_ok
        df = df[active_fund_ok | stock_etf_ok]

    return df.reset_index(drop=True)
