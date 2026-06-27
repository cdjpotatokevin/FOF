"""交易日前申购状态/限额校验。

PIT 快照只能证明“研究时点可见状态”，不能证明下单当天仍可申购。因此这里把
目标持仓与备选产品逐只映射到交易日状态表，并采用 fail-closed 规则：状态缺失、
限额证据缺失、暂停/限购/限额等都不放行。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


from .subscription import SUBSCRIPTION_BLOCK_RE as _BLOCK_RE
from .subscription import SUBSCRIPTION_OPEN_RE as _OPEN_RE
_TRUE_VALUES = {"1", "true", "yes", "y", "是", "有"}


@dataclass(frozen=True)
class PreTradeSummary:
    total: int
    passed: int
    failed: int
    source: str
    asof: str

    @property
    def ok(self) -> bool:
        return self.failed == 0


def normalize_fund_code(value: object) -> str:
    """基金代码统一为六码数字字符串；交易所后缀仅用于供应商请求，不进入主键。"""
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in _TRUE_VALUES


def _order_codes(paths: Iterable[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"code": "string"})
        if "code" not in frame:
            raise ValueError(f"{path} 缺少 code 列")
        keep = [column for column in ("code", "name", "type", "weight") if column in frame.columns]
        piece = frame.loc[:, keep].copy()
        piece["source_file"] = path
        rows.append(piece)
    if not rows:
        raise ValueError("至少需要一份目标组合或备选清单")
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].map(normalize_fund_code)
    return out.drop_duplicates("code", keep="first").reset_index(drop=True)


def load_order_list(portfolio_csv: str | None = None, backup_csv: str | None = None) -> pd.DataFrame:
    """读取目标组合/备选产品清单，返回去重后的下单前检查列表。"""
    paths = [path for path in (portfolio_csv, backup_csv) if path]
    return _order_codes(paths)


def evaluate_pretrade_status(
    order_list: pd.DataFrame,
    status_frame: pd.DataFrame,
    *,
    asof: str,
    source: str,
    require_limit_evidence: bool = True,
) -> tuple[pd.DataFrame, PreTradeSummary]:
    """逐只产品评估交易日前状态。

    ``status_frame`` 至少应含 ``code``；主动基金建议含 ``subscription_status``，
    并以 ``daily_subscription_limit_yi`` 或 ``has_daily_subscription_limit`` 提供限额证据。
    ``require_limit_evidence=True`` 时，缺少上述限额字段即判定为失败。
    ETF 默认按二级市场交易工具处理，不做申购限额校验；后续应另接停牌/交易状态校验。
    """
    if "code" not in order_list or "code" not in status_frame:
        raise ValueError("order_list 和 status_frame 都必须包含 code 列")
    orders = order_list.copy()
    orders["code"] = orders["code"].map(normalize_fund_code)
    live = status_frame.copy()
    live["code"] = live["code"].map(normalize_fund_code)
    if "ingested_at" in live.columns:
        live = live.sort_values(["code", "ingested_at"])
    live = live.drop_duplicates("code", keep="last").set_index("code", drop=False)

    limit_columns_present = {"daily_subscription_limit_yi", "has_daily_subscription_limit"} & set(live.columns)
    rows: list[dict] = []
    for _, order in orders.iterrows():
        code = str(order["code"])
        base = order.to_dict()
        base.update({"asof": asof, "check_source": source})
        if code not in live.index:
            rows.append({**base, "pretrade_pass": False, "reason": "missing_live_record"})
            continue
        record = live.loc[code]
        if isinstance(record, pd.DataFrame):  # defensive, drop_duplicates should prevent this
            record = record.iloc[-1]
        asset_type = str(record.get("asset_type", order.get("type", ""))).strip().lower()
        is_etf = asset_type == "etf" or str(order.get("type", "")).upper() == "ETF"
        status = str(record.get("subscription_status", "") or "").strip()
        merged = {
            **base,
            "live_name": record.get("name", base.get("name", "")),
            "asset_type": asset_type or record.get("asset_type", ""),
            "subscription_status": status,
            "redemption_status": record.get("redemption_status", ""),
            "daily_subscription_limit_yi": record.get("daily_subscription_limit_yi", pd.NA),
            "has_daily_subscription_limit": record.get("has_daily_subscription_limit", pd.NA),
        }
        if is_etf:
            rows.append({**merged, "pretrade_pass": True, "reason": "etf_secondary_market_not_subscription_checked"})
            continue
        if not status:
            rows.append({**merged, "pretrade_pass": False, "reason": "missing_subscription_status"})
            continue
        if pd.Series([status]).str.contains(_BLOCK_RE, regex=True, na=False).iloc[0]:
            rows.append({**merged, "pretrade_pass": False, "reason": "blocked_subscription_status"})
            continue
        if not pd.Series([status]).str.contains(_OPEN_RE, regex=True, na=False).iloc[0]:
            rows.append({**merged, "pretrade_pass": False, "reason": "not_open_subscription"})
            continue

        has_limit = False
        if "has_daily_subscription_limit" in live.columns:
            has_limit = _bool_value(record.get("has_daily_subscription_limit", ""))
        if "daily_subscription_limit_yi" in live.columns:
            limit_value = pd.to_numeric(record.get("daily_subscription_limit_yi"), errors="coerce")
            has_limit = has_limit or pd.notna(limit_value)
        if has_limit:
            rows.append({**merged, "pretrade_pass": False, "reason": "daily_subscription_limit_present"})
            continue
        if require_limit_evidence and not limit_columns_present:
            rows.append({**merged, "pretrade_pass": False, "reason": "missing_limit_evidence_columns"})
            continue
        if require_limit_evidence and limit_columns_present and all(
            pd.isna(record.get(column, pd.NA)) or str(record.get(column, "")).strip() == ""
            for column in limit_columns_present
        ):
            rows.append({**merged, "pretrade_pass": False, "reason": "missing_limit_evidence_value"})
            continue
        rows.append({**merged, "pretrade_pass": True, "reason": "open_no_limit"})

    report = pd.DataFrame(rows)
    failed = int((~report["pretrade_pass"].astype(bool)).sum()) if not report.empty else 0
    summary = PreTradeSummary(
        total=int(len(report)),
        passed=int(len(report) - failed),
        failed=failed,
        source=source,
        asof=asof,
    )
    return report, summary
