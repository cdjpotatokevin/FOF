"""轻量的 Point-in-Time（PIT）数据仓与数据合同。

回测最常见的错误不是公式，而是把今天才知道的基金状态、经理、规模或分类带回历史。
本模块为基金/ETF主数据和行情保留两条时间轴：

``effective_date``
    数据描述的经济时点，例如基金暂停申购的生效日、ETF 的交易日。
``available_at``
    当时研究系统可合法使用该记录的日期，例如公告披露日或收盘后的下一交易日。

读取 ``asof`` 时只返回 ``effective_date <= asof`` 且 ``available_at <= asof`` 的记录。
CSV 是刻意的依赖最小化选择；每次写入均落盘为不可覆盖的快照和同名 manifest，后续可以
无损迁移到 Parquet/Iceberg，而不会改变上层 PIT 语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .base import DataProvider, ETF_MARKET_COLUMNS


UNIVERSE_COLUMNS = [
    "code", "name", "asset_type", "fund_type", "status", "is_qdii", "is_stock_etf",
    "aum_yi", "manager", "manager_start", "inception", "subscription_status", "redemption_status",
    "management_company",
    "effective_date", "available_at", "source", "source_asof", "ingested_at",
]
PIT_MARKET_COLUMNS = [
    "trading_date", "code", "close", "volume", "amount", "turnover_rate",
    "premium_discount", "available_at", "source", "ingested_at",
]
HOLDINGS_COLUMNS = [
    "fund_code", "security_code", "security_name", "weight", "market_value", "industry",
    "report_period", "available_at", "source", "ingested_at",
]


class PITDataError(ValueError):
    """PIT数据不满足可回测时间语义或字段合同。"""


def _date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise PITDataError(f"{label} 缺少必填字段: {missing}")


@dataclass(frozen=True)
class DatasetWrite:
    path: Path
    manifest_path: Path
    rows: int
    sha256: str


@dataclass(frozen=True)
class DataQualityReport:
    """入库前的轻量质量检查结果；严重问题阻止数据进入PIT仓。"""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, list[str]]:
        return {"errors": list(self.errors), "warnings": list(self.warnings)}

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise PITDataError("数据质量检查失败：" + "；".join(self.errors))


def validate_market_data(frame: pd.DataFrame) -> DataQualityReport:
    """检查ETF日行情的主键、价格和可用于流动性研究的字段。"""
    errors: list[str] = []
    warnings: list[str] = []
    if frame[["trading_date", "code", "close"]].isna().any().any():
        errors.append("trading_date/code/close 不得为空")
    if (pd.to_numeric(frame["close"], errors="coerce") <= 0).any():
        errors.append("close 必须为正数")
    if frame.duplicated(["code", "trading_date"]).any():
        errors.append("同一 code/trading_date 存在重复行情")
    if "amount" not in frame or pd.to_numeric(frame.get("amount"), errors="coerce").notna().sum() == 0:
        warnings.append("缺少成交额，无法进行成交冲击/容量研究")
    if "volume" not in frame or pd.to_numeric(frame.get("volume"), errors="coerce").notna().sum() == 0:
        warnings.append("缺少成交量，无法进行流动性复核")
    return DataQualityReport(tuple(errors), tuple(warnings))


class PITDataStore:
    """本地 PIT 快照仓。

    ``root`` 应位于受备份与版本管理的数据盘，而非代码目录。默认不为历史补写
    ``available_at``：调用方必须明确提供原始快照日期或接受本模块的保守可用性滞后。
    """

    schema_version = "1"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_snapshot(self, dataset: str, source: str, available_at: pd.Timestamp,
                        frame: pd.DataFrame, metadata: dict | None = None) -> DatasetWrite:
        part = self.root / dataset / f"source={_safe_part(source)}" / f"available_at={available_at.date()}"
        part.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = part / f"snapshot-{stamp}.csv"
        temp = path.with_suffix(".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = {
            "schema_version": self.schema_version,
            "dataset": dataset,
            "source": source,
            "available_at": str(available_at.date()),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": digest,
            **(metadata or {}),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return DatasetWrite(path=path, manifest_path=manifest_path, rows=len(frame), sha256=digest)

    def _read_dataset(self, dataset: str) -> pd.DataFrame:
        base = self.root / dataset
        files = sorted(base.rglob("*.csv")) if base.exists() else []
        if not files:
            return pd.DataFrame()
        # 基金/ETF代码必须按字符串读取；例如 000001 被推断成整数会破坏主键与PIT join。
        return pd.concat([
            pd.read_csv(path, dtype={"code": "string", "fund_code": "string", "security_code": "string"})
            for path in files
        ], ignore_index=True)

    def write_universe_snapshot(
        self,
        records: pd.DataFrame,
        source: str,
        available_at: str | pd.Timestamp,
        effective_date: str | pd.Timestamp | None = None,
        source_asof: str | pd.Timestamp | None = None,
        source_metadata: dict | None = None,
    ) -> DatasetWrite:
        """写入一份基金/ETF主数据快照。

        ``available_at`` 需使用这份清单实际可见的日期；不要把今天抓取的全市场清单
        标记成多年前可见。未知分类保留为空/False，不能靠名称推断为非QDII股票ETF。
        """
        _ensure_columns(records, ("code", "name"), "universe snapshot")
        available = _date(available_at)
        effective = _date(effective_date or available)
        source_asof_ts = _date(source_asof or effective)
        frame = records.copy()
        if frame["code"].isna().any():
            raise PITDataError("universe snapshot 的 code 不得为空")
        frame["code"] = frame["code"].astype(str).str.strip()
        if frame["code"].isin(("", "<NA>", "nan", "None")).any() or frame["code"].duplicated().any():
            raise PITDataError("universe snapshot 的 code 不能为空且同一快照内必须唯一")
        defaults = {
            "asset_type": "fund", "fund_type": "", "status": "active",
            "is_qdii": False, "is_stock_etf": False,
        }
        for column, default in defaults.items():
            if column not in frame:
                frame[column] = default
            frame[column] = frame[column].where(frame[column].notna(), default)
        frame["effective_date"] = effective.date().isoformat()
        frame["available_at"] = available.date().isoformat()
        frame["source"] = source
        frame["source_asof"] = source_asof_ts.date().isoformat()
        frame["ingested_at"] = datetime.now(timezone.utc).isoformat()
        # 保留供应商额外字段（例如基金经理变更公告编号、ETF份额、分类置信度），避免
        # 数据仓为了“规范”而丢失后续审计会需要追溯的原始信息。
        columns = UNIVERSE_COLUMNS + [column for column in frame.columns if column not in UNIVERSE_COLUMNS]
        return self._write_snapshot(
            "universe", source, available, frame.reindex(columns=columns),
            metadata={
                "effective_date": str(effective.date()), "source_asof": str(source_asof_ts.date()),
                **({"source_request": source_metadata} if source_metadata else {}),
            },
        )

    def read_universe_asof(
        self,
        asof: str | pd.Timestamp,
        asset_types: Iterable[str] | None = None,
        active_only: bool = True,
    ) -> pd.DataFrame:
        """读取 ``asof`` 当时可获得的最新逐代码主数据状态。"""
        frame = self._read_dataset("universe")
        if frame.empty:
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        _ensure_columns(frame, ("code", "effective_date", "available_at"), "stored universe")
        point = _date(asof)
        for column in ("effective_date", "available_at", "ingested_at"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        eligible = frame[
            (frame["effective_date"] <= point) & (frame["available_at"] <= point)
        ].dropna(subset=["code", "effective_date", "available_at"])
        if eligible.empty:
            return pd.DataFrame(columns=UNIVERSE_COLUMNS)
        eligible = eligible.sort_values(["code", "effective_date", "available_at", "ingested_at"])
        latest = eligible.groupby("code", as_index=False).tail(1)
        if asset_types is not None:
            latest = latest[latest["asset_type"].isin(set(asset_types))]
        if active_only:
            latest = latest[latest["status"].eq("active")]
        return latest.sort_values("code").reset_index(drop=True)

    def write_market_data(
        self,
        market: pd.DataFrame,
        source: str,
        available_at: str | pd.Timestamp | None = None,
        availability_lag_days: int = 1,
        source_metadata: dict | None = None,
    ) -> DatasetWrite:
        """写入日行情，默认以交易日后一天作为可用于收盘后研究的保守可用时点。"""
        _ensure_columns(market, ("trading_date", "code", "close"), "market data")
        frame = market.copy()
        frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="coerce").dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["trading_date", "code", "close"])
        if frame.empty:
            raise PITDataError("market data 没有有效行情行")
        if available_at is None:
            frame["available_at"] = frame["trading_date"] + pd.Timedelta(days=availability_lag_days)
        else:
            frame["available_at"] = _date(available_at)
        for column in ("volume", "amount", "turnover_rate", "premium_discount"):
            if column not in frame:
                frame[column] = float("nan")
        frame["code"] = frame["code"].astype(str).str.strip()
        if frame["code"].isin(("", "<NA>", "nan", "None")).any():
            raise PITDataError("market data 的 code 不得为空")
        frame["source"] = source
        frame["ingested_at"] = datetime.now(timezone.utc).isoformat()
        quality = validate_market_data(frame)
        quality.raise_if_invalid()
        snapshot_available = pd.to_datetime(frame["available_at"]).max()
        return self._write_snapshot(
            "market", source, snapshot_available,
            frame.reindex(columns=PIT_MARKET_COLUMNS),
            metadata={
                "availability_lag_days": availability_lag_days, "quality": quality.as_dict(),
                **({"source_request": source_metadata} if source_metadata else {}),
            },
        )

    def read_market_asof(
        self,
        code: str,
        asof: str | pd.Timestamp,
        start: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """读取指定证券在 ``asof`` 时点合法可见的行情历史。"""
        frame = self._read_dataset("market")
        if frame.empty:
            return pd.DataFrame(columns=PIT_MARKET_COLUMNS)
        _ensure_columns(frame, ("code", "trading_date", "available_at"), "stored market")
        point = _date(asof)
        frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="coerce")
        frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
        eligible = frame[(frame["code"].astype(str) == str(code)) & (frame["available_at"] <= point)]
        if start is not None:
            eligible = eligible[eligible["trading_date"] >= _date(start)]
        eligible = eligible.sort_values(["trading_date", "available_at", "ingested_at"])
        return eligible.drop_duplicates("trading_date", keep="last").reset_index(drop=True)

    def write_holdings_snapshot(
        self,
        holdings: pd.DataFrame,
        source: str,
        report_period: str | pd.Timestamp,
        available_at: str | pd.Timestamp,
    ) -> DatasetWrite:
        """写入一季基金持仓。

        ``report_period`` 是持仓所属报告期末，``available_at`` 是报告实际披露日；两者
        不可混淆。读取时总是以披露日约束，避免季度末尚未公开的持仓进入回测。
        """
        _ensure_columns(holdings, ("fund_code", "security_code"), "holdings snapshot")
        if "weight" not in holdings and "market_value" not in holdings:
            raise PITDataError("holdings snapshot 至少需要 weight 或 market_value")
        frame = holdings.copy()
        for column in ("fund_code", "security_code"):
            if frame[column].isna().any():
                raise PITDataError(f"{column} 不得为空")
            frame[column] = frame[column].astype(str).str.strip()
        if frame[["fund_code", "security_code"]].isin(("", "<NA>", "nan", "None")).any().any():
            raise PITDataError("fund_code/security_code 不得为空")
        if "weight" not in frame:
            frame["weight"] = float("nan")
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        if (frame["weight"].dropna() < 0).any():
            raise PITDataError("holding weight 不得为负")
        for column in ("security_name", "market_value", "industry"):
            if column not in frame:
                frame[column] = float("nan") if column == "market_value" else ""
        available = _date(available_at)
        period = _date(report_period)
        frame["report_period"] = period.date().isoformat()
        frame["available_at"] = available.date().isoformat()
        frame["source"] = source
        frame["ingested_at"] = datetime.now(timezone.utc).isoformat()
        weight_sums = frame.groupby("fund_code")["weight"].sum(min_count=1)
        warnings = [f"{code} 持仓权重和 {total:.1%}>105%，请核对口径" for code, total in weight_sums.dropna().items()
                    if total > 1.05]
        columns = HOLDINGS_COLUMNS + [column for column in frame.columns if column not in HOLDINGS_COLUMNS]
        return self._write_snapshot(
            "holdings", source, available, frame.reindex(columns=columns),
            metadata={"report_period": str(period.date()), "quality": {"errors": [], "warnings": warnings}},
        )

    def read_holdings_asof(
        self,
        asof: str | pd.Timestamp,
        fund_code: str | None = None,
    ) -> pd.DataFrame:
        """读取每只基金在 ``asof`` 时点已披露的最新报告期持仓。"""
        frame = self._read_dataset("holdings")
        if frame.empty:
            return pd.DataFrame(columns=HOLDINGS_COLUMNS)
        _ensure_columns(frame, ("fund_code", "security_code", "report_period", "available_at"), "stored holdings")
        point = _date(asof)
        for column in ("report_period", "available_at", "ingested_at"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        eligible = frame[(frame["report_period"] <= point) & (frame["available_at"] <= point)]
        if fund_code is not None:
            eligible = eligible[eligible["fund_code"].astype(str) == str(fund_code)]
        if eligible.empty:
            return pd.DataFrame(columns=HOLDINGS_COLUMNS)
        latest_period = eligible.groupby("fund_code")["report_period"].transform("max")
        latest = eligible[eligible["report_period"] == latest_period]
        latest = latest.sort_values(["fund_code", "security_code", "available_at", "ingested_at"])
        return latest.drop_duplicates(["fund_code", "security_code"], keep="last").reset_index(drop=True)

    def snapshot_provider_universe(
        self,
        provider: DataProvider,
        source: str,
        available_at: str | pd.Timestamp,
    ) -> DatasetWrite:
        """抓取当前 provider 的基金与ETF清单并落为一份有日期的快照。"""
        funds = provider.list_funds().copy()
        _ensure_columns(funds, ("code", "name"), "provider fund list")
        funds["asset_type"] = "fund"
        try:
            etfs = provider.list_etfs().copy()
        except NotImplementedError:
            etfs = pd.DataFrame(columns=["code", "name", "asset_type", "fund_type"])
        if not etfs.empty:
            _ensure_columns(etfs, ("code", "name"), "provider ETF list")
            if "asset_type" not in etfs:
                etfs["asset_type"] = "etf"
            else:
                etfs["asset_type"] = etfs["asset_type"].fillna("etf")
        universe = pd.concat([funds, etfs], ignore_index=True, sort=False)
        # 若同代码同时出现，以交易所 ETF 的资产类型优先，避免它被误识别为开放式基金。
        universe = universe.drop_duplicates("code", keep="last")
        return self.write_universe_snapshot(universe, source=source, available_at=available_at)


def provider_etf_market(provider: DataProvider, code: str, start: str, end: str) -> pd.DataFrame:
    """将 provider ETF 行情标准化为 PIT 入库前的市场合同。"""
    market = provider.get_etf_market(code, start, end).copy()
    _ensure_columns(market, ("trading_date", "code", "close"), "provider ETF market")
    return market.reindex(columns=[column for column in ETF_MARKET_COLUMNS if column in market.columns])
