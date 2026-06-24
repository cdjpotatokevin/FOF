"""iFinD HTTP API：短期 access token 与专题报表的最小安全适配。

HTTP API 的 refresh token 只从环境变量 ``IFIND_REFRESH_TOKEN`` 读取，access token
只保存在本进程内存。两者都不写入配置、PIT 原始数据或日志。

基金全市场筛选在 iFinD HTTP API 中不是固定端点，而是 ``data_pool`` 专题报表：
用户需先在 iFinD 终端/超级命令中生成 ``reportname``、参数和输出字段。这里负责
执行该报表、将字段映射到本项目的 PIT 主数据合同；不会根据名称猜测 ETF/QDII 分类。
"""
from __future__ import annotations

import os
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests


_BASE_URL = "https://quantapi.51ifind.com/api/v1"
_TRUE_VALUES = {"1", "true", "yes", "y", "是", "有", "股票型"}
_FALSE_VALUES = {"0", "false", "no", "n", "否", "无", "", "nan", "none", "<na>"}
_P04955_REQUIRED_COLUMNS = {
    "jydm", "jydm_mc", "p04955_f002", "p04955_f019", "p04955_f021", "p04955_f023",
    "p04955_f024", "p04955_f025", "p04955_f026",
}
P04955_UNIVERSE_OUTPUT = (
    "jydm:Y,jydm_mc:Y,p04955_f001:Y,p04955_f002:Y,p04955_f019:Y,p04955_f021:Y,p04955_f023:Y,"
    "p04955_f024:Y,p04955_f025:Y,p04955_f026:Y"
)


class IFindHTTPError(RuntimeError):
    """iFinD HTTP API 返回的非成功响应（不包含任何 token）。"""


class IFindHTTPArchive:
    """不含凭据的iFinD请求/响应归档与缓存。

    每个请求由 endpoint + 规范化payload 哈希唯一标识。raw JSON保存供应商原始响应，
    manifest保存请求参数、响应哈希、数据量与获取时间；refresh/access token永不入库。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _canonical(payload: Mapping[str, Any]) -> str:
        return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def _paths(self, endpoint: str, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
        digest = hashlib.sha256(f"{endpoint}\n{self._canonical(payload)}".encode("utf-8")).hexdigest()
        directory = self.root / endpoint.strip("/").replace("/", "_")
        return directory / f"{digest}.json", directory / f"{digest}.manifest.json", digest

    def load(self, endpoint: str, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        raw_path, manifest_path, _ = self._paths(endpoint, payload)
        if not raw_path.exists() or not manifest_path.exists():
            return None
        try:
            return json.loads(raw_path.read_text(encoding="utf-8")), json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def store(self, endpoint: str, payload: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
        raw_path, manifest_path, digest = self._paths(endpoint, payload)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_text = json.dumps(dict(response), ensure_ascii=False, sort_keys=True, default=str)
        raw_path.write_text(raw_text, encoding="utf-8")
        response_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        manifest = {
            "endpoint": endpoint,
            "request_fingerprint": digest,
            "request_payload": json.loads(self._canonical(payload)),
            "raw_response": str(raw_path),
            "response_sha256": response_sha,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "data_vol": response.get("dataVol"),
            "errorcode": response.get("errorcode"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path), "cache_hit": False}


class IFindHTTPClient:
    """iFinD HTTP API 客户端，refresh/access token 均只驻留内存。"""

    def __init__(
        self,
        refresh_token: str | None = None,
        timeout: int = 45,
        session: requests.Session | None = None,
        base_url: str = _BASE_URL,
        cache_dir: str | Path | None = None,
    ):
        self._refresh_token = refresh_token or os.environ.get("IFIND_REFRESH_TOKEN", "")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self._access_token: str | None = None
        archive_root = cache_dir or os.environ.get("IFIND_HTTP_CACHE_DIR")
        self.archive = IFindHTTPArchive(archive_root) if archive_root else None
        if not self._refresh_token and self.archive is None:
            raise IFindHTTPError(
                "未设置 IFIND_REFRESH_TOKEN；请仅在当前终端会话设置该环境变量，勿写入代码或配置文件。"
            )
        self.last_request_metadata: dict[str, Any] = {}

    @staticmethod
    def _decode(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise IFindHTTPError(f"iFinD HTTP {response.status_code} 返回非JSON响应") from exc
        if not isinstance(payload, dict):
            raise IFindHTTPError(f"iFinD HTTP {response.status_code} 响应格式异常")
        if response.status_code != 200 or payload.get("errorcode") not in (0, "0", None):
            code = payload.get("errorcode", response.status_code)
            message = str(payload.get("errmsg", "请求失败"))[:300]
            raise IFindHTTPError(f"iFinD HTTP请求失败（errorcode={code}）：{message}")
        return payload

    def refresh_access_token(self, force: bool = False) -> str:
        """换取并缓存 access token；成功后仅保存在该对象生命周期内。"""
        if self._access_token and not force:
            return self._access_token
        if not self._refresh_token:
            raise IFindHTTPError("缓存未命中且未设置 IFIND_REFRESH_TOKEN，无法向iFinD发起新请求")
        response = self.session.post(
            f"{self.base_url}/get_access_token",
            headers={"Content-Type": "application/json", "refresh_token": self._refresh_token},
            timeout=self.timeout,
        )
        payload = self._decode(response)
        data = payload.get("data", {})
        token = data.get("access_token") if isinstance(data, Mapping) else None
        if not token:
            raise IFindHTTPError("iFinD鉴权成功响应中未找到 access_token")
        self._access_token = str(token)
        return self._access_token

    def post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """使用当前 access token 发送数据请求；token 失效时仅刷新一次并重试。"""
        path = endpoint.strip("/")
        if self.archive:
            cached = self.archive.load(path, payload)
            if cached is not None:
                response, manifest = cached
                self.last_request_metadata = {**manifest, "cache_hit": True}
                return response
        for attempt in range(2):
            token = self.refresh_access_token(force=attempt > 0)
            response = self.session.post(
                f"{self.base_url}/{path}",
                headers={"Content-Type": "application/json", "access_token": token},
                json=dict(payload),
                timeout=self.timeout,
            )
            try:
                decoded = self._decode(response)
                if self.archive:
                    self.last_request_metadata = self.archive.store(path, payload, decoded)
                else:
                    self.last_request_metadata = {
                        "endpoint": path, "data_vol": decoded.get("dataVol"), "cache_hit": False,
                    }
                return decoded
            except IFindHTTPError as exc:
                # 手册中 -1010 是 token 失效；只在这种情形刷新一次，避免隐藏其它数据错误。
                if attempt == 0 and "errorcode=-1010" in str(exc):
                    self._access_token = None
                    continue
                raise
        raise IFindHTTPError("iFinD HTTP请求失败")  # pragma: no cover - 循环必定 return/raise

    def data_pool(
        self,
        reportname: str,
        functionpara: Mapping[str, Any] | None = None,
        outputpara: str | None = None,
    ) -> dict[str, Any]:
        """执行 iFinD 专题报表（``/data_pool``）。"""
        if not reportname.strip():
            raise ValueError("reportname 不能为空")
        payload: dict[str, Any] = {"reportname": reportname}
        if functionpara:
            payload["functionpara"] = dict(functionpara)
        if outputpara:
            payload["outputpara"] = outputpara
        return self.post("data_pool", payload)

    def date_sequence(
        self,
        codes: str,
        indipara: list[Mapping[str, Any]],
        startdate: str,
        enddate: str,
        functionpara: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调用 ``THS_DS`` 对应的日期序列接口。"""
        payload: dict[str, Any] = {
            "codes": codes,
            "indipara": [dict(item) for item in indipara],
            "startdate": startdate,
            "enddate": enddate,
        }
        if functionpara:
            payload["functionpara"] = dict(functionpara)
        return self.post("date_sequence", payload)

    def history_quotation(
        self,
        codes: str,
        indicators: str,
        startdate: str,
        enddate: str,
        functionpara: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调用 ``THS_HQ`` 对应的历史行情接口。"""
        payload: dict[str, Any] = {
            "codes": codes,
            "indicators": indicators,
            "startdate": startdate,
            "enddate": enddate,
        }
        if functionpara:
            payload["functionpara"] = dict(functionpara)
        return self.post("cmd_history_quotation", payload)


def response_to_frame(response: Mapping[str, Any]) -> pd.DataFrame:
    """将 HTTP API 常见的 ``tables`` 返回转换成表；不依赖具体报表字段。"""
    tables = response.get("tables")
    if tables is None and isinstance(response.get("data"), Mapping):
        tables = response["data"].get("tables")
    if isinstance(tables, list):
        if not tables:
            return pd.DataFrame()
        if all(isinstance(row, Mapping) for row in tables):
            # iFinD data_pool 的实际响应常包一层 [{"table": {"字段": [列值...]}}]。
            # 该形式不能直接 DataFrame(tables)，否则会得到一行名为 table 的嵌套对象。
            if len(tables) == 1 and isinstance(tables[0].get("table"), Mapping):
                nested = tables[0]["table"]
                try:
                    return pd.DataFrame(nested)
                except ValueError as exc:
                    raise IFindHTTPError("专题报表嵌套 table 的列长度不一致") from exc
            # 日期序列/历史行情常为每个代码一个 {thscode, table}；展开后保留代码列。
            if all(isinstance(row.get("table"), Mapping) for row in tables):
                frames: list[pd.DataFrame] = []
                for row in tables:
                    try:
                        frame = pd.DataFrame(row["table"])
                    except ValueError as exc:
                        raise IFindHTTPError("行情响应嵌套 table 的列长度不一致") from exc
                    for key, value in row.items():
                        if key != "table" and not isinstance(value, (list, dict)):
                            frame[key] = value
                    frames.append(frame)
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            return pd.DataFrame(tables)
    if isinstance(tables, Mapping):
        for key in ("rows", "data", "table", "records"):
            rows = tables.get(key)
            if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
                return pd.DataFrame(rows)
        # 有些报表以“列名: 列值列表”形式返回，DataFrame 可以直接复原。
        try:
            return pd.DataFrame(tables)
        except ValueError as exc:
            raise IFindHTTPError("专题报表 tables 无法转换为二维表") from exc
    raise IFindHTTPError("专题报表响应中未找到可解析的 tables")


def _as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    # 严格池的默认值为 False；未知值不作乐观推断。
    return False


def map_universe_fields(
    raw: pd.DataFrame,
    field_map: Mapping[str, str],
    *,
    default_asset_type: str | None = None,
    aum_unit: str = "yi",
) -> pd.DataFrame:
    """把 iFinD 报表字段映射成 PIT universe 字段。

    ``field_map`` 示例：
    ``{"code":"p000_f001", "name":"p000_f002", "aum_yi":"p000_f003", ...}``。
    映射后的原始字段仍保留，便于审计。``aum_unit`` 只能是 ``yi``（亿元）、
    ``wan``（万元）或 ``yuan``（元）。缺失的ETF/QDII分类保持为 False，严格池不会放行。
    """
    required = {"code", "name"}
    missing_targets = required - set(field_map)
    missing_sources = [source for source in field_map.values() if source not in raw.columns]
    if missing_targets:
        raise ValueError(f"field_map 缺少必填映射: {sorted(missing_targets)}")
    if missing_sources:
        raise ValueError(f"报表结果缺少字段: {missing_sources}")
    frame = raw.copy()
    for target, source in field_map.items():
        frame[target] = frame[source]
    # iFinD 常以 005827.OF / 510300.SH 返回 thscode，而现有 akshare 数据源和 PIT
    # 主键使用六码代码。只规范这些国内常见后缀，原始 thscode 列继续保留以便审计。
    frame["code"] = (
        frame["code"].astype("string").str.strip()
        .str.replace(r"\.(?:OF|SH|SZ|BJ)$", "", regex=True, case=False)
    )
    if default_asset_type and "asset_type" not in field_map:
        frame["asset_type"] = default_asset_type
    if "aum_yi" in frame:
        scale = {"yi": 1.0, "wan": 1e-4, "yuan": 1e-8}.get(aum_unit.lower())
        if scale is None:
            raise ValueError("aum_unit 只支持 yi、wan 或 yuan")
        frame["aum_yi"] = pd.to_numeric(frame["aum_yi"], errors="coerce") * scale
    for column in ("is_qdii", "is_stock_etf"):
        if column in frame:
            frame[column] = frame[column].map(_as_bool)
    return frame


def p04955_to_universe(raw: pd.DataFrame) -> pd.DataFrame:
    """将“基金业绩回报”(p04955)专题报表转换为基金PIT主数据。

    该报表的 ``p04955_f021`` 是“最新规模(亿份)”而非资产净值；对于人民币公募基金，
    ``亿份 × 最新单位净值`` 得到可用于筛选的近似基金资产规模（亿元）。QDII 行虽会
    同时写入供审计，但标记为 ``is_qdii``，不会进入本策略的严格投资池。

    经投资经理确认的产品分类规则：投资类型为“被动指数型股票基金”、名称含“ETF”且
    不含“联接”的产品属于股票ETF；名称同时含“QDII”和“ETF”的产品为QDII ETF。
    该规则比仅按名称识别ETF更严格，并作为本数据源的可审计分类口径。
    """
    missing = sorted(_P04955_REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise IFindHTTPError(f"p04955报表缺少必要字段: {missing}")
    frame = map_universe_fields(
        raw,
        {
            "code": "jydm",
            "name": "jydm_mc",
            "fund_type": "p04955_f024",
            "inception": "p04955_f023",
            "manager": "p04955_f025",
            "management_company": "p04955_f026",
        },
        default_asset_type="fund",
    )
    # iFinD thscode 可能被上游读取为无前导零的数字；PIT主键统一为六码数字字符串。
    numeric_code = frame["code"].astype("string").str.fullmatch(r"\d{1,6}", na=False)
    if not numeric_code.all():
        raise IFindHTTPError(f"p04955存在 {int((~numeric_code).sum())} 个非六码基金代码")
    frame["code"] = frame["code"].astype("string").str.zfill(6)
    unit_nav = pd.to_numeric(raw["p04955_f002"], errors="coerce")
    shares_yi = pd.to_numeric(raw["p04955_f021"], errors="coerce")
    frame["aum_yi"] = unit_nav * shares_yi
    fund_type = frame["fund_type"].fillna("").astype(str)
    name = frame["name"].fillna("").astype(str)
    domestic_stock_etf = (
        fund_type.eq("被动指数型股票基金")
        & name.str.contains("ETF", case=False, na=False)
        & ~name.str.contains("联接", na=False)
    )
    qdii_etf = name.str.contains("QDII", case=False, na=False) & name.str.contains("ETF", case=False, na=False)
    frame["asset_type"] = (domestic_stock_etf | qdii_etf).map({True: "etf", False: "fund"})
    frame["is_qdii"] = fund_type.str.contains("QDII", case=False, na=False) | qdii_etf
    frame["is_stock_etf"] = domestic_stock_etf
    trade_status = raw["p04955_f019"].fillna("").astype(str)
    frame["subscription_status"] = trade_status.str.extract(r"([^|]*申购[^|]*)", expand=False).fillna("")
    frame["redemption_status"] = trade_status.str.extract(r"([^|]*赎回[^|]*)", expand=False).fillna("")
    # 保留净值、份额与报告字段，便于之后审计 AUM 计算过程。
    return frame


def p04955_pit_frame(raw: pd.DataFrame, asof: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成可写入PIT的p04955主数据，并隔离投资范围外的时间异常行。

    供应商历史报表偶尔会混入截点后成立的非权益产品。此类行绝不能进入PIT；但若
    它们本来就不属于本策略投资范围，不应阻断同一期合规主动权益/股票ETF快照。
    任一投资范围内的未来成立或未来净值记录仍会直接报错，避免掩盖前视风险。
    """
    frame = p04955_to_universe(raw)
    cutoff = pd.Timestamp(asof).normalize()
    inception = pd.to_datetime(frame["inception"], errors="coerce")
    future = inception > cutoff
    if "p04955_f001" in raw:
        nav_date = pd.to_datetime(raw["p04955_f001"], errors="coerce")
        future = future | (nav_date > cutoff).to_numpy()
    quarantined = frame.loc[future].copy()
    if quarantined.empty:
        return frame, quarantined

    # 延迟导入，避免数据层在普通解析路径上加载投资引擎。
    from ..engine.universe import filter_universe
    in_scope = filter_universe(quarantined, asof=cutoff, strict_eligibility=False)
    if not in_scope.empty:
        codes = ", ".join(in_scope["code"].astype(str).head(10))
        raise IFindHTTPError(
            f"p04955历史快照含 {len(in_scope)} 只投资范围内的截点后记录（如 {codes}），拒绝写入PIT"
        )
    return frame.loc[~future].copy(), quarantined


def validate_p04955_asof(raw: pd.DataFrame, asof: str | pd.Timestamp) -> None:
    """验证p04955历史快照没有返回截点之后才存在/才产生的数据。

    这是把供应商历史查询写入PIT前的最小时间一致性闸门：基金成立日与报告中的
    “最新日期”均不得晚于查询截点。它不能证明所有字段的真实披露时点，故调用方仍须
    用保守 ``available_at`` 标注数据可用日。
    """
    cutoff = pd.Timestamp(asof).normalize()
    if "p04955_f023" not in raw:
        raise IFindHTTPError("p04955历史校验缺少成立日期字段")
    inception = pd.to_datetime(raw["p04955_f023"], errors="coerce")
    future_inception = int((inception > cutoff).sum())
    if future_inception:
        raise IFindHTTPError(f"p04955历史快照含 {future_inception} 只截点后成立的基金")
    if "p04955_f001" in raw:
        nav_date = pd.to_datetime(raw["p04955_f001"], errors="coerce")
        future_nav = int((nav_date > cutoff).sum())
        if future_nav:
            raise IFindHTTPError(f"p04955历史快照含 {future_nav} 条截点后的净值日期")
