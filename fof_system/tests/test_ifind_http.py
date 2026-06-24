"""iFinD HTTP适配器的离线合同测试：不发起真实网络请求，也不需要任何凭据。"""
from __future__ import annotations

import pandas as pd
import pytest

from fof_system.data.ifind_http import (
    IFindHTTPClient,
    IFindHTTPError,
    map_universe_fields,
    p04955_pit_frame,
    p04955_to_universe,
    response_to_frame,
    validate_p04955_asof,
)


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("get_access_token"):
            return _Response({"errorcode": 0, "errmsg": "success", "data": {"access_token": "memory-only"}})
        return _Response({"errorcode": 0, "tables": [{"code": "000001", "name": "测试基金"}]})


def test_http_client_refreshes_in_memory_and_sends_data_pool_payload():
    session = _Session()
    client = IFindHTTPClient(refresh_token="test-only", session=session, base_url="https://example.test/api")
    result = client.data_pool("p00001", {"date": "2024-12-31"}, "f001:Y")
    assert result["errorcode"] == 0
    assert len(session.calls) == 2
    assert session.calls[0][1]["headers"]["refresh_token"] == "test-only"
    assert session.calls[1][1]["json"] == {"reportname": "p00001", "functionpara": {"date": "2024-12-31"}, "outputpara": "f001:Y"}
    assert "access_token" in session.calls[1][1]["headers"]


def test_http_client_archives_raw_response_and_reuses_same_request(tmp_path):
    session = _Session()
    client = IFindHTTPClient(
        refresh_token="test-only", session=session, base_url="https://example.test/api", cache_dir=tmp_path,
    )
    first = client.data_pool("p00001", {"date": "2024-12-31"}, "f001:Y")
    calls_after_first = len(session.calls)
    second = client.data_pool("p00001", {"date": "2024-12-31"}, "f001:Y")
    manifests = list(tmp_path.rglob("*.manifest.json"))
    assert first == second
    assert len(session.calls) == calls_after_first
    assert client.last_request_metadata["cache_hit"] is True
    assert len(manifests) == 1
    manifest_text = manifests[0].read_text(encoding="utf-8")
    assert "refresh_token" not in manifest_text


def test_http_client_reads_existing_archive_without_credentials(tmp_path, monkeypatch):
    seeded = IFindHTTPClient(
        refresh_token="test-only", session=_Session(), base_url="https://example.test/api", cache_dir=tmp_path,
    )
    seeded.data_pool("p00001", {"date": "2024-12-31"}, "f001:Y")
    monkeypatch.delenv("IFIND_REFRESH_TOKEN", raising=False)
    offline = IFindHTTPClient(session=_Session(), base_url="https://example.test/api", cache_dir=tmp_path)
    response = offline.data_pool("p00001", {"date": "2024-12-31"}, "f001:Y")
    assert response["errorcode"] == 0
    assert offline.last_request_metadata["cache_hit"] is True


def test_response_mapping_preserves_raw_fields_converts_units_and_chinese_bools():
    raw = response_to_frame({"tables": [{
        "f_code": "510300.SH", "f_name": "沪深300ETF", "f_aum": "50000",
        "f_stock": "是", "f_qdii": "否", "f_type": "ETF",
    }]})
    mapped = map_universe_fields(
        raw,
        {"code": "f_code", "name": "f_name", "aum_yi": "f_aum", "is_stock_etf": "f_stock",
         "is_qdii": "f_qdii", "fund_type": "f_type"},
        default_asset_type="etf", aum_unit="wan",
    )
    assert mapped.loc[0, "code"] == "510300"
    assert mapped.loc[0, "aum_yi"] == 5.0
    assert bool(mapped.loc[0, "is_stock_etf"]) is True
    assert bool(mapped.loc[0, "is_qdii"]) is False
    assert mapped.loc[0, "f_name"] == "沪深300ETF"


def test_response_to_frame_parses_real_data_pool_nested_columnar_table():
    response = {"tables": [{"table": {"jydm": ["000001.OF", "000002.OF"], "jydm_mc": ["甲", "乙"]}}]}
    frame = response_to_frame(response)
    assert list(frame.columns) == ["jydm", "jydm_mc"]
    assert frame.loc[0, "jydm"] == "000001.OF"


def test_response_to_frame_parses_multiple_market_tables_with_thscode():
    response = {"tables": [
        {"thscode": "399370.SZ", "table": {"time": ["2026-06-23"], "value": [12.3]}},
        {"thscode": "399371.SZ", "table": {"time": ["2026-06-23"], "value": [8.7]}},
    ]}
    frame = response_to_frame(response)
    assert list(frame["thscode"]) == ["399370.SZ", "399371.SZ"]
    assert list(frame["value"]) == [12.3, 8.7]


def test_p04955_maps_aum_codes_and_user_approved_etf_rules():
    raw = pd.DataFrame({
        "jydm": ["1234.OF", "512000.OF", "159001.OF", "012345.OF"],
        "jydm_mc": ["主动权益", "券商ETF", "纳指ETF(QDII)", "券商ETF联接"],
        "p04955_f002": ["1.5", "1.0", "0.2", "1.0"], "p04955_f021": ["2.0", "10.0", "10.0", "3.0"],
        "p04955_f023": ["2020/01/01"] * 4,
        "p04955_f024": ["偏股混合型基金", "被动指数型股票基金", "QDII股票型基金", "被动指数型股票基金"],
        "p04955_f025": ["经理甲"] * 4, "p04955_f026": ["公司甲"] * 4,
        "p04955_f019": ["开放申购|开放赎回", "开放申购|开放赎回", "暂停申购|开放赎回", "开放申购|开放赎回"],
    })
    mapped = p04955_to_universe(raw)
    assert mapped.loc[0, "code"] == "001234"
    assert mapped.loc[0, "aum_yi"] == 3.0
    assert mapped.loc[1, "asset_type"] == "etf" and bool(mapped.loc[1, "is_stock_etf"]) is True
    assert mapped.loc[2, "asset_type"] == "etf" and bool(mapped.loc[2, "is_qdii"]) is True
    assert mapped.loc[3, "asset_type"] == "fund" and bool(mapped.loc[3, "is_stock_etf"]) is False
    assert mapped.loc[2, "subscription_status"] == "暂停申购"
    assert mapped.loc[0, "p04955_f021"] == "2.0"


def test_p04955_historical_validation_rejects_future_inception_or_nav_date():
    valid = pd.DataFrame({"p04955_f023": ["2020-01-01"], "p04955_f001": ["2024-12-31"]})
    validate_p04955_asof(valid, "2024-12-31")
    with pytest.raises(IFindHTTPError, match="后成立"):
        validate_p04955_asof(valid.assign(p04955_f023="2025-01-01"), "2024-12-31")
    with pytest.raises(IFindHTTPError, match="后的净值日期"):
        validate_p04955_asof(valid.assign(p04955_f001="2025-01-01"), "2024-12-31")


def test_p04955_pit_frame_quarantines_out_of_scope_future_record_but_rejects_in_scope():
    raw = pd.DataFrame({
        "jydm": ["000001.OF", "000002.OF"], "jydm_mc": ["权益基金", "短债基金D"],
        "p04955_f002": ["1.0", "1.0"], "p04955_f021": ["3.0", "3.0"],
        "p04955_f023": ["2020-01-01", "2025-01-01"],
        "p04955_f024": ["偏股混合型基金", "短期纯债券型基金"],
        "p04955_f025": ["经理", "经理"], "p04955_f026": ["公司", "公司"],
        "p04955_f019": ["开放申购|开放赎回", "开放申购|开放赎回"],
    })
    frame, quarantined = p04955_pit_frame(raw, "2024-12-31")
    assert frame["code"].tolist() == ["000001"]
    assert quarantined["code"].tolist() == ["000002"]

    with pytest.raises(IFindHTTPError, match="投资范围内"):
        p04955_pit_frame(raw.assign(p04955_f024="偏股混合型基金"), "2024-12-31")


def test_mapping_rejects_missing_required_or_source_fields():
    raw = pd.DataFrame({"f_name": ["基金"]})
    with pytest.raises(ValueError, match="必填映射"):
        map_universe_fields(raw, {"name": "f_name"})
    with pytest.raises(ValueError, match="缺少字段"):
        map_universe_fields(raw, {"code": "f_code", "name": "f_name"})


def test_http_client_requires_transient_refresh_token(monkeypatch):
    monkeypatch.delenv("IFIND_REFRESH_TOKEN", raising=False)
    with pytest.raises(IFindHTTPError, match="IFIND_REFRESH_TOKEN"):
        IFindHTTPClient()
