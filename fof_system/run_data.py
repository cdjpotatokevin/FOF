#!/usr/bin/env python3
"""命令行入口：第①层PIT数据快照与ETF行情入库。

示例：
  # 记录今天可见的基金/ETF清单；这份快照只能用于今天及之后的研究时点
  python -m fof_system.run_data snapshot-universe --source akshare --root /data/fof_pit

  # 将实际股票ETF行情写入带 available_at 的PIT市场数据集
  python -m fof_system.run_data ingest-etf --source akshare --root /data/fof_pit \\
      --codes 510300,159915 --start 2018-01-01 --end 2026-06-23

  # 导入供应商留存的历史基金池/经理/AUM快照（available_at 必须是当时真实披露日）
  python -m fof_system.run_data import-universe --root /data/fof_pit --csv 2023Q4_universe.csv \\
      --source ifind --effective-date 2023-12-31 --available-at 2024-01-25

  # 导入季报持仓；报告期与披露日必须分别填写
  python -m fof_system.run_data import-holdings --root /data/fof_pit --csv 2023Q4_holdings.csv \\
      --source ifind --report-period 2023-12-31 --available-at 2024-01-20
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .data import PITDataStore, get_provider
from .data.ifind_http import (
    IFindHTTPClient, P04955_UNIVERSE_OUTPUT, map_universe_fields, p04955_to_universe,
    p04955_pit_frame, response_to_frame,
)
from .data.pit import provider_etf_market


def _codes(value: str) -> list[str]:
    codes = [code.strip() for code in value.split(",") if code.strip()]
    if not codes:
        raise argparse.ArgumentTypeError("请至少提供一个代码")
    return codes


def _p04955_frame_with_quarantine(raw: pd.DataFrame, asof: str | pd.Timestamp,
                                  store: PITDataStore) -> tuple[pd.DataFrame, dict]:
    frame, quarantined = p04955_pit_frame(raw, asof)
    if quarantined.empty:
        return frame, {}
    directory = store.root / "quarantine" / "p04955_time_anomaly"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"asof={pd.Timestamp(asof).strftime('%Y-%m-%d')}.csv"
    quarantined.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"⚠ 已隔离 {len(quarantined)} 条投资范围外的时间异常记录 → {path}")
    return frame, {"quarantined_rows": int(len(quarantined)), "quarantine_path": str(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF 第①层：PIT基金/ETF数据管理")
    parser.add_argument("--root", default="fof_pit_data", help="PIT数据仓目录（建议放独立数据盘）")
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot-universe", help="记录当前基金/ETF主数据快照")
    snapshot.add_argument("--source", default="akshare", choices=["akshare", "mock"])
    snapshot.add_argument("--asof", default="", help="该快照实际可获得日期，默认今天")

    imported = sub.add_parser("import-universe", help="导入供应商/自有留存的历史PIT主数据CSV")
    imported.add_argument("--csv", required=True, help="至少含 code,name 的UTF-8 CSV")
    imported.add_argument("--source", required=True, help="供应商或内部数据集标识")
    imported.add_argument("--available-at", required=True, help="该记录实际可被研究使用的日期")
    imported.add_argument("--effective-date", default="", help="状态生效日期，默认 available_at")
    imported.add_argument("--source-asof", default="", help="供应商快照标注的业务日期")

    http_pool = sub.add_parser("fetch-ifind-data-pool", help="执行iFinD专题报表并写入PIT基金池")
    http_pool.add_argument("--reportname", required=True, help="iFinD超级命令生成的专题报表编号")
    http_pool.add_argument("--functionpara", default="{}", help="报表参数JSON，默认 {}")
    http_pool.add_argument("--outputpara", default="", help="输出字段参数，例如 p000_f001:Y,p000_f002:Y")
    http_pool.add_argument("--field-map", required=True,
                           help='JSON字段映射，如 {"code":"p000_f001","name":"p000_f002"}')
    http_pool.add_argument("--asset-type", default="", choices=["", "fund", "etf"],
                           help="报表单一资产类型时可指定；混合报表请在 field-map 中映射 asset_type")
    http_pool.add_argument("--aum-unit", default="yi", choices=["yi", "wan", "yuan"],
                           help="报表规模字段单位：亿元/万元/元")
    http_pool.add_argument("--source", default="ifind-http")
    http_pool.add_argument("--available-at", required=True, help="报表实际可用日期（PIT约束）")
    http_pool.add_argument("--effective-date", default="", help="状态生效日期，默认 available_at")
    http_pool.add_argument("--source-asof", default="", help="iFinD报表业务日期")

    performance = sub.add_parser("fetch-ifind-p04955", help="将iFinD基金业绩回报报表写入PIT主动基金主数据")
    performance.add_argument("--edate", required=True, help="报表截止日期，格式 YYYYMMDD")
    performance.add_argument("--p0", default="0", help="iFinD报表投资类型参数，默认 0")
    performance.add_argument("--jjlb", required=True, help="iFinD报表板块成分参数")
    performance.add_argument("--user-sectorid", required=True, help="iFinD报表板块ID，例如 username|")
    performance.add_argument("--source", default="ifind-http:p04955")
    performance.add_argument("--available-at", required=True, help="报表实际可用日期（PIT约束）")
    performance.add_argument("--effective-date", default="", help="状态生效日期，默认 available_at")
    performance.add_argument("--source-asof", default="", help="iFinD报表业务日期")

    backfill = sub.add_parser("backfill-ifind-p04955", help="按季度/月度回补经时间校验的iFinD p04955历史PIT快照")
    backfill.add_argument("--start", required=True, help="回补开始日期 YYYY-MM-DD")
    backfill.add_argument("--end", required=True, help="回补结束日期 YYYY-MM-DD")
    backfill.add_argument("--frequency", default="QE", choices=["QE", "ME"],
                          help="快照频率：QE季度末（默认）或ME月末")
    backfill.add_argument("--p0", default="0")
    backfill.add_argument("--jjlb", required=True)
    backfill.add_argument("--user-sectorid", required=True)
    backfill.add_argument("--availability-lag-business-days", type=int, default=2,
                          help="相对报表截点的保守可用滞后工作日，默认2")
    backfill.add_argument("--source", default="ifind-http:p04955")

    audit = sub.add_parser("ifind-cache-audit", help="汇总iFinD原始响应缓存与供应商数据量")

    holdings = sub.add_parser("import-holdings", help="导入基金季报持仓CSV（按披露日PIT化）")
    holdings.add_argument("--csv", required=True, help="至少含 fund_code,security_code,weight 或 market_value 的CSV")
    holdings.add_argument("--source", required=True)
    holdings.add_argument("--report-period", required=True, help="持仓所属报告期末")
    holdings.add_argument("--available-at", required=True, help="报告实际披露/可用日期")

    etf = sub.add_parser("ingest-etf", help="写入真实ETF日行情")
    etf.add_argument("--source", default="akshare", choices=["akshare", "ifind_http"])
    etf.add_argument("--codes", type=_codes, required=True, help="ETF代码，逗号分隔")
    etf.add_argument("--start", required=True)
    etf.add_argument("--end", default="")
    etf.add_argument("--available-at", default="",
                     help="统一可得日期；留空则按交易日后一天的保守假设")
    etf.add_argument("--availability-lag-days", type=int, default=1)

    args = parser.parse_args(argv)
    store = PITDataStore(args.root)
    if args.command == "snapshot-universe":
        provider = get_provider(args.source)
        available_at = args.asof or pd.Timestamp.today().strftime("%Y-%m-%d")
        write = store.snapshot_provider_universe(provider, source=args.source, available_at=available_at)
        print(f"已写入主数据快照：{write.path}（{write.rows} 行）")
        print(f"manifest：{write.manifest_path}")
        print("注意：今天抓取的清单不能回填为历史可见数据；历史PIT需导入当期留存的清单/公告。")
        print("⚠ 当前公开源快照通常不含历史AUM/经理/ETF股票型与QDII标签；"
              "严格投资范围应使用 iFinD 或内部CSV通过 import-universe 补齐。")
        return 0

    if args.command == "import-universe":
        frame = pd.read_csv(args.csv, dtype={"code": "string"})
        write = store.write_universe_snapshot(
            frame, source=args.source, available_at=args.available_at,
            effective_date=args.effective_date or None, source_asof=args.source_asof or None,
        )
        print(f"已导入PIT主数据：{write.path}（{write.rows} 行）")
        print(f"manifest：{write.manifest_path}")
        return 0

    if args.command == "fetch-ifind-data-pool":
        try:
            functionpara = json.loads(args.functionpara)
            field_map = json.loads(args.field_map)
        except json.JSONDecodeError as exc:
            parser.error(f"functionpara/field-map 必须是合法JSON：{exc.msg}")
        if not isinstance(functionpara, dict) or not isinstance(field_map, dict):
            parser.error("functionpara 和 field-map 必须是JSON对象")
        client = IFindHTTPClient(cache_dir=store.root / "raw" / "ifind_http")
        response = client.data_pool(args.reportname, functionpara, args.outputpara or None)
        raw = response_to_frame(response)
        frame = map_universe_fields(
            raw, field_map, default_asset_type=args.asset_type or None, aum_unit=args.aum_unit,
        )
        write = store.write_universe_snapshot(
            frame, source=args.source, available_at=args.available_at,
            effective_date=args.effective_date or None, source_asof=args.source_asof or None,
            source_metadata=client.last_request_metadata,
        )
        print(f"iFinD专题报表已写入PIT主数据：{write.path}（{write.rows} 行）")
        print(f"manifest：{write.manifest_path}")
        print("提示：运行严格基金池前，请确认报表映射了 fund_type、aum_yi、inception；ETF还需 is_stock_etf/is_qdii。")
        return 0

    if args.command == "fetch-ifind-p04955":
        client = IFindHTTPClient(cache_dir=store.root / "raw" / "ifind_http")
        response = client.data_pool(
            "p04955",
            {"edate": args.edate, "p0": args.p0, "jjlb": args.jjlb, "user_sectorid": args.user_sectorid},
            P04955_UNIVERSE_OUTPUT,
        )
        raw = response_to_frame(response)
        frame, quarantine_meta = _p04955_frame_with_quarantine(raw, args.edate, store)
        write = store.write_universe_snapshot(
            frame, source=args.source, available_at=args.available_at,
            effective_date=args.effective_date or None, source_asof=args.source_asof or args.edate,
            source_metadata={**client.last_request_metadata, **quarantine_meta},
        )
        print(f"p04955基金主数据已写入PIT：{write.path}（{write.rows} 行）")
        print(f"manifest：{write.manifest_path}")
        print("ETF分类口径：被动指数型股票基金且名称含ETF、不含联接；名称含QDII+ETF者标为QDII ETF并排除。")
        return 0

    if args.command == "backfill-ifind-p04955":
        if args.availability_lag_business_days < 1:
            parser.error("availability-lag-business-days 至少为1，避免把截点日未知数据带入回测")
        dates = pd.date_range(args.start, args.end, freq=args.frequency)
        if dates.empty:
            parser.error("指定区间内没有可回补的期末日期")
        client = IFindHTTPClient(cache_dir=store.root / "raw" / "ifind_http")
        for asof in dates:
            edate = asof.strftime("%Y%m%d")
            response = client.data_pool(
                "p04955",
                {"edate": edate, "p0": args.p0, "jjlb": args.jjlb, "user_sectorid": args.user_sectorid},
                P04955_UNIVERSE_OUTPUT,
            )
            raw = response_to_frame(response)
            frame, quarantine_meta = _p04955_frame_with_quarantine(raw, asof, store)
            available_at = asof + pd.offsets.BDay(args.availability_lag_business_days)
            write = store.write_universe_snapshot(
                frame, source=args.source, effective_date=asof, available_at=available_at,
                source_asof=asof, source_metadata={**client.last_request_metadata, **quarantine_meta},
            )
            print(f"{asof.date()}: {write.rows} 行 → {write.path}")
        print("提示：available_at 使用保守工作日滞后；若供应商可提供真实披露时间，应以真实时间覆写。")
        return 0

    if args.command == "ifind-cache-audit":
        cache_root = store.root / "raw" / "ifind_http"
        manifests = sorted(cache_root.rglob("*.manifest.json")) if cache_root.exists() else []
        data_vol = 0.0
        known_volume = 0
        cache_rows = []
        for path in manifests:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                value = pd.to_numeric(manifest.get("data_vol"), errors="coerce")
                if pd.notna(value):
                    data_vol += float(value)
                    known_volume += 1
                cache_rows.append({
                    "endpoint": manifest.get("endpoint"),
                    "data_vol": manifest.get("data_vol"),
                    "retrieved_at": manifest.get("retrieved_at"),
                    "fingerprint": manifest.get("request_fingerprint"),
                })
            except (OSError, ValueError, TypeError):
                continue
        print(f"iFinD原始缓存：{len(cache_rows)} 份响应，目录 {cache_root}")
        if known_volume:
            print(f"供应商dataVol累计：{data_vol:g}（基于 {known_volume} 份含dataVol的响应）")
        else:
            print("供应商dataVol：历史响应未留存或供应商未返回；后续请求将自动记录。")
        if cache_rows:
            print(pd.DataFrame(cache_rows).tail(10).to_string(index=False))
        return 0

    if args.command == "import-holdings":
        frame = pd.read_csv(args.csv, dtype={"fund_code": "string", "security_code": "string"})
        write = store.write_holdings_snapshot(
            frame, source=args.source, report_period=args.report_period, available_at=args.available_at,
        )
        print(f"已导入PIT持仓：{write.path}（{write.rows} 行）")
        print(f"manifest：{write.manifest_path}")
        return 0

    provider_kwargs = {}
    if args.source == "ifind_http":
        provider_kwargs["cache_dir"] = store.root / "raw" / "ifind_http"
    provider = get_provider(args.source, **provider_kwargs)
    for code in args.codes:
        market = provider_etf_market(provider, code, args.start, args.end)
        write = store.write_market_data(
            market, source=f"{args.source}:etf", available_at=args.available_at or None,
            availability_lag_days=args.availability_lag_days,
            source_metadata=market.attrs.get("ifind_request") if args.source == "ifind_http" else None,
        )
        print(f"{code}: 已写入 {write.rows} 行 → {write.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
