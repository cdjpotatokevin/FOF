#!/usr/bin/env python3
"""交易日前申购状态与限额校验。

典型用法：

  python -m fof_system.run_pretrade_check \
      --portfolio-csv run_outputs/2026-06-24/full_eligible_open_subscription_20pct_capacity_portfolio.csv \
      --backup-csv run_outputs/2026-06-24/full_eligible_open_subscription_20pct_capacity_backups.csv \
      --status-source pit --pit-root fof_pit_data --asof 2026-06-23 \
      --report-out pretrade_report.csv --summary-out pretrade_summary.json

生产下单前应优先使用 ``--status-source csv`` 导入渠道/直销/TA 回传的当日状态与限额；
若改用 iFinD p04955，需要先在当前终端会话配置 ``IFIND_REFRESH_TOKEN``。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from .data.ifind_http import IFindHTTPClient, P04955_UNIVERSE_OUTPUT, p04955_to_universe, response_to_frame
from .data.pit import PITDataStore
from .engine.pretrade import evaluate_pretrade_status, load_order_list


def _read_status(args: argparse.Namespace) -> pd.DataFrame:
    if args.status_source == "pit":
        if not args.pit_root:
            raise ValueError("--status-source pit 需要 --pit-root")
        return PITDataStore(args.pit_root).read_universe_asof(args.asof, asset_types=("fund", "etf"))
    if args.status_source == "csv":
        if not args.status_csv:
            raise ValueError("--status-source csv 需要 --status-csv")
        return pd.read_csv(args.status_csv, dtype={"code": "string"})
    if args.status_source == "ifind-p04955":
        edate = args.edate or pd.Timestamp(args.asof).strftime("%Y%m%d")
        client = IFindHTTPClient(cache_dir=(Path(args.pit_root) / "raw" / "ifind_http") if args.pit_root else None)
        response = client.data_pool(
            "p04955",
            {"edate": edate, "p0": args.p0, "jjlb": args.jjlb, "user_sectorid": args.user_sectorid},
            P04955_UNIVERSE_OUTPUT,
        )
        raw = response_to_frame(response)
        return p04955_to_universe(raw)
    raise ValueError(f"未知 status_source: {args.status_source}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF 交易日前申购状态/限额校验")
    parser.add_argument("--portfolio-csv", default="", help="目标组合CSV，需含 code 列")
    parser.add_argument("--backup-csv", default="", help="备选产品CSV，需含 code 列")
    parser.add_argument("--status-source", default="pit", choices=["pit", "csv", "ifind-p04955"],
                        help="交易日状态来源：pit用于研发复核，csv用于渠道/TA回传，ifind-p04955用于iFinD现查")
    parser.add_argument("--status-csv", default="", help="渠道/TA/iFinD导出的当日状态CSV")
    parser.add_argument("--pit-root", default="", help="PIT数据仓；pit来源必填，ifind来源可用于缓存原始响应")
    parser.add_argument("--asof", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="校验日期 YYYY-MM-DD")
    parser.add_argument("--edate", default="", help="iFinD p04955 截止日期 YYYYMMDD，默认由 --asof 转换")
    parser.add_argument("--p0", default="0")
    parser.add_argument("--jjlb", default="051001004")
    parser.add_argument("--user-sectorid", default="", help="iFinD板块ID，例如 username|；仅ifind-p04955需要")
    parser.add_argument("--allow-missing-limit-evidence", action="store_true",
                        help="允许限额字段缺失；仅用于研发诊断，正式下单前不建议使用")
    parser.add_argument("--report-out", default="", help="保存逐产品校验报告CSV")
    parser.add_argument("--summary-out", default="", help="保存摘要JSON")
    parser.add_argument("--soft", action="store_true", help="即使校验失败也返回0；用于生成诊断报告")
    args = parser.parse_args(argv)

    if not args.portfolio_csv and not args.backup_csv:
        parser.error("至少需要 --portfolio-csv 或 --backup-csv")
    if args.status_source == "ifind-p04955" and not args.user_sectorid:
        parser.error("--status-source ifind-p04955 需要 --user-sectorid")

    order_list = load_order_list(args.portfolio_csv or None, args.backup_csv or None)
    status = _read_status(args)
    report, summary = evaluate_pretrade_status(
        order_list,
        status,
        asof=args.asof,
        source=args.status_source,
        require_limit_evidence=not args.allow_missing_limit_evidence,
    )

    print("=== 交易日前校验 ===")
    print(f"状态来源：{args.status_source} | 日期：{args.asof}")
    print(f"校验产品：{summary.total} 只 | 通过：{summary.passed} | 失败：{summary.failed}")
    if summary.failed:
        print("\n失败项：")
        print(report.loc[~report["pretrade_pass"].astype(bool), [
            column for column in ("code", "name", "live_name", "subscription_status", "reason")
            if column in report.columns
        ]].to_string(index=False))
    else:
        print("全部通过。")

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report_out, index=False, encoding="utf-8-sig")
        print(f"逐产品报告已写入：{args.report_out}")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(
            json.dumps({
                "ok": summary.ok,
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "source": summary.source,
                "asof": summary.asof,
                "require_limit_evidence": not args.allow_missing_limit_evidence,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"摘要已写入：{args.summary_out}")
    return 0 if summary.ok or args.soft else 1


if __name__ == "__main__":
    sys.exit(main())
