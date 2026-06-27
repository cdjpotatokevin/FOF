import pandas as pd

from fof_system.engine.subscription import (
    fetch_akshare_subscription_status,
    open_subscription_codes_from_frame,
    subscription_open_mask,
    subscription_status_open,
)


def test_subscription_status_open_cases():
    assert subscription_status_open("开放申购") is True
    assert subscription_status_open("开放大额申购") is True
    assert subscription_status_open("暂停大额申购") is False
    assert subscription_status_open("暂停申购") is False
    assert subscription_status_open("限大额") is False
    assert subscription_status_open("限额申购") is False
    assert subscription_status_open("") is False


def test_subscription_open_mask_blocks_daily_limit():
    df = pd.DataFrame([
        {"code": "A", "subscription_status": "开放申购"},
        {"code": "B", "subscription_status": "限大额", "daily_subscription_limit_yi": 0.01},
        {"code": "C", "subscription_status": "开放申购", "daily_subscription_limit_yi": 0.1},
    ])
    mask = subscription_open_mask(df)
    assert mask.tolist() == [True, False, False]


def test_open_subscription_codes_from_frame():
    frame = pd.DataFrame([
        {"code": "000001", "asset_type": "fund", "subscription_status": "开放申购"},
        {"code": "000002", "asset_type": "fund", "subscription_status": "限大额"},
        {"code": "159307", "asset_type": "etf", "subscription_status": ""},
    ])
    assert open_subscription_codes_from_frame(frame) == {"000001"}


def test_fetch_akshare_subscription_status_smoke():
    frame = fetch_akshare_subscription_status()
    assert {"code", "subscription_status"}.issubset(frame.columns)
    assert not frame.empty
