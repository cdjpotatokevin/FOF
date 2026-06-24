import pandas as pd

from fof_system.data.ifind_http_provider import IFindHTTPProvider


class _Client:
    def date_sequence(self, *args, **kwargs):
        return {"tables": [{"thscode": "399370.SZ", "table": {
            "time": ["2026-06-23", "2026-06-24"], "ths_pe_ttm_sr_index": [12.1, 12.2],
        }}]}

    def history_quotation(self, *args, **kwargs):
        return {"tables": [{"thscode": "589990.SH", "table": {
            "time": ["2026-06-23"], "close": [1.02], "amount": [123_000_000],
        }}]}


def test_http_provider_parses_super_command_valuation_and_etf_amount():
    provider = IFindHTTPProvider(client=_Client())
    pe = provider.get_index_valuation("399370.SZ", "pe_ttm", "2026-06-01", "2026-06-24")
    market = provider.get_etf_market("589990.SH", "2026-06-01", "2026-06-24")
    assert pe.iloc[-1] == 12.2
    assert pe.index[-1] == pd.Timestamp("2026-06-24")
    assert market.loc[0, "code"] == "589990"
    assert market.loc[0, "amount"] == 123_000_000


class _UndatedClient:
    def date_sequence(self, *args, **kwargs):
        return {"tables": [{"thscode": "399370.SZ", "table": {
            "ths_pb_latest_index": [1.0, 1.1],
        }}]}


def test_http_provider_marks_undated_date_sequence_for_calendar_alignment():
    provider = IFindHTTPProvider(client=_UndatedClient())
    pb = provider.get_index_valuation("399370", "pb", "2026-06-01", "2026-06-02")

    assert pb.tolist() == [1.0, 1.1]
    assert pb.attrs["ifind_date_sequence_without_dates"] is True


class _MarketCalendar:
    def get_fund_nav(self, _code, _start, _end):
        return pd.Series([1.0, 1.1], index=pd.to_datetime(["2026-06-01", "2026-06-02"]))


class _UndatedMarketClient:
    def history_quotation(self, *args, **kwargs):
        return {"tables": [{"thscode": "589990.SH", "table": {
            "close": [1.0, 1.1], "amount": [100.0, 110.0],
        }}]}


def test_http_provider_aligns_undated_etf_market_to_verified_calendar():
    provider = IFindHTTPProvider(client=_UndatedMarketClient(), calendar_provider=_MarketCalendar())
    market = provider.get_etf_market("589990.SH", "2026-06-01", "2026-06-02")

    assert market["trading_date"].tolist() == [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")]
    assert market["amount"].tolist() == [100.0, 110.0]


class _ChunkClient:
    def __init__(self):
        self.ranges = []

    def date_sequence(self, _code, _indipara, start, end, **_kwargs):
        self.ranges.append((start, end))
        return {"tables": [{"thscode": "399370.SZ", "table": {"ths_pb_latest_index": [1.0]}}]}


def test_http_provider_splits_long_undated_sequences_into_annual_requests():
    client = _ChunkClient()
    provider = IFindHTTPProvider(client=client)

    pb = provider.get_index_valuation("399370", "pb", "2024-01-01", "2025-01-02")

    assert client.ranges == [("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-01-02")]
    assert pb.tolist() == [1.0, 1.0]
