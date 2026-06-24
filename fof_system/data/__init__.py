"""数据层：可插拔的行情/基金数据源。"""
from .base import DataProvider, ETF_MARKET_COLUMNS, FundMeta
from .pit import PITDataError, PITDataStore

__all__ = ["DataProvider", "FundMeta", "ETF_MARKET_COLUMNS", "PITDataError", "PITDataStore", "get_provider"]


def get_provider(name: str = "akshare", **kwargs) -> DataProvider:
    """工厂：按名称返回数据源实例。

    name: 'akshare' | 'ifind' | 'ifind_http' | 'mock'
    """
    name = name.lower()
    if name == "akshare":
        from .akshare_provider import AkshareProvider
        return AkshareProvider(**kwargs)
    if name == "ifind":
        from .ifind_provider import IFinDProvider
        return IFinDProvider(**kwargs)
    if name == "ifind_http":
        from .ifind_http_provider import IFindHTTPProvider
        return IFindHTTPProvider(**kwargs)
    if name == "mock":
        from .mock_provider import MockProvider
        return MockProvider(**kwargs)
    raise ValueError(f"未知数据源: {name}")
