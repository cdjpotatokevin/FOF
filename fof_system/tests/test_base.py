from __future__ import annotations

import pandas as pd
import pytest

from fof_system.data.base import DataProvider


def test_weekly_returns_exclude_partial_final_week():
    prices = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.to_datetime(["2026-06-12", "2026-06-19", "2026-06-23"]),
    )

    result = DataProvider.to_returns(prices, "W")

    assert result.index.tolist() == [pd.Timestamp("2026-06-19")]
    assert result.iloc[0] == pytest.approx(0.05)
