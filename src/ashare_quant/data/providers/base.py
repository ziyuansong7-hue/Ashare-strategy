from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    """Provider contract used by the synchronizer and fake test providers."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def fetch_security_master(self) -> pd.DataFrame: ...

    def fetch_index_constituents(self, index_code: str) -> pd.DataFrame: ...

    def fetch_stock_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        adjust: str,
    ) -> pd.DataFrame: ...

    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame: ...

