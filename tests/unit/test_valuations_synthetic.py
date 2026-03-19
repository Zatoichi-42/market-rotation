"""
Valuations unit tests — sigma flags and formatting.
Tests use mocked data (no live yfinance calls).
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from dashboard.components.valuations import (
    compute_valuation_flags, SECTOR_PE_AVERAGES, fetch_valuations_raw,
)


def _mock_raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestValuationFlags:

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_above_1_sigma_flagged(self, mock_fetch):
        """PE well above avg → flagged with direction='above'."""
        # XLK avg=28, std=5 → 35 = 1.4σ above
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLK", "_pe_raw": 35.0, "_fwd_pe_raw": None}])
        flags = compute_valuation_flags(["XLK"], sigma_threshold=1.0)
        pe_flags = [f for f in flags if f["field"] == "P/E"]
        assert len(pe_flags) == 1
        assert pe_flags[0]["direction"] == "above"
        assert pe_flags[0]["sigma"] >= 1.0

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_below_1_sigma_flagged(self, mock_fetch):
        """PE well below avg → flagged with direction='below'."""
        # XLE avg=15, std=5 → 8 = -1.4σ below
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLE", "_pe_raw": 8.0, "_fwd_pe_raw": None}])
        flags = compute_valuation_flags(["XLE"], sigma_threshold=1.0)
        pe_flags = [f for f in flags if f["field"] == "P/E"]
        assert len(pe_flags) == 1
        assert pe_flags[0]["direction"] == "below"

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_within_1_sigma_not_flagged(self, mock_fetch):
        """PE near avg → not flagged."""
        # XLK avg=28, std=5 → 30 = 0.4σ
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLK", "_pe_raw": 30.0, "_fwd_pe_raw": 25.0}])
        flags = compute_valuation_flags(["XLK"], sigma_threshold=1.0)
        assert len(flags) == 0

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_missing_avg_skips_flag(self, mock_fetch):
        """Ticker not in SECTOR_PE_AVERAGES → skip, no crash."""
        mock_fetch.return_value = _mock_raw([{"Ticker": "SMH", "_pe_raw": 25.0, "_fwd_pe_raw": 20.0}])
        flags = compute_valuation_flags(["SMH"], sigma_threshold=1.0)
        assert len(flags) == 0

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_nan_current_value_skips(self, mock_fetch):
        """Current PE is NaN → skip flagging."""
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLK", "_pe_raw": None, "_fwd_pe_raw": None}])
        flags = compute_valuation_flags(["XLK"], sigma_threshold=1.0)
        assert len(flags) == 0

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_flag_message_format(self, mock_fetch):
        """Message includes ticker, field, current, avg, sigma, direction."""
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLK", "_pe_raw": 35.0, "_fwd_pe_raw": None}])
        flags = compute_valuation_flags(["XLK"])
        assert len(flags) >= 1
        msg = flags[0]["message"]
        assert "XLK" in msg
        assert "P/E" in msg
        assert "35" in msg
        assert "σ" in msg

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_expensive_language(self, mock_fetch):
        """PE above avg → 'historically expensive'."""
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLK", "_pe_raw": 40.0, "_fwd_pe_raw": None}])
        flags = compute_valuation_flags(["XLK"])
        pe_flags = [f for f in flags if f["field"] == "P/E"]
        assert "expensive" in pe_flags[0]["message"]

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_cheap_language(self, mock_fetch):
        """PE below avg → 'historically cheap'."""
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLE", "_pe_raw": 7.0, "_fwd_pe_raw": None}])
        flags = compute_valuation_flags(["XLE"])
        pe_flags = [f for f in flags if f["field"] == "P/E"]
        assert "cheap" in pe_flags[0]["message"]

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_fwd_pe_also_flagged(self, mock_fetch):
        """Forward P/E extremes also flagged."""
        # XLK fwd_pe avg=24, std=4 → 35 = 2.75σ
        mock_fetch.return_value = _mock_raw([{"Ticker": "XLK", "_pe_raw": 28.0, "_fwd_pe_raw": 35.0}])
        flags = compute_valuation_flags(["XLK"])
        fwd_flags = [f for f in flags if f["field"] == "Fwd P/E"]
        assert len(fwd_flags) == 1
        assert fwd_flags[0]["direction"] == "above"

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_multiple_tickers(self, mock_fetch):
        """Multiple tickers processed independently."""
        mock_fetch.return_value = _mock_raw([
            {"Ticker": "XLK", "_pe_raw": 40.0, "_fwd_pe_raw": 30.0},
            {"Ticker": "XLE", "_pe_raw": 7.0, "_fwd_pe_raw": 5.0},
            {"Ticker": "XLF", "_pe_raw": 14.0, "_fwd_pe_raw": 12.5},  # Within range
        ])
        flags = compute_valuation_flags(["XLK", "XLE", "XLF"])
        flagged_tickers = {f["ticker"] for f in flags}
        assert "XLK" in flagged_tickers
        assert "XLE" in flagged_tickers
        assert "XLF" not in flagged_tickers  # Within 1σ

    @patch("dashboard.components.valuations.fetch_valuations_raw")
    def test_empty_tickers_returns_empty(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame()
        flags = compute_valuation_flags([])
        assert flags == []
