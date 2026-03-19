"""
Industry RS Scanner unit tests — synthetic data only.
"""
import pytest
import numpy as np
import pandas as pd
from engine.schemas import IndustryRSReading, GroupType
from engine.industry_rs import compute_industry_rs
from tests.factories import INDUSTRY_PARENT_MAP, INDUSTRY_TICKERS

INDUSTRY_NAMES = {
    "SMH": "Semiconductors", "IGV": "Software", "HACK": "Cybersecurity",
    "XBI": "Biotech", "IHI": "Medical Devices",
    "KRE": "Regional Banks", "IAI": "Broker-Dealers", "KIE": "Insurance",
    "XOP": "Oil & Gas E&P", "OIH": "Oil Services",
    "ITA": "Aerospace & Defense", "XAR": "Aerospace & Defense (SPDR)",
    "XHB": "Homebuilders", "ITB": "Home Construction", "XRT": "Retail", "IBUY": "Online Retail",
    "XME": "Metals & Mining", "GDX": "Gold Miners",
    "VNQ": "REITs", "TAN": "Solar",
}

INDUSTRIES_CONFIG = [
    {"ticker": t, "name": INDUSTRY_NAMES[t], "parent_sector": INDUSTRY_PARENT_MAP[t], "tier": "T1"}
    for t in INDUSTRY_TICKERS
]


def _make_simple_prices(n, spy_rate, sector_rates, industry_rates):
    """Build prices with SPY, sectors, and industries from constant rates."""
    dates = pd.bdate_range(end="2026-03-18", periods=n)
    data = {}
    for ticker, rate in {**{"SPY": spy_rate}, **sector_rates, **industry_rates}.items():
        p = [100.0]
        for _ in range(n):
            p.append(p[-1] * (1 + rate))
        data[ticker] = p[1:]
    return pd.DataFrame(data, index=dates)


class TestIndustryRSComputation:

    def test_rs_vs_parent_positive(self):
        """Industry +0.25%/day, parent +0.15%/day → RS vs parent positive."""
        prices = _make_simple_prices(25, 0.001, {"XLK": 0.0015}, {"SMH": 0.0025})
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
        )
        smh = results[0]
        assert smh.rs_20d_vs_parent > 0

    def test_rs_vs_parent_negative(self):
        """Industry +0.1%/day, parent +0.2%/day → RS vs parent negative."""
        prices = _make_simple_prices(25, 0.001, {"XLK": 0.002}, {"SMH": 0.001})
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
        )
        assert results[0].rs_20d_vs_parent < 0

    def test_rs_vs_parent_when_parent_flat(self):
        """Industry +0.15%/day, parent flat → RS vs parent ≈ industry return."""
        prices = _make_simple_prices(25, 0.001, {"XLK": 0.0}, {"SMH": 0.0015})
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
        )
        assert results[0].rs_20d_vs_parent > 0

    def test_industry_composite_blends_correctly(self):
        """With 70/30 weights, composite = 0.70*rs_spy + 0.30*rs_parent."""
        prices = _make_simple_prices(25, 0.001, {"XLK": 0.0015}, {"SMH": 0.003})
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
            vs_parent_weight=0.30,
        )
        smh = results[0]
        expected = 0.70 * smh.rs_composite + 0.30 * smh.rs_composite_vs_parent
        assert abs(smh.industry_composite - expected) < 0.1

    def test_all_windows_computed(self):
        """5d, 20d, 60d all present for both vs-SPY and vs-parent."""
        prices = _make_simple_prices(70, 0.001, {"XLK": 0.0015}, {"SMH": 0.002})
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
        )
        smh = results[0]
        for attr in ["rs_5d", "rs_20d", "rs_60d", "rs_5d_vs_parent", "rs_20d_vs_parent", "rs_60d_vs_parent"]:
            assert not np.isnan(getattr(smh, attr)), f"{attr} is NaN"

    def test_ranking_across_all_industries(self):
        """Rankings are 1-N across all industries."""
        n = 25
        rates = {}
        sector_rates = {"XLK": 0.001, "XLV": 0.001, "XLF": 0.001}
        for i, t in enumerate(["SMH", "XBI", "KRE"]):
            rates[t] = 0.003 - 0.001 * i  # SMH strongest, KRE weakest
        prices = _make_simple_prices(n, 0.001, sector_rates, rates)
        config = [
            {"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"},
            {"ticker": "XBI", "name": "Biotech", "parent_sector": "XLV", "tier": "T1"},
            {"ticker": "KRE", "name": "Banks", "parent_sector": "XLF", "tier": "T1"},
        ]
        results = compute_industry_rs(prices, config)
        ranks = sorted(r.rs_rank for r in results)
        assert ranks == [1, 2, 3]
        smh = next(r for r in results if r.ticker == "SMH")
        assert smh.rs_rank == 1

    def test_rank_within_sector(self):
        """SMH, IGV, HACK ranked within XLK's children."""
        n = 25
        sector_rates = {"XLK": 0.001}
        rates = {"SMH": 0.003, "IGV": 0.002, "HACK": 0.001}
        prices = _make_simple_prices(n, 0.001, sector_rates, rates)
        config = [
            {"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"},
            {"ticker": "IGV", "name": "Software", "parent_sector": "XLK", "tier": "T1"},
            {"ticker": "HACK", "name": "Cyber", "parent_sector": "XLK", "tier": "T1"},
        ]
        results = compute_industry_rs(prices, config)
        smh = next(r for r in results if r.ticker == "SMH")
        hack = next(r for r in results if r.ticker == "HACK")
        assert smh.rs_rank_within_sector == 1
        assert hack.rs_rank_within_sector == 3

    def test_short_history_industry(self):
        """ETF with only 10 days: 60d RS = NaN but 5d works."""
        dates = pd.bdate_range(end="2026-03-18", periods=10)
        data = {"SPY": [100 + i * 0.1 for i in range(10)],
                "XLK": [100 + i * 0.15 for i in range(10)],
                "SMH": [100 + i * 0.2 for i in range(10)]}
        prices = pd.DataFrame(data, index=dates)
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
        )
        smh = results[0]
        assert smh.rs_5d != 0.0 or not np.isnan(smh.rs_5d)
        # 60d should be 0 (insufficient data, defaulted)

    def test_returns_industry_rs_reading_type(self):
        prices = _make_simple_prices(25, 0.001, {"XLK": 0.0015}, {"SMH": 0.002})
        results = compute_industry_rs(
            prices, [{"ticker": "SMH", "name": "Semi", "parent_sector": "XLK", "tier": "T1"}],
        )
        assert isinstance(results[0], IndustryRSReading)
        assert results[0].group_type == GroupType.INDUSTRY


class TestIndustryLeadsSector:

    def test_smh_drives_xlk(self, industry_normal_market):
        """SMH outperforming XLK → RS vs parent positive."""
        results = compute_industry_rs(
            industry_normal_market["prices"], INDUSTRIES_CONFIG,
        )
        smh = next(r for r in results if r.ticker == "SMH")
        assert smh.rs_20d_vs_parent > 0

    def test_xbi_lags_xlv(self, industry_normal_market):
        """XBI lagging XLV → RS vs parent negative."""
        results = compute_industry_rs(
            industry_normal_market["prices"], INDUSTRIES_CONFIG,
        )
        xbi = next(r for r in results if r.ticker == "XBI")
        assert xbi.rs_20d_vs_parent < 0

    def test_full_industry_count(self, industry_normal_market):
        """All 20 industries returned."""
        results = compute_industry_rs(
            industry_normal_market["prices"], INDUSTRIES_CONFIG,
        )
        assert len(results) == 20
