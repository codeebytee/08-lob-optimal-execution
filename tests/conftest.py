"""Shared fixtures.

The test suite never touches the network and never reads the committed data
snapshot: every fixture here is constructed in code. That is deliberate -
a test that fails because a price moved is not a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market import NameStats  # noqa: E402
from src.utils.config import (AppConfig, BookConfig, ExecutionConfig,  # noqa: E402
                              FlowConfig, MarketConfig)


@pytest.fixture
def stats() -> NameStats:
    """A round-numbered name: $100, 25% vol, 20M ADV, penny tick.

    Round numbers matter here. Several tests check an analytic quantity against
    a simulated one, and being able to do the arithmetic by hand is what makes
    a failure diagnosable.
    """
    return NameStats(ticker="TEST", name="Test Co", price=100.0,
                     sigma_annual=0.25, adv_shares=20_000_000.0,
                     tick_size=0.01, spread_ticks=1.0)


@pytest.fixture
def book_cfg() -> BookConfig:
    return BookConfig(n_levels=10, tick_size=0.01, lot_size=100,
                      initial_depth_lots=20)


@pytest.fixture
def flow_cfg() -> FlowConfig:
    return FlowConfig()


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260818)
