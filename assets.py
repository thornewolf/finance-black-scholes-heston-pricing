from dataclasses import dataclass, field

from price_models import (
    PriceChangeModel,
    RiskFreeGrowthModel,
    BrownianWalkModel,
    GeometricBrownianWalkModel,
    GeometricBrownianMotionWithVaryingVariance,
)
from typing import Callable
import numpy as np


@dataclass
class Asset:
    """
    Represents any market asset. Keeps track of its own price history and
    subscribes to some PriceChangeModel for its price movement.
    """

    price_change_model: PriceChangeModel
    name: str
    time: float = field(init=False, default=0)
    time_history: list[float] = field(init=False, default_factory=lambda: [0])
    price_history: list[float] = field(init=False)
    return_history: np.ndarray = field(init=False, default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.price_history = [self.price_change_model.price]

    @property
    def price(self):
        return self.price_change_model.price

    @property
    def volatility(self):
        """
        Standard deviation is the measure of volatility
        """
        return np.std(self.return_history[:60]) / np.sqrt(
            self.time_history[1] - self.time_history[0]
        )

    def step(self, dt: float):
        self.price_change_model.step(dt)
        self.time += dt
        self.time_history.append(self.time)
        self.return_history = np.append(
            self.return_history, self.price / self.price_history[-1]
        )
        self.price_history.append(self.price)


class AssetFactory:
    """
    Generates assets with specific charactaristics.
    """

    def __init__(
        self,
        initial_price_generator: Callable[[], float],
    ) -> None:
        self.initial_value_fn = initial_price_generator

    def get_riskless_asset(self, interest_rate=0.01, name=None):
        """
        Gets a riskless asset
        """
        if name is None:
            name = f"Risk Free r={interest_rate}"
        return Asset(
            RiskFreeGrowthModel(self.initial_value_fn(), interest_rate),
            name,
        )

    def get_brownian_walk_asset(self, drift, variance, name=None):
        """
        Gets a risky asset
        """
        if name is None:
            name = f"BM_{drift:.2f}_{variance:f.2f}"
        return Asset(BrownianWalkModel(self.initial_value_fn(), drift, variance), name)

    def get_geo_brownian_walk_asset(self, drift, variance, name=None):
        if name is None:
            name = f"GBM_{drift:.2f}_{variance:.2f}"
        return Asset(
            GeometricBrownianWalkModel(self.initial_value_fn(), drift, variance),
            name,
        )

    def get_geo_brownian_varying_variance(
        self, drift, variance, long_variance, vol_vol, reversion, correlation
    ):
        return Asset(
            GeometricBrownianMotionWithVaryingVariance(
                self.initial_value_fn(),
                drift,
                variance,
                long_variance,
                vol_vol,
                reversion,
                correlation
            ),
            f"GBM_VV_{drift:.2f}_{variance:.2f}_{long_variance:.2f}_{vol_vol:.2f}",
        )


@dataclass
class Position:
    asset: Asset
    amount: int

    @property
    def value(self):
        return self.asset.price * self.amount
