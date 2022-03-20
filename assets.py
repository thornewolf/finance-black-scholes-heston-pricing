from price_models import (
    PriceChangeModel,
    RiskFreeGrowthModel,
    BrownianWalkModel,
    GeometricBrownianWalkModel,
    GeometricBrownianMotionWithVaryingVariance,
)
from typing import Callable
import numpy as np


class Asset:
    """
    Represents any market asset. Keeps track of its own price history and
    subscribes to some PriceChangeModel for its price movement.
    """

    def __init__(self, name, price_change_model: PriceChangeModel):
        self.name = name
        self.price_change_model = price_change_model
        self.time = 0
        self.time_history: list[float] = [0]
        self.price_history: list[float] = [price_change_model.price]
        self.return_history = np.array([])

    @property
    def price(self):
        return self.price_change_model.price

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
            name,
            RiskFreeGrowthModel(self.initial_value_fn(), interest_rate),
        )

    def get_brownian_walk_asset(self, drift, variance, name=None):
        """
        Gets a risky asset
        """
        if name is None:
            name = f"BM_{drift:.2f}_{variance:f.2f}"
        return Asset(
            "brownian_walk", BrownianWalkModel(self.initial_value_fn(), drift, variance)
        )

    def get_geo_brownian_walk_asset(self, drift, variance, name=None):
        if name is None:
            name = f"GBM_{drift:.2f}_{variance:.2f}"
        return Asset(
            name,
            GeometricBrownianWalkModel(self.initial_value_fn(), drift, variance),
        )

    def get_geo_brownian_varying_variance(
        self, drift, variance, long_variance, vol_vol, reversion
    ):
        return Asset(
            "new",
            GeometricBrownianMotionWithVaryingVariance(
                self.initial_value_fn(),
                drift,
                variance,
                long_variance,
                vol_vol,
                reversion,
            ),
        )
