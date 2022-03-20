from abc import ABC, abstractmethod
import numpy as np


class PriceChangeModel(ABC):
    """
    Abstract implementation of the API that a PriceChangeModel should implement.
    This drives the price of an asset over time.
    It is expected that Var[step(dt);step(dt)] ~= Var[step(2dt)]
    """

    @abstractmethod
    def __init__(self, initial_price, *args, **kwargs) -> None:
        self._price = initial_price

    @property
    def price(self):
        return self._price

    @abstractmethod
    def step(self, dt):
        """
        Update the price according to the model implemented
        """
        ...

    def __repr__(self):
        return f"PriceChangeModel({vars(self)})"


class RiskFreeGrowthModel(PriceChangeModel):
    """
    The characteristic growth model. Expected to grow at interest_rate over dt=1.
    """

    def __init__(self, initial_price, interest_rate) -> None:
        super().__init__(initial_price)
        self.interest_rate = interest_rate

    def step(self, dt):
        self._price = self.price * (1 + self.interest_rate) ** dt


class BrownianWalkModel(PriceChangeModel):
    """
    Implements a stochastic brownian walk of some drift and variance.
    Var[2X] = Var[X] since the step size is the random variable.
    """

    def __init__(self, initial_price, drift, variance) -> None:
        super().__init__(initial_price)
        self.drift = drift
        self.variance = variance

    def step(self, dt):
        self._price = self._price + np.random.normal(
            self.drift / dt, self.variance * np.sqrt(dt)
        )


class GeometricBrownianWalkModel(PriceChangeModel):
    """
    Geometric Brownian motion.
    This generates a lognormal distribution with drift.
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """

    def __init__(self, initial_price, drift, std) -> None:
        super().__init__(initial_price)
        self.drift = drift
        self.std = std

    def step(self, dt):
        self._price = self._price * np.exp(
            np.random.normal(
                (self.drift - self.std ** 2 / 2) * dt,
                self.std * np.sqrt(dt),
            )
        )


class GeometricBrownianMotionWithVaryingVariance(PriceChangeModel):
    """ """

    def __init__(
        self,
        initial_price,
        drift,
        initial_volatility,
        long_variance,
        vol_vol,
        reversion_rate,
    ) -> None:
        super().__init__(initial_price)
        self.drift = drift
        self.volatility = initial_volatility
        self.long_variance = long_variance
        self.reversion_rate = reversion_rate
        self.vol_vol = vol_vol
        self.volatility_history = [initial_volatility]

    def step(self, dt):
        dS = self.drift * self._price * dt + np.random.normal(
            0, self._price * self.volatility * np.sqrt(dt)
        )

        dv = self.reversion_rate * (
            self.long_variance - self.volatility
        ) * dt + self.vol_vol * np.sqrt(self.volatility) * np.random.normal(
            0, 1
        ) * np.sqrt(
            dt
        )

        self._price += dS
        self.volatility += dv
        if self.price < 5:
            exit()
        if self.volatility < 0:
            self.volatility = 0
        self.volatility_history.append(self.volatility)
