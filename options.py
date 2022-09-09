from raw_heston_model import Heston
from common import fast_cdf

from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Callable, Optional, Protocol
import numpy as np

if TYPE_CHECKING:
    from account import Account
    from assets import Asset


@dataclass(kw_only=True)
class OptionSettings:
    assets_sold_on: list["Asset"]
    risk_free_return: float
    expiry: float
    multiplicity: int
    sell_frequency: float
    pricing_model: "OptionsPricingModel"


@dataclass(kw_only=True)
class Option:
    created: float
    expiry: float
    expired: bool = field(default=False, init=False)
    strike: float
    premium: float
    asset: "Asset"
    owner: "Account"
    terminal_price: Optional[float] = field(default=None, init=False)
    multiplicity: int


class OptionsPricingModel(Protocol):
    strike_price_method: Callable[["Asset"], float]
    risk_free_return: float

    def get_price(self, asset, expiry) -> float:
        ...


def price_option_black_scholes(
    asset_price, strike_price, volatility, risk_free_rate, expiry_time
):

    d1 = (
        np.log(asset_price / strike_price)
        + (risk_free_rate + volatility ** 2 / 2) * expiry_time
    ) / (volatility * np.sqrt(expiry_time))
    d2 = d1 - volatility * np.sqrt(expiry_time)
    cost = asset_price * fast_cdf(d1) - strike_price * np.exp(
        -risk_free_rate * expiry_time
    ) * fast_cdf(d2)

    return cost


@dataclass(kw_only=True)
class BlackScholesPricingModel(OptionsPricingModel):
    strike_price_method: Callable[["Asset"], float]
    risk_free_return: float

    def get_price(self, asset, expiry) -> float:
        return price_option_black_scholes(
            asset.price,
            self.strike_price_method(asset),
            asset.volatility,
            self.risk_free_return,
            expiry,
        )


@dataclass(kw_only=True)
class HestonPricingModel(OptionsPricingModel):
    stock_price: float
    strike_price_method: Callable[["Asset"], float]
    time_to_expiry: float
    risk_free_return: float
    reversion_rate: float
    theta: float
    initial_volatility: float
    lamda: float
    sigma: float
    rho: float

    def __post_init__(self):
        self.raw_heston = Heston(
            1,
            1,
            self.time_to_expiry,
            self.risk_free_return,
            self.reversion_rate,
            self.theta,
            self.initial_volatility,
            self.lamda,
            self.sigma,
            self.rho,
        )

    def get_price(self, asset: "Asset", expiry):
        self.raw_heston.reset_parameters(
            asset.price,
            self.strike_price_method(asset),
            self.time_to_expiry,
            self.risk_free_return,
            self.reversion_rate,
            self.theta,
            self.initial_volatility,
            self.lamda,
            self.sigma,
            self.rho,
        )
        return self.raw_heston.price(0.00001, 100, 10_000)["Call price"]
