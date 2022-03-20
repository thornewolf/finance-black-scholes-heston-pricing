from common import fast_cdf

from dataclasses import dataclass

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from account import Account
    from assets import Asset


@dataclass
class OptionSettings:
    assets_sold_on: list["Asset"]
    risk_free_return: float
    expiry: float
    multiplicity: int
    sell_frequency: float


@dataclass
class Option:
    created: float
    expiry: float
    expired: bool
    strike: float
    premium: float
    asset: "Asset"
    owner: "Account"
    terminal_price: float = -1.0
    multiplicity: int = 0


def price_option_black_scholes(
    asset_price, strike, variance, risk_free_return, expiry_time
):

    d1 = (
        np.log(asset_price / strike)
        + (risk_free_return + variance ** 2 / 2) * expiry_time
    ) / (variance * np.sqrt(expiry_time))
    d2 = d1 - variance * np.sqrt(expiry_time)
    cost = asset_price * fast_cdf(d1) - strike * np.exp(
        -risk_free_return * expiry_time
    ) * fast_cdf(d2)

    return cost
