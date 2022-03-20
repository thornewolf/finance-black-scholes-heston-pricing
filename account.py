from assets import Asset
from options import OptionSettings, price_option_black_scholes, Option
from market import Market

from dataclasses import dataclass
from functools import cache
import logging
import random

import numpy as np


class Account:
    """
    An Account/Portfolio that manages it's own value through the creation of options.
    """

    def __init__(
        self,
        name: str,
        initial_value: int,
        market: Market | None,
        option_settings: OptionSettings,
    ) -> None:
        self.name = name
        self.initial_value = initial_value
        self.value = initial_value
        self.value_history = [initial_value]

        self.market = market

        self.time_last_sold_options = 0
        self.options_settings = option_settings

        self.logger = logging.Logger(__name__ + f".Account.{self.name}")
        self.logger.addHandler(logging.StreamHandler())

    @property
    def returns(self):
        return np.array(self.value_history) / self.initial_value

    @property
    def total_return(self):
        return self.returns[-1]

    def step(self):
        assert self.market is not None
        self.value_history.append(self.value)

        if self.value <= 0 or self.market.time_left < self.options_settings.expiry:
            return
        asset = random.choice(self.options_settings.assets_sold_on)
        if len(asset.return_history) < 3:
            return

        if (
            self.market.time
            < self.time_last_sold_options + self.options_settings.sell_frequency
        ):
            return

        self.time_last_sold_options = self.market.time

        asset_price = asset.price
        strike_price = asset.price
        variance = np.std(asset.return_history[:60]) / np.sqrt(
            asset.time_history[1] - asset.time_history[0]
        )

        premium = price_option_black_scholes(
            asset_price,
            strike_price,
            variance,
            self.options_settings.risk_free_return,
            self.options_settings.expiry,
        )

        created_option = Option(
            self.market.time,
            self.market.time + self.options_settings.expiry,
            False,
            strike_price,
            premium,
            asset,
            self,
            multiplicity=self.options_settings.multiplicity,
        )
        self.value += premium * self.options_settings.multiplicity
        self.market.add_option(created_option)
