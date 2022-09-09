from assets import Asset, Position
from options import OptionSettings, price_option_black_scholes, Option
from market import SimulatedMarket

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
        initial_capital: float,
        market: SimulatedMarket | None,
        option_settings: OptionSettings,
    ) -> None:
        self.name = name
        self.market = market

        self.market = market
        self.initial_value = initial_capital
        self.capital = initial_capital

        self.positions: list["Position"] = []
        self.value_history = [initial_capital]

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

    @property
    def value(self):
        return self.capital + sum(p.value for p in self.positions)

    def step(self):
        assert self.market is not None
        self.value_history.append(self.capital)

        if self.capital <= 0 or self.market.time_left < self.options_settings.expiry:
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

        strike_price = asset.price

        premium = self.options_settings.pricing_model.get_price(
            asset, self.options_settings.expiry
        )

        created_option = Option(
            created=self.market.time,
            expiry=self.market.time + self.options_settings.expiry,
            strike=strike_price,
            premium=premium,
            asset=asset,
            owner=self,
            multiplicity=self.options_settings.multiplicity,
        )
        self.capital += premium * self.options_settings.multiplicity
        self.market.sell_option(created_option)
