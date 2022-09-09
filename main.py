import functools

from assets import AssetFactory

from account import Account

from market import SimulatedMarket
from options import BlackScholesPricingModel, HestonPricingModel, OptionSettings


import logging
import itertools

import matplotlib.pyplot as plt
import numpy as np


def print_stats(accounts, logger, risk_free_return, market_time):
    logger.info(
        f"Final Account values\n" + "\n".join([f"{a.name}:{a.value}" for a in accounts])
    )
    avg_value = sum(a.value for a in accounts) / len(accounts)
    avg_return = np.average([a.total_return for a in accounts])
    logger.info(f"Average account value: {avg_value} | Returns of {avg_return-1:%}")
    logger.info(
        f"Average returns over risk free: {(avg_return) - (1+risk_free_return)**(market_time):%}"
    )


def main():
    logger = logging.Logger(__name__)
    logger.addHandler(logging.StreamHandler())
    TIMESTEP = 0.5 / 12  # Twice a month
    MARKET_TIME = 3
    TIME_STEPS_TO_RUN = int(MARKET_TIME / TIMESTEP) + 1
    RISK_FREE_RETURN = 0.0
    OPTION_EXPIRY_TIME = 1 / 12
    logger.info(
        f"""
    {MARKET_TIME=}
    {TIME_STEPS_TO_RUN=}
    """
    )

    initial_price_generator = lambda: 100
    asset_factory = AssetFactory(initial_price_generator)

    investigated_asset = asset_factory.get_geo_brownian_varying_variance(
        0, 0.01, 0.01, 0.01, 0.5, 0.2
    )
    assets = [asset_factory.get_geo_brownian_walk_asset(0.0, 0.03), investigated_asset]

    market = SimulatedMarket(MARKET_TIME, assets)
    accounts = []

    pricing_models = [
        BlackScholesPricingModel,
        functools.partial(
            HestonPricingModel,
            stock_price=None,
            time_to_expiry=OPTION_EXPIRY_TIME,
            theta=1,
            reversion_rate=1,
            initial_volatility=1,
            lamda=1,
            sigma=1,
            rho=1,
        ),
    ]

    for i, asset, pricing_model in zip(
        itertools.count(), assets, itertools.cycle(pricing_models)
    ):
        options_settings = OptionSettings(
            assets_sold_on=[asset],
            risk_free_return=RISK_FREE_RETURN,
            multiplicity=1,
            sell_frequency=0.25 / 12,
            expiry=OPTION_EXPIRY_TIME,
            pricing_model=pricing_model(
                strike_price_method=lambda asset: asset.price,
                risk_free_return=RISK_FREE_RETURN,
            ),
        )
        accounts.append(Account(f"account {i}", 1000, market, options_settings))

    market.register_accounts(accounts)

    for _ in range(TIME_STEPS_TO_RUN):
        market.step(TIMESTEP)

    _, axs = plt.subplots(2, 2)
    market.plot_options(axs)
    market.plot_market(axs[1, 1])  # type: ignore
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


if __name__ == "__main__":
    main()
