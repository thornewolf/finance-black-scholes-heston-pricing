import enum
from assets import AssetFactory
from account import Account
from market import Market
from options import OptionSettings

import logging

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
    MARKET_TIME = 300
    TIME_STEPS_TO_RUN = int(MARKET_TIME / TIMESTEP)
    RISK_FREE_RETURN = 0.00
    logger.info(
        f"""
    {MARKET_TIME=}
    {TIME_STEPS_TO_RUN=}
    """
    )

    initial_price_generator = lambda: 100
    asset_factory = AssetFactory(initial_price_generator)

    investigated_asset = asset_factory.get_geo_brownian_varying_variance(
        0, 0.01, 0.01, 0.01, 0.5
    )
    assets = [asset_factory.get_geo_brownian_walk_asset(0.0, 0.03), investigated_asset]

    market = Market(MARKET_TIME, assets)
    accounts = []
    for i, a in enumerate(assets):
        options_settings = OptionSettings([a], RISK_FREE_RETURN, 3 / 12, 1, 0.25 / 12)
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
