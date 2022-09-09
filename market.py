from options import Option

from matplotlib import pyplot as plt
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from account import Account
    from assets import Asset


class SimulatedMarket:
    """
    Simulates a market containing assets and options.
    """

    def __init__(
        self,
        end_time: float,
        assets: list["Asset"] | None = None,
        accounts: list["Account"] | None = None,
    ) -> None:
        self.time = 0
        self.end_time = end_time
        self.times = [0]
        if assets is None:
            assets = []
        if accounts is None:
            accounts = []
        self.assets = assets
        self.accounts = accounts
        self._live_options: list["Option"] = []
        self._all_options: list["Option"] = []
        self.in_the_money_count = 0
        self.out_of_the_money_count = 0

    @property
    def time_left(self):
        return self.end_time - self.time

    def sell_option(self, option: "Option"):
        self._live_options.append(option)
        self._all_options.append(option)

    def evaluate_options(self):
        freshly_expired = [o for o in self._live_options if self.time > o.expiry]
        for o in freshly_expired:
            if o.asset.price > o.strike:
                self.in_the_money_count += 1
                o.owner.capital -= (o.asset.price - o.strike) * o.multiplicity
            else:
                self.out_of_the_money_count += 1
            o.expired = True
            o.terminal_price = o.asset.price
            self._live_options.remove(o)

    def step(self, dt):
        self.time += dt
        self.times.append(self.time)
        for asset in self.assets:
            asset.step(dt)
        self.evaluate_options()
        for account in self.accounts:
            account.step()

    def register_accounts(self, accounts: list["Account"]):
        self.accounts += accounts

    def plot_market(self, ax):
        for acc in self.accounts:
            ph = np.array(acc.value_history)
            ph = ph / ph[0]
            ax.plot(self.times, ph, "--", linewidth=2)
        for a in self.assets:
            ph = np.array(a.price_history)
            ph = ph / ph[0]
            if "riskless" in a.name:
                plt.plot(self.times, ph, linewidth=3)
            else:
                plt.plot(self.times, ph)
        plt.legend([acc.name for acc in self.accounts] + [a.name for a in self.assets])
        plt.title("Returns of continually priced option accounts")
        plt.xlabel("Time elapsed (yr)")
        plt.ylabel("Return proportion")

    def plot_options(self, axs):
        premiums = np.array([o.premium for o in self._all_options])

        assert all(o.terminal_price is not None for o in self._all_options)
        profits = np.array(
            [
                min(o.premium, o.premium - (o.terminal_price - o.strike))  # type: ignore
                for o in self._all_options
            ]
        )
        ax = axs[0, 0]  # type: ignore
        ax.hist(premiums, 15)
        ax.set_title("Distribution of options premiums")
        ax.set_ylabel("Number of options")
        ax.set_xlabel("Premium ($)")

        ax = axs[0, 1]  # type: ignore
        buckets = np.append(
            np.linspace(min(profits), 0, 5), np.linspace(0, max(profits), 5)
        )
        ax.hist(profits, buckets)
        ax.set_title("Distribution of options profits")
        ax.set_ylabel("Number of options")
        ax.set_xlabel("Profit ($)")

        ax = axs[1, 0]  # type: ignore
        ax.hist2d(premiums, profits, [7, 7])
        ax.set_title("Options profits vs the underlying premium")
        ax.set_xlabel("Premium ($)")
        ax.set_ylabel("Profit ($)")
