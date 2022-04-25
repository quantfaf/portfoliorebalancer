from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import seaborn as sns

# module constants used in portfolio construction
ASSET_NAMES = 'asset_names'
ASSET_POSITIONS = 'asset_positions'
ASSET_PRICES = 'asset_prices'
TARGET_ASSET_ALLOCATION = 'target_asset_allocation'
MARKET_VALUE = 'market_value'
TOTAL_MARKET_VALUE = 'total_market_value'
ASSET_ALLOCATION = 'asset_allocation'
REBALANCING_AMOUNT = 'rebalancing_amount'
FUNDS = 'funds'
PREFIX = 'new_'


class RebalancingStrategy(Enum):
    """Enum of available rebalancing strategies."""
    buy_and_sell = 'buy_and_sell'
    buy_only = 'buy_only'
    sell_only = 'sell_only'


class Rebalancer:
    """A simple buy-and-sell, buy-only and sell-only portfolio rebalancing class.

    Financial assets produce different returns *over time* that can change the portfolio's target asset allocation. To
    restore the portfolio's original risk profile, the portfolio should therefore be rebalanced. Standard portfolio
    rebalancing involves mainly two strategies: (1) Rebalancing via buying and selling of assets to maintain a target
    asset allocation, (2) Rebalancing via cash flows, i.e., buying underweight assets with new contributions or in case
    of drawing down the portfolio, selling overweight assets with withdrawals.
    """

    def __init__(self,
                 asset_names: list,
                 asset_positions: list,
                 asset_prices: list,
                 target_asset_allocation: list,
                 rebalancing_strategy: str = 'buy_and_sell',
                 funds: float = 0.0):
        """Validates and constructs all the necessary attributes of the Rebalancer object."""
        Rebalancer._validate_equal_length(asset_names, asset_positions, asset_prices, target_asset_allocation)
        Rebalancer._validate_rebal_strategy_and_funds(rebalancing_strategy, funds)
        self.asset_names = asset_names
        self.asset_positions = asset_positions
        self.asset_prices = asset_prices
        self.target_asset_allocation = target_asset_allocation
        self.rebalancing_strategy = rebalancing_strategy
        self._funds = funds
        self._portfolio = self._get_portfolio()
        self._portfolio_rebalanced = pd.DataFrame()
        self._methods_map = {RebalancingStrategy.buy_and_sell: self._rebalancing_buy_and_sell,
                             RebalancingStrategy.buy_only: self._rebalancing_buy_or_sell_only,
                             RebalancingStrategy.sell_only: self._rebalancing_buy_or_sell_only}

    @staticmethod
    def _validate_all_types(value, validation_type, property_name):
        # validates if all elements of a list object are of a particular type
        if not all(isinstance(element, validation_type) for element in value):
            raise TypeError(f'Elements of property {property_name} must be of type {validation_type.__name__}.')

    @staticmethod
    def _validate_larger_or_equal_zero(value, property_name):
        # validates if all elements of a list object are larger or equal to zero
        if not all(element >= 0.0 for element in value):
            raise ValueError(f"Elements of property {property_name} must be larger or equal to zero.")

    @staticmethod
    def _validate_equal_length(*args):
        # validates if all list objects are of same length
        if not all(len(args[0]) == len(element) for element in args):
            raise ValueError('Portfolio properties must be of same length.')

    @staticmethod
    def _validate_rebal_strategy_and_funds(rebal_strategy, funds):
        # validates if rebalancing strategy is consistent with funds
        if RebalancingStrategy(rebal_strategy) == RebalancingStrategy.buy_only:
            if not funds > 0:
                raise ValueError("Rebalancing strategy 'buy_only' requires property funds to be larger than zero.")
        elif RebalancingStrategy(rebal_strategy) == RebalancingStrategy.sell_only:
            if not funds < 0:
                raise ValueError("Rebalancing strategy 'sell_only' requires property funds to be smaller than zero.")

    @property
    def asset_names(self):
        """list(str): The names of the portfolio assets."""
        return self._asset_names

    @asset_names.setter
    def asset_names(self, value):
        Rebalancer._validate_all_types(value, str, ASSET_NAMES)
        self._asset_names = value

    @property
    def asset_positions(self):
        """list(float): The asset positions of the portfolio.

        A position in an asset must be larger or equal to zero."""
        return self._asset_positions

    @asset_positions.setter
    def asset_positions(self, value):
        Rebalancer._validate_all_types(value, float, ASSET_POSITIONS)
        Rebalancer._validate_larger_or_equal_zero(value, ASSET_POSITIONS)
        self._asset_positions = value

    @property
    def asset_prices(self):
        """list(float): The asset prices of the portfolio.

        A price of an asset must be larger or equal to zero."""
        return self._asset_prices

    @asset_prices.setter
    def asset_prices(self, value):
        Rebalancer._validate_all_types(value, float, ASSET_PRICES)
        Rebalancer._validate_larger_or_equal_zero(value, ASSET_PRICES)
        self._asset_prices = value

    @property
    def target_asset_allocation(self):
        """list(float): The target asset allocation of the portfolio.

        The target allocation of an asset must be between zero and one. The sum of the target asset allocation must be
        equal to one."""
        return self._target_asset_allocation

    @target_asset_allocation.setter
    def target_asset_allocation(self, value):
        Rebalancer._validate_all_types(value, float, TARGET_ASSET_ALLOCATION)
        if not all(((element >= 0.0) & (element <= 1.0)) for element in value):
            raise ValueError(
                f"Elements of property {TARGET_ASSET_ALLOCATION} must be between zero and one.")
        if not abs(sum(value) - 1.0) <= 1e-6:
            raise ValueError(f"Elements of property {TARGET_ASSET_ALLOCATION} must be sum to one.")
        self._target_asset_allocation = value

    @property
    def rebalancing_strategy(self):
        """str: The strategy of rebalancing to calculate rebalancing amounts of the portfolio.

        The rebalancing strategy must be 'buy_and_sell', 'buy_only', or 'sell_only'."""
        return self._rebalancing_strategy

    @rebalancing_strategy.setter
    def rebalancing_strategy(self, value):
        rebalancing_strategies = [element.value for element in RebalancingStrategy]
        if value not in rebalancing_strategies:
            str_rt = ', '.join(rebalancing_strategies)
            raise ValueError(f"Property rebalancing_strategy must be of value: {str_rt}.")
        self._rebalancing_strategy = value

    @property
    def funds(self):
        """float: The cash flow to be added or deducted from the total market value of the portfolio."""
        return self._funds

    @property
    def portfolio(self):
        """pd.DataFrame: An overview of the portfolio."""
        return self._portfolio

    @property
    def portfolio_rebalanced(self):
        """pd.DataFrame: An overview of the rebalanced portfolio."""
        return self._portfolio_rebalanced

    def _get_portfolio(self):
        # creates a portfolio and returns it as a dataframe
        d = {ASSET_POSITIONS: self._asset_positions,
             ASSET_PRICES: self._asset_prices,
             TARGET_ASSET_ALLOCATION: self._target_asset_allocation}
        df = pd.DataFrame(data=d)
        df.index = self._asset_names
        df = df.sort_values(by=[TARGET_ASSET_ALLOCATION], ascending=False)
        df[MARKET_VALUE] = df[ASSET_PRICES] * df[ASSET_POSITIONS]
        if df[MARKET_VALUE].sum() == 0.0:
            df[ASSET_ALLOCATION] = 0.0
        else:
            df[ASSET_ALLOCATION] = df[MARKET_VALUE].div(df[MARKET_VALUE].sum())
        df[TOTAL_MARKET_VALUE] = df[MARKET_VALUE].sum()
        df[FUNDS] = self._funds
        df[add_prefix(TOTAL_MARKET_VALUE)] = df[TOTAL_MARKET_VALUE] + df[FUNDS]
        return df

    def rebalance(self):
        """Calculates the rebalancing amounts for the portfolio.

        Returns:
            Rebalancer: Returns the instance of the class (self).
        """
        df = self._portfolio.copy()
        # rebalancing amount dependent on method
        df[REBALANCING_AMOUNT] = self._methods_map[RebalancingStrategy(self._rebalancing_strategy)]()
        # new market value
        df[add_prefix(MARKET_VALUE)] = df[MARKET_VALUE] + df[REBALANCING_AMOUNT]
        # new asset position
        df[add_prefix(ASSET_POSITIONS)] = df[ASSET_POSITIONS] + df[REBALANCING_AMOUNT].div(
            df[ASSET_PRICES])
        # new asset allocation
        df[add_prefix(ASSET_ALLOCATION)] = df[add_prefix(MARKET_VALUE)].div(
            df[add_prefix(TOTAL_MARKET_VALUE)])
        # save rebalanced portfolio
        self._portfolio_rebalanced = df
        return self

    def _rebalancing_buy_and_sell(self):
        # calculates rebalancing amounts for buy-and-sell strategy
        return self._portfolio[TARGET_ASSET_ALLOCATION] * self._portfolio[add_prefix(TOTAL_MARKET_VALUE)] - \
               self._portfolio[MARKET_VALUE]

    def _rebalancing_buy_or_sell_only(self):
        # calculates rebalancing amounts for buy-only and sell-only strategy
        set_of_linear_constraints = self._get_linear_constraints()
        # initial guess for minimizer
        x0 = np.zeros(len(self._asset_names))
        # minimize quadratic loss function
        res = minimize(Rebalancer._quadratic_loss_function, x0,
                       args=(self._portfolio[MARKET_VALUE],
                             self._portfolio[TARGET_ASSET_ALLOCATION]),
                       method='trust-constr',
                       constraints=set_of_linear_constraints)
        return res.x

    def _get_linear_constraints(self):
        # number of assets
        num_assets = len(self._asset_names)
        # construct linear budget constraint for each asset
        lc_selector = np.eye(num_assets).tolist()
        bound_zero = [0.0] * num_assets
        bound_funds = [self._funds] * num_assets
        if RebalancingStrategy(self._rebalancing_strategy) == RebalancingStrategy.buy_only:
            lc_lower = bound_zero
            lc_upper = bound_funds
        elif RebalancingStrategy(self._rebalancing_strategy) == RebalancingStrategy.sell_only:
            lc_lower = bound_funds
            lc_upper = bound_zero
        else:
            raise ValueError('Unknown rebalancing strategy for minimization problem.')
        # construct equality budget constraint
        lc_selector.append(np.ones(num_assets).tolist())
        lc_lower.append(self._funds)
        lc_upper.append(self._funds)
        # define complete set of linear constraints
        set_of_linear_constraints = LinearConstraint(lc_selector, lc_lower, lc_upper)
        return set_of_linear_constraints

    def plot(self, content: str = 'rebalanced_portfolio', normalized: bool = True, fig_size: tuple = (10, 6)):
        """Creates a different content of plots of the rebalanced portfolio.

        Args:
            content (str, optional): Specifies the content of plot. Select between 'rebalanced_portfolio' (default),
                'portfolio', 'rebalancing_amounts' and 'deviation'.
            normalized (bool): Specifies if amounts are expressed in percent of portfolio's total market value.
                Default is True.
            fig_size (tuple): Specifies the width and height of the figure in inches. Default is (10, 6).

        Returns:
            (matplotlib.figure, matplotlib.axes): Returns figure and axes objects.
        """
        fig, ax = RebalancerPlotter(self._portfolio_rebalanced.copy(), content, normalized, fig_size).plot()
        plt.show()
        return fig, ax

    @staticmethod
    def _quadratic_loss_function(x, market_value, target_asset_allocation):
        # sum of squared deviations from target asset allocation
        mse = 0
        for i_position in range(len(target_asset_allocation)):
            mse += ((market_value[i_position] + x[i_position]) /
                    (sum(market_value) + sum(x)) - target_asset_allocation[i_position]) ** 2
        return mse / len(target_asset_allocation)


class RebalancerPlotter:
    """Visualizes the rebalanced portfolio."""

    def __init__(self, portfolio_rebalanced: pd.DataFrame, content, normalized, fig_size):
        self._portfolio_rebalanced = portfolio_rebalanced
        self._content = content
        self._normalized = normalized
        self._fig_size = fig_size
        self._color_palette = sns.color_palette()
        self._bar_width = 0.3
        self._methods_map = {'portfolio': self._plot_portfolio,
                             'rebalanced_portfolio': self._plot_rebalanced_portfolio,
                             'rebalancing_amounts': self._plot_rebalancing_amounts,
                             'deviation': self._plot_deviation}

    def plot(self):
        sns.set_theme()
        return self._methods_map[self._content]()

    def _plot_portfolio(self):
        # get x values
        x_values = self._get_x_values()
        edges_target_line = 0.2
        # get y values
        if self._normalized:
            y_values = self._portfolio_rebalanced[ASSET_ALLOCATION] * 100
            target_values = self._portfolio_rebalanced[TARGET_ASSET_ALLOCATION] * 100
        else:
            y_values = self._portfolio_rebalanced[MARKET_VALUE]
            target_values = self._portfolio_rebalanced[TARGET_ASSET_ALLOCATION] * \
                            self._portfolio_rebalanced[TOTAL_MARKET_VALUE]
        x_target, y_target = self._get_target_lines(target_values, edges_target_line)
        # create figure
        fig, ax = plt.subplots(figsize=self._fig_size)
        # bars
        ax.bar(x_values, y_values, self._bar_width, color=self._color_palette[0])
        # create horizontal line
        for x_i in x_values:
            ax.plot(x_target[x_i], y_target[x_i], color=self._color_palette[3], linestyle='dashed', linewidth=2.0)
        ax.set_xticks(x_values)
        ax.set_xticklabels(self._get_x_ticklabels())
        ax.set_ylabel(self._get_y_label())
        return fig, ax

    def _plot_rebalanced_portfolio(self):
        # get x values
        x_values = self._get_x_values()
        edges_target_line = 0.4
        # get y values
        if self._normalized:
            y_values_portfolio = self._portfolio_rebalanced[ASSET_ALLOCATION] * 100
            y_values_rebal_portfolio = self._portfolio_rebalanced[add_prefix(ASSET_ALLOCATION)] * 100
            target_values = self._portfolio_rebalanced[TARGET_ASSET_ALLOCATION] * 100
        else:
            y_values_portfolio = self._portfolio_rebalanced[MARKET_VALUE]
            y_values_rebal_portfolio = self._portfolio_rebalanced[add_prefix(MARKET_VALUE)]
            target_values = self._portfolio_rebalanced[add_prefix(TOTAL_MARKET_VALUE)] * \
                            self._portfolio_rebalanced[TARGET_ASSET_ALLOCATION]
        x_target, y_target = self._get_target_lines(target_values, edges_target_line)
        # create figure
        fig, ax = plt.subplots(figsize=self._fig_size)
        # bar group plot
        ax.bar([x - self._bar_width / 2 for x in x_values], y_values_portfolio, self._bar_width,
               color=self._color_palette[0], label='portfolio')
        ax.bar([x + self._bar_width / 2 for x in x_values], y_values_rebal_portfolio, self._bar_width,
               color=self._color_palette[1], label='portfolio rebalanced')
        # create horizontal line
        for x_i, _ in enumerate(x_values):
            ax.plot(x_target[x_i], y_target[x_i], color=self._color_palette[3], linestyle='dashed', linewidth=2.0)
        ax.set_xticks(x_values)
        ax.set_xticklabels(self._get_x_ticklabels())
        ax.set_ylabel(self._get_y_label())
        ax.legend()
        return fig, ax

    def _plot_rebalancing_amounts(self):
        # get x values
        x_values = self._get_x_values()
        # get y values
        if self._normalized:
            y_values = self._portfolio_rebalanced[REBALANCING_AMOUNT].div(
                self._portfolio_rebalanced[add_prefix(TOTAL_MARKET_VALUE)])
            y_label = 'rebalancing amounts (in %)'
        else:
            y_values = self._portfolio_rebalanced[REBALANCING_AMOUNT]
            y_label = 'rebalancing amounts'
        # create figure
        fig, ax = plt.subplots(figsize=self._fig_size)
        # check if buy-only, sell-only or buy-and-sell
        if all(y_values > 0.0):
            ax.bar(x_values, y_values, self._bar_width, color=self._color_palette[2], label='buy')
        elif all(y_values <= 0.0):
            ax.bar(x_values, y_values, self._bar_width, color=self._color_palette[3], label='sell')
        else:
            ax.bar(x_values[y_values >= 0.0], y_values[y_values >= 0.0], self._bar_width, color=self._color_palette[2],
                   label='buy')
            ax.bar(x_values[y_values < 0.0], y_values[y_values < 0.0], self._bar_width, color=self._color_palette[3],
                   label='sell')
            # zero line
            ax.axhline(linewidth=2.0, color=self._color_palette[7])
        ax.set_xticks(x_values)
        ax.set_xticklabels(self._get_x_ticklabels())
        ax.set_ylabel(y_label)
        ax.legend()
        return fig, ax

    def _plot_deviation(self):
        # get x values
        x_values = self._get_x_values()
        # get y values
        y_values = (self._portfolio_rebalanced[
                        ASSET_ALLOCATION] - self._portfolio_rebalanced[TARGET_ASSET_ALLOCATION]) * 100
        # create figure
        fig, ax = plt.subplots(figsize=self._fig_size)
        if all(y_values > 0.0):
            ax.bar(x_values, y_values, self._bar_width, color=self._color_palette[2])
        elif all(y_values <= 0.0):
            ax.bar(x_values, y_values, self._bar_width, color=self._color_palette[3])
        else:
            ax.bar(x_values[y_values >= 0.0], y_values[y_values >= 0.0], self._bar_width, color=self._color_palette[2])
            ax.bar(x_values[y_values < 0.0], y_values[y_values < 0.0], self._bar_width, color=self._color_palette[3])
            # zero line
            ax.axhline(linewidth=2.0, color=self._color_palette[7])
        ax.set_xticks(x_values)
        ax.set_xticklabels(self._get_x_ticklabels())
        ax.set_ylabel('deviation from target (in ppt)')
        return fig, ax

    def _get_target_lines(self, target_values, edges_target_line):
        x = []
        y = []
        for x_i in self._get_x_values():
            x.append([-edges_target_line + x_i, edges_target_line + x_i])
            y.append([target_values[x_i], target_values[x_i]])
        return x, y

    def _get_x_values(self):
        return np.array([*range(len(self._portfolio_rebalanced.index))])

    def _get_x_ticklabels(self):
        return self._portfolio_rebalanced.index

    def _get_y_label(self):
        if self._normalized:
            y_label = 'portfolio allocation (in %)'
        else:
            y_label = 'portfolio allocation'
        return y_label


def add_prefix(name):
    # attaches PREFIX to a string
    return PREFIX + name
