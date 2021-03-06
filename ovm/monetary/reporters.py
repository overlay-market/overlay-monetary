import typing as tp

import numpy as np

from ovm.monetary.agents import MonetaryAgent
from ovm.monetary.data_collection import (
    AbstractModelReporter,
    AbstractMarketLevelReporter,
    AbstractAgentTypeLevelReporter,
    AbstractAgentReporter
)

from ovm.monetary.markets import MonetaryFPosition
from ovm.monetary.model import MonetaryModel
from ovm.tickers import OVL_TICKER


################################################################################
# Model Level Reporters
################################################################################
class GiniReporter(AbstractAgentTypeLevelReporter[MonetaryModel, MonetaryAgent]):
    def report(self, model) -> float:
        agents = [
            a for a in model.schedule.agents
            if self.agent_type is None or type(a) == self.agent_type
        ]

        agent_wealths = [agent.wealth for agent in agents]
        x = sorted(agent_wealths)
        N = len(agents)
        B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
        return 1.0 + (1.0 / N) - 2.0*B


################################################################################
# Price Difference
################################################################################
def compute_price_difference(model: MonetaryModel, ticker: str) -> float:
    idx = model.schedule.steps
    sprice = model.sims[ticker][idx]
    fprice = model.fmarkets[ticker].price
    return (fprice - sprice) / sprice


class PriceDifferenceReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_price_difference(model, self.ticker)


################################################################################
# Futures Price
################################################################################
class FuturesPriceReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model) -> float:
        return model.fmarkets[self.ticker].price


################################################################################
# Spot Price
################################################################################
def compute_spot_price(model: MonetaryModel, ticker: str) -> float:
    idx = model.schedule.steps
    return model.sims[ticker][idx]


class SpotPriceReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_spot_price(model, self.ticker)


################################################################################
# Supply
################################################################################
def compute_supply(model: MonetaryModel) -> float:
    return model.supply


class SupplyReporter(AbstractModelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_supply(model)


################################################################################
# Liquidity
################################################################################
class LiquidityReporter(AbstractModelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return model.liquidity


################################################################################
# Treasury
################################################################################
def compute_treasury(model: MonetaryModel) -> float:
    return model.treasury


class TreasuryReporter(AbstractModelReporter[MonetaryModel]):
    def report(self, model) -> float:
        return compute_treasury(model)


################################################################################
# Aggregate Wealth
################################################################################
class AggregateWealthForAgentTypeReporter(
        AbstractAgentTypeLevelReporter[MonetaryModel, MonetaryAgent]):
    def report(self, model) -> float:
        if not self.agent_type:
            wealths = [a.wealth for a in model.schedule.agents]
        else:
            wealths = [a.wealth
                       for a
                       in model.schedule.agents
                       if type(a) == self.agent_type
                       ]

        return sum(wealths)


################################################################################
# Inventory Wealth
################################################################################
def compute_inventory_wealth_for_agent(model: MonetaryModel,
                                       agent: MonetaryAgent,
                                       inventory_type: tp.Optional[str] = None,
                                       in_quote: bool = False):
    idx = model.schedule.steps
    sprice_ovl_quote = model.sims[model.ovl_quote_ticker][idx]
    sprice = model.sims[agent.fmarket.unique_id][idx]
    base_curr = agent.fmarket.base_currency

    p_constants_ovl = {
        OVL_TICKER: 1.0,
        model.quote_ticker: 1.0/sprice_ovl_quote,
        base_curr: sprice/sprice_ovl_quote,
    }
    p_constants_quote = {
        k: sprice_ovl_quote * v
        for k, v in p_constants_ovl.items()
    }
    p_constants = {}
    if not in_quote:
        p_constants = p_constants_ovl
    else:
        p_constants = p_constants_quote

    if inventory_type in agent.inventory:
        return p_constants[inventory_type] * agent.inventory[inventory_type]

    return sum([v*p_constants[k] for k, v in agent.inventory.items()])


def compute_inventory_wealth_for_agent_type(
        model: MonetaryModel,
        agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None,
        inventory_type: tp.Optional[str] = None,
        in_quote: bool = False):
    agents = model.schedule.agents
    if agent_type:
        agents = filter(lambda a: type(a) == agent_type, model.schedule.agents)

    return sum(map(lambda a: compute_inventory_wealth_for_agent(model,
                                                                a,
                                                                inventory_type=inventory_type,
                                                                in_quote=in_quote),
                   agents))


class AggregateInventoryWealthForAgentTypeReporter(
        AbstractAgentTypeLevelReporter[MonetaryModel, MonetaryAgent]):
    def __init__(self,
                 agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None,
                 inventory_type: tp.Optional[str] = None,
                 in_quote: bool = False):
        self.inventory_type = inventory_type
        self.in_quote = in_quote
        super().__init__(agent_type=agent_type)

    def report(self, model: MonetaryModel) -> float:
        return compute_inventory_wealth_for_agent_type(model=model,
                                                       agent_type=self.agent_type,
                                                       inventory_type=self.inventory_type,
                                                       in_quote=self.in_quote)


################################################################################
# Skew (Positional Imbalance)
################################################################################
def compute_skew_for_market(model: MonetaryModel, ticker: str, relative: bool = False) -> float:
    monetary_futures_market = model.fmarkets[ticker]
    uuid_to_position_map: tp.Dict[tp.Any, MonetaryFPosition] = monetary_futures_market.positions
    if len(uuid_to_position_map) > 0:
        positional_imbalance = \
            monetary_futures_market.locked_long - monetary_futures_market.locked_short

        total_locked = monetary_futures_market.locked_long + monetary_futures_market.locked_short

        if relative and total_locked != 0.0:
            positional_imbalance = positional_imbalance / total_locked

        return positional_imbalance
    else:
        return 0.0


class SkewReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_skew_for_market(model, self.ticker)


class SkewRelativeReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_skew_for_market(model, self.ticker, relative=True)


################################################################################
# Notional Skew (Positional Imbalance)
################################################################################

def compute_notional_skew_for_market(model: MonetaryModel, ticker: str, relative: bool = False) -> float:
    monetary_futures_market = model.fmarkets[ticker]
    uuid_to_position_map: tp.Dict[tp.Any, MonetaryFPosition] = monetary_futures_market.positions
    if len(uuid_to_position_map) > 0:
        positional_imbalance = \
            monetary_futures_market.locked_long_notional - monetary_futures_market.locked_short_notional

        total_locked = monetary_futures_market.locked_long_notional + monetary_futures_market.locked_short_notional

        if relative and total_locked != 0.0:
            positional_imbalance = positional_imbalance / total_locked

        return positional_imbalance
    else:
        return 0.0


def compute_notional_skew_for_market_per_supply(model: MonetaryModel, ticker: str) -> float:
    monetary_futures_market = model.fmarkets[ticker]
    uuid_to_position_map: tp.Dict[tp.Any, MonetaryFPosition] = monetary_futures_market.positions
    if len(uuid_to_position_map) > 0:
        positional_imbalance = \
            monetary_futures_market.locked_long_notional - monetary_futures_market.locked_short_notional

        return positional_imbalance/model.supply
    else:
        return 0.0


class NotionalSkewReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_notional_skew_for_market(model, self.ticker)


class NotionalSkewRelativeReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_notional_skew_for_market(model, self.ticker, relative=True)


class NotionalSkewRelativeSupplyReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_notional_skew_for_market_per_supply(model, self.ticker)

################################################################################
# Reserve Skew (Virtual Reserve + Locked OVL Imbalance)
################################################################################
def compute_reserve_skew_for_market(model: MonetaryModel, ticker: str, relative: bool = False) -> float:
    monetary_futures_market = model.fmarkets[ticker]
    uuid_to_position_map: tp.Dict[tp.Any, MonetaryFPosition] = monetary_futures_market.positions
    if len(uuid_to_position_map) > 0:
        reserve_imbalance = \
            monetary_futures_market.nx - monetary_futures_market.ny
        if relative:
            reserve_imbalance = reserve_imbalance / monetary_futures_market.ny
        return reserve_imbalance
    else:
        return 0.0


class ReserveSkewReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_reserve_skew_for_market(model, self.ticker)


class ReserveSkewRelativeReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_reserve_skew_for_market(model, self.ticker, relative=True)


################################################################################
# Cost basis
################################################################################
def compute_avg_cost_for_market(model: MonetaryModel, ticker: str, long: bool) -> float:
    fmarket = model.fmarkets[ticker]
    if long:
        return fmarket.locked_long_avg_cost
    else:
        return fmarket.locked_short_avg_cost


def compute_unrealized_pnl_for_market(model: MonetaryModel, ticker: str, long: bool) -> float:
    fmarket = model.fmarkets[ticker]
    if long:
        return fmarket.locked_long_unrealized_pnl
    else:
        return fmarket.locked_short_unrealized_pnl


class AvgCostLongReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_avg_cost_for_market(model, self.ticker, long=True)


class AvgCostShortReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_avg_cost_for_market(model, self.ticker, long=False)


class UnrealizedPnlLongReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_unrealized_pnl_for_market(model, self.ticker, long=True)


class UnrealizedPnlShortReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_unrealized_pnl_for_market(model, self.ticker, long=False)


################################################################################
# Open Positions
################################################################################
def compute_open_positions_per_market(model: MonetaryModel, ticker: str) -> int:
    monetary_futures_market = model.fmarkets[ticker]
    return len(monetary_futures_market.positions)


class OpenPositionReporter(AbstractMarketLevelReporter[MonetaryModel]):
    @property
    def dtype(self) -> np.generic:
        return np.int64

    def report(self, model: MonetaryModel) -> int:
        return compute_open_positions_per_market(model, self.ticker)


################################################################################
# Cumulative funding payments
################################################################################

def compute_cumulative_funding_ds(model: MonetaryModel, ticker: str) -> float:
    fmarket = model.fmarkets[ticker]
    return fmarket.cum_funding_ds


def compute_cumulative_funding_pay_long(model: MonetaryModel, ticker: str) -> float:
    fmarket = model.fmarkets[ticker]
    return fmarket.cum_funding_pay_long


def compute_cumulative_funding_pay_short(model: MonetaryModel, ticker: str) -> float:
    fmarket = model.fmarkets[ticker]
    return fmarket.cum_funding_pay_short


def compute_cumulative_funding_fees(model: MonetaryModel, ticker: str) -> float:
    fmarket = model.fmarkets[ticker]
    return fmarket.cum_funding_fees


class FundingSupplyChangeReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_cumulative_funding_ds(model, self.ticker)


class FundingPaymentsLongReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_cumulative_funding_pay_long(model, self.ticker)


class FundingPaymentsShortReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_cumulative_funding_pay_short(model, self.ticker)


class FundingFeesReporter(AbstractMarketLevelReporter[MonetaryModel]):
    def report(self, model: MonetaryModel) -> float:
        return compute_cumulative_funding_fees(model, self.ticker)

################################################################################
# Agent Level Reporters
################################################################################
class AgentWealthReporter(AbstractAgentReporter[MonetaryAgent]):
    def report(self, agent: MonetaryAgent) -> float:
        return agent.wealth
