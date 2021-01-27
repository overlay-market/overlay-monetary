from abc import ABC, abstractmethod
import typing as tp

from ovm.monetary.agents import MonetaryAgent
from ovm.tickers import OVL_TICKER


REPORT_RESULT_TYPE = tp.Union[int, float]


class AbstractReporter(ABC):
    @abstractmethod
    def report(self, model) -> REPORT_RESULT_TYPE:
        pass

    def __call__(self, model) -> REPORT_RESULT_TYPE:
        return self.report(model)


class AbstractMarketLevelReporter(AbstractReporter, ABC):
    def __init__(self, ticker: str):
        self.ticker = ticker


class AbstractAgentLevelReporter(AbstractReporter, ABC):
    def __init__(self,
                 agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None):
        self.agent_type = agent_type


class GiniReporter(AbstractAgentLevelReporter):
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


def compute_price_difference(model, ticker: str) -> float:
    idx = model.schedule.steps
    sprice = model.sims[ticker][idx]
    fprice = model.fmarkets[ticker].price
    return (fprice - sprice) / sprice


class PriceDifferenceReporter(AbstractMarketLevelReporter):
    def report(self, model) -> float:
        return compute_price_difference(model, self.ticker)


class FuturesPriceReporter(AbstractMarketLevelReporter):
    def report(self, model) -> float:
        return model.fmarkets[self.ticker].price


def compute_spot_price(model, ticker: str) -> float:
    idx = model.schedule.steps
    return model.sims[ticker][idx]


class SpotPriceReporter(AbstractMarketLevelReporter):
    def report(self, model) -> float:
        return compute_spot_price(model, self.ticker)


def compute_supply(model) -> float:
    return model.supply


class SupplyReporter(AbstractReporter):
    def report(self, model) -> float:
        return compute_supply(model)


class LiquidityReporter(AbstractReporter):
    def report(self, model) -> float:
        return model.liquidity


def compute_treasury(model) -> float:
    return model.treasury


class TreasuryReporter(AbstractReporter):
    def report(self, model) -> float:
        return compute_treasury(model)


class WealthForAgentTypeReporter(AbstractAgentLevelReporter):
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


def compute_inventory_wealth_for_agent(model,
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


def compute_inventory_wealth_for_agent_type(model,
                                            agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None,
                                            inventory_type: tp.Optional[str] = None,
                                            in_quote: bool = False):
    if not agent_type:
        wealths = [
            compute_inventory_wealth_for_agent(
                model, a, inventory_type=inventory_type, in_quote=in_quote)
            for a in model.schedule.agents
        ]
    else:
        wealths = [
            compute_inventory_wealth_for_agent(
                model, a, inventory_type=inventory_type, in_quote=in_quote)
            for a in model.schedule.agents if type(a) == agent_type
        ]

    return sum(wealths)


def compute_positional_imbalance_by_market(model, ticker: str) -> float:
    from ovm.monetary.markets import MonetaryFPosition
    monetary_futures_market = model.fmarkets[ticker]
    uuid_to_position_map: tp.Dict[tp.Any, MonetaryFPosition] = monetary_futures_market.positions
    if len(uuid_to_position_map) > 0:
        # import numpy as np

        # positional_imbalance_1 = \
        #     sum(position.directional_size for position in uuid_to_position_map.values())

        positional_imbalance_2 = \
            monetary_futures_market.locked_long - monetary_futures_market.locked_short

        # assert np.isclose(positional_imbalance_1, positional_imbalance_2)

        # positional_imbalance_3 = \
        #     monetary_futures_market.nx - monetary_futures_market.ny

        # print(f'positional_imbalance_1={positional_imbalance_1}')
        # print(f'positional_imbalance_2={positional_imbalance_2}')
        # print(f'positional_imbalance_3={positional_imbalance_3}')
        return positional_imbalance_2
    else:
        return 0.0


class SkewReporter(AbstractMarketLevelReporter):
    def report(self, model) -> float:
        return compute_positional_imbalance_by_market(model, self.ticker)


def compute_open_positions_per_market(model, ticker: str) -> float:
    monetary_futures_market = model.fmarkets[ticker]
    return len(monetary_futures_market.positions)


class OpenPositionReporter(AbstractMarketLevelReporter):
    def report(self, model) -> int:
        return compute_open_positions_per_market(model, self.ticker)
