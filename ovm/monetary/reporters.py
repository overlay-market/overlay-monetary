import typing as tp

from ovm.monetary.agents import MonetaryAgent
from ovm.tickers import OVL_TICKER


def compute_gini(model,
                 agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None):
    agents = [
        a for a in model.schedule.agents
        if agent_type is None or type(a) == agent_type
    ]
    agent_wealths = [agent.wealth for agent in agents]
    x = sorted(agent_wealths)
    N = len(agents)
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return 1.0 + (1.0 / N) - 2.0*B


def compute_price_difference(model,
                             ticker: str):
    idx = model.schedule.steps
    sprice = model.sims[ticker][idx]
    fprice = model.fmarkets[ticker].price
    return (fprice - sprice) / sprice


def compute_futures_price(model, ticker: str):
    return model.fmarkets[ticker].price


def compute_spot_price(model, ticker: str):
    idx = model.schedule.steps
    return model.sims[ticker][idx]


def compute_supply(model):
    return model.supply


def compute_liquidity(model):
    return model.liquidity


def compute_treasury(model):
    return model.treasury


def compute_wealth_for_agent_type(model,
                                  agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None):
    if not agent_type:
        wealths = [a.wealth for a in model.schedule.agents]
    else:
        wealths = [a.wealth
                   for a
                   in model.schedule.agents
                   if type(a) == agent_type
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


def compute_position_count_by_market(model, ticker: str) -> int:
    fmarket = model.fmarkets[ticker]
    pos_map = fmarket.positions
    return len(pos_map)
