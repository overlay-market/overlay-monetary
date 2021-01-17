import typing as tp
from agents import MonetaryAgent


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
    spot_price = model.ticker_to_time_series_of_prices_map[ticker][model.schedule.steps]
    futures_price = model.ticker_to_futures_market_map[ticker].price
    return (futures_price - spot_price) / spot_price


def compute_futures_price(model,
                          ticker: str):
    return model.ticker_to_futures_market_map[ticker].price


def compute_spot_price(model,
                       ticker: str):
    # model.schedule.steps represents the number of time-steps simulated so far
    return model.ticker_to_time_series_of_prices_map[ticker][model.schedule.steps]


def compute_supply(model):
    return model.supply_of_ovl


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
                                       in_usd: bool = False):
    spot_price_ovl_usd = model.ticker_to_time_series_of_prices_map["OVL-USD"][model.schedule.steps]
    spot_price = \
        model.ticker_to_time_series_of_prices_map[agent.futures_market.unique_id][model.schedule.steps]
    base_curr = agent.futures_market.base_currency

    if not in_usd:
        return agent.inventory["OVL"] + agent.inventory["USD"]/spot_price_ovl_usd \
            + agent.inventory[base_curr]*spot_price/spot_price_ovl_usd
    else:
        return agent.inventory["OVL"]*spot_price_ovl_usd + agent.inventory["USD"] \
            + agent.inventory[base_curr]*spot_price


def compute_inventory_wealth_for_agent_type(model,
                                            agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None,
                                            in_usd: bool = False):
    if not agent_type:
        wealths = [
            compute_inventory_wealth_for_agent(model, a, in_usd=in_usd)
            for a in model.schedule.agents
        ]
    else:
        wealths = [
            compute_inventory_wealth_for_agent(model, a, in_usd=in_usd)
            for a in model.schedule.agents if type(a) == agent_type
        ]

    return sum(wealths)


def compute_positional_imbalance_by_market(model, ticker: str) -> float:
    from ovm.monetary.markets import MonetaryFPosition
    monetary_futures_market = model.ticker_to_futures_market_map[ticker]
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

        # print(f'{positional_imbalance_1=}')
        # print(f'{positional_imbalance_2=}')
        # print(f'{positional_imbalance_3=}')
        return positional_imbalance_2
    else:
        return 0.0
