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


def compute_price_diff(model,
                       ticker: str):
    idx = model.schedule.steps
    sprice = model.sims[ticker][idx]
    fprice = model.fmarkets[ticker].price
    return (fprice - sprice) / sprice


def compute_fprice(model,
                   ticker: str):
    return model.fmarkets[ticker].price


def compute_sprice(model,
                   ticker: str):
    idx = model.schedule.steps
    return model.sims[ticker][idx]


def compute_supply(model):
    return model.supply


def compute_liquidity(model):
    return model.liquidity


def compute_treasury(model):
    return model.treasury


def compute_wealth(model,
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


def calc_inventory_wealth(model,
                          agent: MonetaryAgent,
                          inventory_type: tp.Optional[str] = None,
                          in_usd: bool = False):
    idx = model.schedule.steps
    sprice_ovlusd = model.sims["OVL-USD"][idx]
    sprice = model.sims[agent.fmarket.unique_id][idx]
    base_curr = agent.fmarket.base_currency

    p_constants_ovl = {
        'OVL': 1.0,
        'USD': 1.0/sprice_ovlusd,
        base_curr: sprice/sprice_ovlusd,
    }
    p_constants_usd = {
        k: sprice_ovlusd * v
        for k, v in p_constants_ovl.items()
    }
    p_constants = {}
    if not in_usd:
        p_constants = p_constants_ovl
    else:
        p_constants = p_constants_usd

    if inventory_type in agent.inventory:
        return p_constants[inventory_type] * agent.inventory[inventory_type]

    return sum([v*p_constants[k] for k, v in agent.inventory.items()])


def compute_inventory_wealth(model,
                             agent_type: tp.Optional[tp.Type[MonetaryAgent]] = None,
                             inventory_type: tp.Optional[str] = None,
                             in_usd: bool = False):
    if not agent_type:
        wealths = [
            calc_inventory_wealth(
                model, a, inventory_type=inventory_type, in_usd=in_usd)
            for a in model.schedule.agents
        ]
    else:
        wealths = [
            calc_inventory_wealth(
                model, a, inventory_type=inventory_type, in_usd=in_usd)
            for a in model.schedule.agents if type(a) == agent_type
        ]

    return sum(wealths)
