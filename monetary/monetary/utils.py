def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return 1.0 + (1.0 / N) - 2.0*B


def compute_price_diff(model, ticker):
    idx = model.schedule.steps
    sprice = model.sims[ticker][idx]
    fprice = model.fmarkets[ticker].price()
    return (fprice - sprice) / sprice


def compute_fprice(model, ticker):
    return model.fmarkets[ticker].price()


def compute_sprice(model, ticker):
    idx = model.schedule.steps
    return model.sims[ticker][idx]


def compute_supply(model):
    return model.supply


def compute_liquidity(model):
    return model.liquidity


def compute_treasury(model):
    return model.treasury


def compute_wealth(model, agent_type=None):
    wealths = []
    if not agent_type:
        wealths = [a.wealth for a in model.schedule.agents]
    else:
        wealths = [
            a.wealth
            for a in model.schedule.agents if type(a) == agent_type
        ]

    return sum(wealths)


def calc_inventory_wealth(model, agent, in_usd=False):
    idx = model.schedule.steps
    sprice_ovlusd = model.sims["OVL-USD"][idx]
    sprice = model.sims[agent.fmarket.unique_id][idx]
    base_curr = agent.fmarket.base_currency

    if not in_usd:
        return agent.inventory["OVL"] + agent.inventory["USD"]/sprice_ovlusd \
            + agent.inventory[base_curr]*sprice/sprice_ovlusd
    else:
        return agent.inventory["OVL"]*sprice_ovlusd + agent.inventory["USD"] \
            + agent.inventory[base_curr]*sprice


def compute_inventory_wealth(model, agent_type=None, in_usd=False):
    wealths = []
    if not agent_type:
        wealths = [
            calc_inventory_wealth(model, a, in_usd=in_usd)
            for a in model.schedule.agents
        ]
    else:
        wealths = [
            calc_inventory_wealth(model, a, in_usd=in_usd)
            for a in model.schedule.agents if type(a) == agent_type
        ]

    return sum(wealths)
