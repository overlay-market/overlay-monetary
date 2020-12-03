from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return (1 + (1/N) - 2*B)


# NOTE: Assuming we have an already existing array of price values
# to feed into the model. Make it is larger than expected number
# of time steps, otherwise throw an error
# TODO: Can simulate the OVLETH underlying spot with Uniswap x*y=k as a dynamic
# market whereas other feeds like AAVEETH, etc. are pre-simulated


class MonetaryPosition(object):
    def __init__(self, fmarket, lock_price=0.0, amount=0.0, long=True):
        self.fmarket = fmarket
        self.lock_price = lock_price
        self.amount = amount
        self.long = long


class MonetaryFMarket(object):
    def __init__(self, unique_id, x, y, px, py, k, y_denom='ETH'):
        self.unique_id = unique_id
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.k = k
        self.y_denom = y_denom  # either 'ETH' or 'USD'

    def price(self):
        return self.x / self.y

    def swap(self, dn, buy=True):
        # k = self.px * (nx + dnx) * self.py * (ny - dny)
        # dny = ny - (k/(self.px * self.py))/(nx+dnx) .. mult by py ..
        # dy = y - k/(x + dx)
        if buy:
            print("dn = +dx")
            dx = self.px*dn
            dy = self.y - self.k/(self.x + dx)
            self.x += dx
            self.y -= dy
        else:
            print("dn = -dx")
            dy = self.py*dn
            dx = self.x - self.k/(self.y + dy)
            self.y += dy
            self.x -= dx
        return (self.x/self.y)


class MonetarySMarket(object):
    def __init__(self, unique_id, x, y, k):
        self.unique_id = unique_id
        self.x = x
        self.y = y
        self.k = k

    def price(self):
        return self.x / self.y

    def swap(self, dn, buy=True):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        if buy:
            print("dn = +dx")
            dx = dn
            dy = self.y - self.k/(self.x + dx)
            self.x += dx
            self.y -= dy
        else:
            print("dn = -dx")
            dy = dn
            dx = self.x - self.k/(self.y + dy)
            self.y += dy
            self.x -= dx
        return self.price()


class MonetaryTrader(Agent):  # noqa
    """
    An agent ... these are the arbers with stop losses.
    Add in position hodlers as a different agent
    later (maybe also with stop losses)
    """

    def __init__(self, unique_id, model, fmarket, pos_max=0.1, deploy_max=0.75):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.fmarket = fmarket  # each 'trader' focuses on one market for now
        self.wealth = model.base_wealth
        self.locked = 0
        self.pos_max = pos_max
        self.deploy_max = deploy_max
        self.positions = {
            ticker: MonetaryPosition(ticker)
            for ticker, _ in model.fmarkets.items()
        }
        # TODO: store wealth in ETH and OVL, have feeds agent can trade on be
        # OVL/ETH (spot, futures) & TOKEN/ETH (spot, futures) .. start with futures trading only first so can
        # use sim values on underlying spot market. Then can do a buy/sell on spot as well if we want using
        # sims as off-chain price values(?)

    def pay_funding(self):
        # mint/burn funding for each outstanding position
        # depending on
        i = self.model.schedule.steps
        if i % self.model.sampling_interval != 0:
            return

        ds = 0
        for k, pos in self.positions.items():
            spot = self.model.sims[k][i]
            price = self.model.fmarkets[k].price()
            if pos.amount > 0:
                # TODO: use twap instead (won't make a huge diff for analysis tho)
                funding = pos.amount * (price - spot) / spot
                if not pos.long:
                    funding *= -1

                # So no debts in the sim ...
                if funding > pos.amount:
                    funding = pos.amount

                pos.amount -= funding
                self.positions.update({k: pos})
                ds -= funding

        self.model.supply += ds

    def trade(self):
        pass

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        print("Agent {} activated", self.unique_id)
        if self.wealth > 0 and self.locked / self.wealth < self.deploy_max:
            # Assume only make one trade per step ...
            self.trade()


class MonetaryArbitrageur(MonetaryTrader):
    def trade(self):
        # If market futures price > spot then short, otherwise long
        # Calc the slippage first to see if worth it
        # TODO: Check for an arb opportunity. If exists, trade it ... bet Y% of current wealth on the arb ...
        i = self.model.schedule.steps
        for k, market in self.model.fmarkets.items():
            break


class MonetaryKeeper(Agent):
    def __init__(self, unique_id, model, fmarket):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.fmarket = fmarket
        self.wealth = model.base_wealth

    def distribute_funding(self):
        # Figure out funding payments on each agent's positions
        for agent in self.model.schedule.agents:
            if type(agent) != type(self):
                agent.pay_funding()

    def update_markets(self):
        # Update px, py values from funding oracle fetch
        i = self.model.schedule.steps
        ovleth_spot = self.model.sims['OVL-ETH'][i]
        ethusd_spot = self.model.sims['ETH-USD'][i]

        # always assume Y is ETH so py is ETH/OVL
        market = self.model.fmarkets[self.fmarket]
        spot = self.model.sims[self.fmarket][i]
        py = 1.0 / ovleth_spot  # assume y_denom == 'ETH' as standard
        if market.y_denom == 'USD':
            py /= ethusd_spot

        market.py = py  # special spot market OVLETH or OVLUSD
        market.px = spot * market.py  # spot = px/py
        self.model.fmarkets.update({
            self.fmarket: market
        })

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        print("Agent {} activated", self.unique_id)
        i = self.model.schedule.steps
        if i % self.model.sampling_interval == 0:
            self.distribute_funding()
            self.update_markets()


class MonetaryModel(Model):
    """
    The model class holds the model-level attributes, manages the agents, and generally handles
    the global level of our model.

    There is only one model-level parameter: how many agents the model contains. When a new model
    is started, we want it to populate itself with the given number of agents.

    The scheduler is a special model component which controls the order in which agents are activated.
    """

    def __init__(
        self,
        num_arbitrageurs,
        num_keepers,
        num_holders,
        sims,
        base_wealth,
        sampling_interval
    ):
        super().__init__()
        self.num_agents = num_arbitrageurs + num_keepers + num_holders
        self.num_arbitraguers = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_holders = num_holders
        self.base_wealth = base_wealth
        self.sampling_interval = sampling_interval
        self.supply = base_wealth * self.num_agents
        self.schedule = RandomActivation(self)
        self.sims = sims  # { k: [ prices ] }

        # Markets: Assume OVLETH is in here ...
        self.fmarkets = {
            ticker: MonetaryFMarket(
                ticker,
                100000,
                100000,
                15000,
                1,
                (100000 * 15000) * (100000 * 1)
            )  # TODO: remove hardcode of x,y,px,py,k for real vals
            for ticker, _ in sims.items()
        }

        tickers = list(self.fmarkets.keys())
        for i in range(self.num_agents):
            agent = None
            fmarket = tickers[i % len(tickers)]
            if i < self.num_arbitraguers:
                agent = MonetaryArbitrageur(i, self, fmarket)
            elif i < self.num_arbitraguers + self.num_keepers:
                agent = MonetaryKeeper(i, self, fmarket)
            else:
                agent = MonetaryTrader(i, self, fmarket)

            self.schedule.add(agent)

        # data collector
        # TODO: Track how well futures price tracks spot AND currency supply over time
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"},
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        self.datacollector.collect(self)
        self.schedule.step()
