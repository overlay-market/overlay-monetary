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
    def __init__(self, lock_price=0.0, amount=0.0, long=True):
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

    def __init__(self, unique_id, model, pos_max=0.1, deploy_max=0.75):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.wealth = model.base_wealth
        self.locked = 0
        self.pos_max = pos_max
        self.deploy_max = deploy_max
        self.positions = {
            k: MonetaryPosition()
            for k, _ in model.sims.items()
        }
        # TODO: store wealth in ETH and OVL, have feeds agent can trade on be
        # OVL/ETH (spot, futures) & TOKEN/ETH (spot, futures) .. start with futures trading only first so can
        # use sim values on underlying spot market. Then can do a buy/sell on spot as well if we want using
        # sims as off-chain price values(?)

    def pay_funding(self):
        # mint/burn funding for each outstanding position
        # depending on
        i = self.model.scheduler.steps
        if i % self.model.sampling_interval != 0:
            return

        ds = 0
        for k, pos in self.positions.items():
            spot = self.model.sims[k][i]
            price = self.model.markets[k].price()
            if pos.amount > 0:
                # TODO: use twap instead (won't make a huge diff for analysis tho)
                funding = pos.amount * (price - spot) / spot
                if not pos.long:
                    funding *= -1

                # So no debts in the sim ...
                if funding > pos.amount:
                    funding = pos.amount

                pos.amount -= funding
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
        i = self.model.scheduler.steps


class MonetaryKeeper(Agent):
    def __init__(self, unique_id, model):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.wealth = model.base_wealth

    def distribute_funding(self):
        # Figure out funding payments on each agent's positions
        for agent in self.model.scheduler.agents:
            if agent.unique_id != self.unique_id:
                agent.pay_funding()

    def update_markets(self):
        # Update px, py values from funding oracle fetch
        i = self.model.scheduler.steps
        ovleth_spot = self.model.markets['OVLETH'][i]
        ethusd_spot = self.model.markets['ETHUSD'][i]
        for k, market in self.model.markets.items():
            # always assume Y is ETH so py is ETH/OVL
            spot = self.model.sims[k][i]
            py = 1.0 / ovleth_spot  # assume y_denom == 'ETH' as standard
            if market.y_denom == 'USD':
                py /= ethusd_spot

            market.py = py  # special spot market OVLETH or OVLUSD
            market.px = spot * market.py  # spot = px/py

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        print("Agent {} activated", self.unique_id)
        i = self.model.scheduler.steps
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

    def __init__(self, num_arbitrageurs, num_keepers, num_holders, sims, base_wealth):
        super().__init__()
        self.num_agents = num_arbitrageurs + num_keepers + num_holders
        self.num_arbitraguers = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_holders = num_holders
        self.base_wealth = base_wealth
        self.supply = base_wealth * self.num_agents
        self.schedule = RandomActivation(self)
        self.sims = sims  # { k: [ prices ] }

        # Markets: Assume OVLETH is in here ...
        self.fmarkets = {
            k: MonetaryFMarket(k)  # x, y, px, py, k)
            for k, _ in sims.items()
        }

        for i in range(self.num_agents):
            agent = None
            if i < self.num_arbitraguers:
                agent = MonetaryArbitrageur(i, self)
            elif i < self.num_arbitraguers + self.num_keepers:
                agent = MonetaryKeeper(i, self)
            else:
                agent = MonetaryTrader(i, self)

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
