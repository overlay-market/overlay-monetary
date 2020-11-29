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


class MonetaryMarket(object):
    def __init__(self, x, y, px, py, k):
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.k = k


class MonetaryAgent(Agent):  # noqa
    """
    An agent ... these are the arbers with stop losses.
    Add in position hodlers as a different agent
    later (maybe also with stop losses)
    """

    def __init__(self, unique_id, model):
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.wealth = model.base_wealth
        self.positions = {
            k: MonetaryPosition()
            for k, _ in model.sims.items()
        }
        # TODO: store wealth in ETH and OVL, have feeds agent can trade on be
        # OVL/ETH (spot, futures) & TOKEN/ETH (spot, futures) .. start with futures trading only first so can
        # use sim values on underlying spot market. Then can do a buy/sell on spot as well if we want using
        # sims as off-chain price values(?)

    def give_money(self):
        other = self.random.choice(self.model.schedule.agents)
        other.wealth += 1
        self.wealth -= 1

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        print("Agent {} activated", self.unique_id)
        # TODO: Check for an arb opportunity. If exists, trade it ...
        if self.wealth > 0:
            # If market futures price > spot then short, otherwise long
            # Calc the slippage first to see if worth it
            self.give_money()


class MonetaryModel(Model):
    """
    The model class holds the model-level attributes, manages the agents, and generally handles
    the global level of our model.

    There is only one model-level parameter: how many agents the model contains. When a new model
    is started, we want it to populate itself with the given number of agents.

    The scheduler is a special model component which controls the order in which agents are activated.
    """

    def __init__(self, num_agents, sims, base_wealth):
        super().__init__()
        self.num_agents = num_agents
        self.base_wealth = base_wealth
        self.schedule = RandomActivation(self)
        self.sims = sims
        self.markets = {
            k: MonetaryMarket()
            for k, _ in sims.items()
        }

        for i in range(self.num_agents):
            agent = MonetaryAgent(i, self)
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
