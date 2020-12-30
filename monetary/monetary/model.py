import numpy as np
import uuid
from functools import partial
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
        wealths = [ a.wealth for a in model.schedule.agents ]
    else:
        wealths = [ a.wealth for a in model.schedule.agents if type(a) == agent_type ]
    return sum(wealths)

# NOTE: Assuming we have an already existing array of price values
# to feed into the model. Make it is larger than expected number
# of time steps, otherwise throw an error
# TODO: Can simulate the OVLETH underlying spot with Uniswap x*y=k as a dynamic
# market whereas other feeds like AAVEETH, etc. are pre-simulated


class MonetaryPosition(object):
    def __init__(self, fmarket_ticker, lock_price=0.0, amount=0.0, long=True, leverage=1.0):
        self.fmarket_ticker = fmarket_ticker
        self.lock_price = lock_price
        self.amount = amount
        self.long = long
        self.leverage = leverage
        self.id = uuid.uuid4()

class MonetaryFMarket(object):
    def __init__(self, unique_id, x, y, k, base_fee, max_leverage, model):
        self.unique_id = unique_id  # ticker
        self.x = x
        self.y = y
        self.k = k
        self.base_fee = base_fee
        self.max_leverage = max_leverage
        self.model = model
        self.positions = {} # { id: [MonetaryPosition] }
        self.locked_long = 0.0  # Total OVL locked in long positions
        self.locked_short = 0.0  # Total OVL locked in short positions
        self.cum_locked_long = 0.0
        self.cum_locked_long_idx = 0
        self.cum_locked_short = 0.0 # Used for time-weighted open interest on a side within sampling period
        self.cum_locked_short_idx = 0
        self.last_cum_locked_long = 0.0
        self.last_cum_locked_short = 0.0
        self.cum_price = x / y
        self.cum_price_idx = 0
        self.last_cum_price = x / y
        self.last_funding_idx = 0
        print("Init'ing FMarket {}".format(self.unique_id))
        print("FMarket {} x".format(self.unique_id), x)
        print("FMarket {} y".format(self.unique_id), y)

    def price(self):
        return self.x / self.y

    def _update_cum_price(self):
        # TODO: cum_price and time_elapsed setters ...
        # TODO: Need to check that this is the last swap for given timestep ... (slightly different than Uniswap in practice)
        idx = self.model.schedule.steps
        if idx > self.cum_price_idx:  # and last swap for idx ...
            self.cum_price += (idx - self.cum_price_idx) * self.price()
            self.cum_price_idx = idx

    def _update_cum_locked_long(self):
        idx = self.model.schedule.steps
        if idx > self.cum_locked_long_idx:
            self.cum_locked_long += (idx - self.cum_locked_long_idx) * self.locked_long
            self.cum_locked_long_idx = idx

    def _update_cum_locked_short(self):
        idx = self.model.schedule.steps
        if idx > self.cum_locked_short_idx:
            self.cum_locked_short += (idx - self.cum_locked_short_idx) * self.locked_short
            self.cum_locked_short_idx = idx

    def _impose_fees(self, dn, build, long, leverage):
        # Impose fees, burns portion, and transfers rest to treasury
        size = dn*leverage
        fees = min(size*self.base_fee, dn)

        # Burn 50% and other 50% send to treasury
        print("Burning ds={} OVL from total supply".format(0.5*fees))
        self.model.supply -= 0.5*fees
        self.model.treasury += 0.5*fees

        return dn - fees

    def fees(self, dn, build, long, leverage):
        size = dn*leverage
        return min(size*self.base_fee, dn)

    def slippage(self, dn, build, long, leverage):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        assert leverage < self.max_leverage, "slippage: leverage exceeds max_leverage"
        slippage = 0.0
        if (build and long) or (not build and not long):
            dx = dn*leverage
            dy = self.y - self.k/(self.x + dx)
            assert dy < self.y, "slippage: Not enough liquidity in self.y for swap"
            slippage = ((self.x + dx) / (self.y - dy) - self.price()) / self.price()
        else:
            dy = dn*leverage
            dx = self.x - self.k/(self.y + dy)
            assert dx < self.x, "slippage: Not enough liquidity in self.x for swap"
            slippage = ((self.x - dx) / (self.y + dy) - self.price()) / self.price()
        return slippage

    def _swap(self, dn, build, long, leverage):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        # TODO: dynamic k upon funding based off OVLETH liquidity changes
        assert leverage < self.max_leverage, "_swap: leverage exceeds max_leverage"
        if (build and long) or (not build and not long):
            print("dn = +dx")
            dx = dn*leverage
            dy = self.y - self.k/(self.x + dx)
            assert dy < self.y, "_swap: Not enough liquidity in self.y for swap"
            self.x += dx
            self.y -= dy
        else:
            print("dn = -dx")
            dy = dn*leverage
            dx = self.x - self.k/(self.y + dy)
            assert dx < self.x, "_slippage: Not enough liquidity in self.x for swap"
            self.y += dy
            self.x -= dx

        self._update_cum_price()
        return self.price()

    def build(self, dn, long, leverage):
        # TODO: Factor in shares of lock pools for funding payment portions to work
        amount = self._impose_fees(dn, build=True, long=long, leverage=leverage)
        price = self._swap(amount, build=True, long=long, leverage=leverage)
        pos = MonetaryPosition(self.unique_id, lock_price=price, amount=amount, leverage=leverage)
        self.positions[pos.id] = pos

        # Lock into long/short pool last
        if long:
            self.locked_long += amount
            self._update_cum_locked_long()
        else:
            self.locked_short += amount
            self._update_cum_locked_short()
        return pos

    def unwind(self, dn, pid):
        pos = self.positions.get(pid)
        if pos is None:
            print("No position with pid {} exists on market {}".format(pid, self.unique_id))
            return
        elif pos.amount < dn:
            print("Unwind amount {} is too large for locked position with pid {} amount {}".format(dn, pid, pos.amount))

        # Unlock from long/short pool first
        if pos.long:
            self.locked_long -= dn
            self._update_cum_locked_long()
        else:
            self.locked_short -= dn
            self._update_cum_locked_short()

        amount = self._impose_fees(dn, build=False, long=pos.long, leverage=pos.leverage)
        price = self._swap(amount, build=False, long=pos.long, leverage=pos.leverage)
        side = 1 if pos.long else -1

        # Mint/burn from total supply the profits/losses
        self.model.supply += amount * pos.leverage * side * (price - pos.lock_price)/pos.lock_price

        # Adjust position amounts stored
        if dn == pos.amount:
            del self.positions[pid]
            pos = None
        else:
            pos.amount -= amount
            self.positions[pid] = pos

        return pos

    def fund(self):
        # Pay out funding to each respective pool based on underlying market
        # oracle fetch
        # Calculate the TWAP over previous sample
        idx = self.model.schedule.steps
        if (idx % self.model.sampling_interval != 0) or (idx-self.model.sampling_interval < 0) or (idx == self.last_funding_idx):
            return

        # Calculate twap of oracle feed ... each step is value 1 in time weight
        cum_price_feed = np.sum(np.array(
            self.model.sims[self.unique_id][idx-self.model.sampling_interval:idx]
        ))
        print("Paying out funding for {}".format(self.unique_id))
        print("cum_price_feed", cum_price_feed)
        print("sampling_interval", self.model.sampling_interval)
        twap_feed = cum_price_feed / self.model.sampling_interval
        print("twap_feed", twap_feed)

        # Calculate twap of market ... update cum price value first
        self._update_cum_price()
        print("cum_price", self.cum_price)
        print("last_cum_price", self.last_cum_price)
        twap_market = (self.cum_price - self.last_cum_price) / self.model.sampling_interval
        self.last_cum_price = self.cum_price
        print("twap_market", twap_market)

        # Calculate twa open interest for each side over sampling interval
        self._update_cum_locked_long()
        print("cum_locked_long", self.cum_locked_long)
        print("last_cum_locked_long", self.last_cum_locked_long)
        twao_long = (self.cum_locked_long - self.last_cum_locked_long) / self.model.sampling_interval
        print("twao_long", twao_long)
        self.last_cum_locked_long = self.cum_locked_long

        self._update_cum_locked_short()
        print("cum_locked_short", self.cum_locked_short)
        print("last_cum_locked_short", self.last_cum_locked_short)
        twao_short = (self.cum_locked_short - self.last_cum_locked_short) / self.model.sampling_interval
        print("twao_short", twao_short)
        self.last_cum_locked_short = self.cum_locked_short

        # Mark the last funding idx as now
        self.last_funding_idx = idx

        # Mint/burn funding
        funding = (twap_market - twap_feed) / twap_feed
        print("funding %: {}%".format(funding*100.0))
        if funding == 0.0:
            return
        elif funding > 0.0:
            funding = min(funding, 1.0)
            print("Adding ds={} OVL to total supply".format(funding*(twao_short - twao_long)))
            self.model.supply += funding*(twao_short - twao_long)
            print("Adding ds={} OVL to longs".format(twao_long*(-funding)))
            self.locked_long -= twao_long*funding
            print("Adding ds={} OVL to shorts".format(twao_short*(funding)))
            self.locked_short += twao_short*funding
        else:
            funding = max(funding, -1.0)
            print("Adding ds={} OVL to total supply".format(funding*(twao_long - twao_short)))
            self.model.supply += funding*(twao_long - twao_short)
            print("Adding ds={} OVL to longs".format(twao_long*(funding)))
            self.locked_long += twao_long*funding
            print("Adding ds={} OVL to shorts".format(twao_short*(-funding)))
            self.locked_short -= twao_short*funding


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


class MonetaryAgent(Agent):  # noqa
    """
    An agent ... these are the arbers with stop losses.
    Add in position hodlers as a different agent
    later (maybe also with stop losses)
    """

    def __init__(self, unique_id, model, fmarket, pos_max=0.05, deploy_max=0.75, slippage_max=0.02):
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
        self.slippage_max = slippage_max
        self.positions = {} # { pos.id: MonetaryPosition }
        # TODO: store wealth in ETH and OVL, have feeds agent can trade on be
        # OVL/ETH (spot, futures) & TOKEN/ETH (spot, futures) .. start with futures trading only first so can
        # use sim values on underlying spot market. Then can do a buy/sell on spot as well if we want using
        # sims as off-chain price values(?)

    def trade(self):
        pass

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        # print("Trader agent {} activated".format(self.unique_id))
        if self.wealth > 0 and self.locked / self.wealth < self.deploy_max:
            # Assume only make one trade per step ...
            self.trade()


class MonetaryArbitrageur(MonetaryAgent):
    def trade(self):
        # If market futures price > spot then short, otherwise long
        # Calc the slippage first to see if worth it
        # TODO: Check for an arb opportunity. If exists, trade it ... bet Y% of current wealth on the arb ...
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        fprice = self.fmarket.price()

        # Simple for now: tries to enter a pos_max amount of position if it wouldn't
        # breach the deploy_max threshold
        # TODO: make smarter, including thoughts on capturing funding (TWAP'ing it as well) => need to factor in slippage on spot (and have a spot market ...)
        # TODO: ALSO, when should arbitrageur exit their positions? For now, assume at funding they do (again, dumb) => Get dwasse comments here to make smarter
        size = self.pos_max*self.wealth
        if self.locked + size < self.deploy_max*self.wealth:
            if sprice > fprice:
                print("Arb.trade: Checking if long position is profitable after slippage ....")
                fees = self.fmarket.fees(size, build=True, long=True, leverage=1.0)
                slippage = self.fmarket.slippage(size-fees, build=True, long=True, leverage=1.0)
                print("Arb.trade: fees -> {}".format(fees))
                print("Arb.trade: slippage -> {}".format(slippage))
                if self.slippage_max > abs(slippage) and sprice > fprice * (1+slippage):
                    # enter the trade to arb
                    pos = self.fmarket.build(size, long=True, leverage=1.0)
                    print("Arb.trade: Entered long arb trade w pos params ...")
                    print("Arb.trade: pos.amount -> {}".format(pos.amount))
                    print("Arb.trade: pos.long -> {}".format(pos.long))
                    print("Arb.trade: pos.leverage -> {}".format(pos.leverage))
                    print("Arb.trade: pos.lock_price -> {}".format(pos.lock_price))
                    self.positions[pos.id] = pos
                    self.locked += pos.amount
            elif sprice < fprice:
                print("Arb.trade: Checking if short position is profitable after slippage ....")
                fees = self.fmarket.fees(size, build=True, long=False, leverage=1.0)
                slippage = self.fmarket.slippage(size-fees, build=True, long=False, leverage=1.0) # should be negative ...
                print("Arb.trade: fees -> {}".format(fees))
                print("Arb.trade: slippage -> {}".format(slippage))
                if self.slippage_max > abs(slippage) and sprice < fprice * (1+slippage):
                    # enter the trade to arb
                    pos = self.fmarket.build(size, long=False, leverage=1.0)
                    print("Arb.trade: Entered short arb trade w pos params ...")
                    print("Arb.trade: pos.amount -> {}".format(pos.amount))
                    print("Arb.trade: pos.long -> {}".format(pos.long))
                    print("Arb.trade: pos.leverage -> {}".format(pos.leverage))
                    print("Arb.trade: pos.lock_price -> {}".format(pos.lock_price))
                    self.positions[pos.id] = pos
                    self.locked += pos.amount

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        self.trade()

class MonetaryTrader(MonetaryAgent):
    def trade(self):
        pass


class MonetaryHolder(MonetaryAgent):
    def trade(self):
        pass


class MonetaryKeeper(MonetaryAgent):
    def distribute_funding(self):
        # Figure out funding payments on each agent's positions
        self.fmarket.fund()

    def update_market_liquidity(self):
        # Updates k value per funding payment to adjust slippage
        i = self.model.schedule.steps

        # TODO: Adjust slippage to ensure appropriate price sensitivity
        # per OVL in x, y pools => Start with 1/N * OVLETH liquidity and then
        # do a per market risk weighted avg

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        i = self.model.schedule.steps
        if i % self.model.sampling_interval == 0:
            self.distribute_funding()
            self.update_market_liquidity()


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
        num_traders,
        num_holders,
        sims,
        base_wealth,
        base_market_fee,
        base_max_leverage,
        liquidity,
        treasury,
        sampling_interval
    ):
        super().__init__()
        self.num_agents = num_arbitrageurs + num_keepers + num_traders + num_holders
        self.num_arbitraguers = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_traders = num_traders
        self.num_holders = num_holders
        self.base_wealth = base_wealth
        self.base_market_fee = base_market_fee
        self.base_max_leverage = base_max_leverage
        self.liquidity = liquidity
        self.treasury = treasury
        self.sampling_interval = sampling_interval
        self.supply = base_wealth * self.num_agents + liquidity
        self.schedule = RandomActivation(self)
        self.sims = sims  # { k: [ prices ] }

        # Markets: Assume OVL-USD is in here ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        n = len(sims.keys())
        self.fmarkets = {
            ticker: MonetaryFMarket(
                ticker,
                (self.liquidity/(2*n))*prices[0],
                (self.liquidity/(2*n))*1,
                (self.liquidity/(2*n))*prices[0] * (self.liquidity/(2*n))*1,
                base_market_fee,
                base_max_leverage,
                self,
            )  # TODO: remove hardcode of x,y,px,py,k for real vals
            for ticker, prices in sims.items()
        }

        tickers = list(self.fmarkets.keys())
        for i in range(self.num_agents):
            agent = None
            fmarket = self.fmarkets[tickers[i % len(tickers)]]
            if i < self.num_arbitraguers:
                agent = MonetaryArbitrageur(i, self, fmarket)
            elif i < self.num_arbitraguers + self.num_keepers:
                agent = MonetaryKeeper(i, self, fmarket)
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders:
                agent = MonetaryHolder(i, self, fmarket)
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders:
                agent = MonetaryTrader(i, self, fmarket)
            else:
                agent = MonetaryAgent(i, self, fmarket)

            self.schedule.add(agent)

        # data collector
        # TODO: Track how well futures price tracks spot AND currency supply over time
        model_reporters = {
            "{}-{}".format("d", ticker): partial(compute_price_diff, ticker=ticker)
            for ticker in tickers
        }
        model_reporters.update({
            "{}-{}".format("s", ticker): partial(compute_sprice, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            "{}-{}".format("f", ticker): partial(compute_fprice, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            "Gini": compute_gini,
            "Supply": compute_supply,
            "Treasury": compute_treasury,
            "Liquidity": compute_liquidity,
            "Agent": partial(compute_wealth, agent_type=None),
            "Arbitrageurs": partial(compute_wealth, agent_type=MonetaryArbitrageur),
            "Keepers": partial(compute_wealth, agent_type=MonetaryKeeper),
            "Traders": partial(compute_wealth, agent_type=MonetaryTrader),
            "Holders": partial(compute_wealth, agent_type=MonetaryHolder),
        })
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
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
