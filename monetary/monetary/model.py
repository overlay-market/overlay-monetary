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
    def __init__(self, unique_id, nx, ny, px, py, base_fee, max_leverage, model):
        self.unique_id = unique_id  # ticker
        self.nx = nx
        self.ny = ny
        self.px = px
        self.py = py
        self.x = nx*px
        self.y = ny*py
        self.k = self.x*self.y
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
        self.cum_price = self.x / self.y
        self.cum_price_idx = 0
        self.last_cum_price = self.x / self.y
        self.last_liquidity = model.liquidity # For liquidity adjustments
        self.last_funding_idx = 0
        self.last_trade_idx = 0
        print("Init'ing FMarket {}".format(self.unique_id))
        print("FMarket x has {}".format(self.unique_id), self.x)
        print("FMarket nx has {} OVL".format(self.unique_id), self.nx)
        print("FMarket y has {}".format(self.unique_id), self.y)
        print("FMarket ny has {} OVL".format(self.unique_id), self.ny)
        print("FMarket k is {}".format(self.unique_id), self.k)

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
            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)
            assert dy < self.y, "slippage: Not enough liquidity in self.y for swap"
            slippage = ((self.x+dx)/(self.y-dy) - self.price()) / self.price()
        else:
            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)
            assert dx < self.x, "slippage: Not enough liquidity in self.x for swap"
            slippage = ((self.x-dx)/(self.y+dy) - self.price()) / self.price()
        return slippage

    def _swap(self, dn, build, long, leverage):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        # TODO: dynamic k upon funding based off OVLETH liquidity changes
        assert leverage < self.max_leverage, "_swap: leverage exceeds max_leverage"
        avg_price = 0.0
        if (build and long) or (not build and not long):
            print("dn = +px*dx")
            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)
            assert dy < self.y, "_swap: Not enough liquidity in self.y for swap"
            avg_price = self.k / (self.x * (self.x+dx))
            self.x += dx
            self.nx += dx/self.px
            self.y -= dy
            self.ny -= dy/self.py
        else:
            print("dn = -px*dx")
            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)
            assert dx < self.x, "_swap: Not enough liquidity in self.x for swap"
            avg_price = self.k / (self.x * (self.x-dx))
            self.y += dy
            self.ny += dy/self.py
            self.x -= dx
            self.nx -= dx/self.px

        print("_swap: {} {} position on {} of size {} OVL at avg price of {}, with lock price {}".format(
            "Built" if build else "Unwound",
            "long" if long else "short",
            self.unique_id,
            dn*leverage,
            1/avg_price,
            self.price(),
        ))
        print("_swap: Percent diff bw avg and lock price is {}%".format(100*(1/avg_price - self.price())/self.price()))
        print("_swap: locked_long -> {} OVL".format(self.locked_long))
        print("_swap: nx -> {}".format(self.nx))
        print("_swap: x -> {}".format(self.x))
        print("_swap: locked_short -> {} OVL".format(self.locked_short))
        print("_swap: ny -> {}".format(self.ny))
        print("_swap: y -> {}".format(self.y))
        self._update_cum_price()
        idx = self.model.schedule.steps
        self.last_trade_idx = idx
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

        # TODO: Account for pro-rata share of funding!
        # TODO: Fix this! something's wrong and I'm getting negative reserve amounts upon unwind :(
        # TODO: Locked long seems to go negative which is wrong. Why here?

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
        ds = amount * pos.leverage * side * (price - pos.lock_price)/pos.lock_price
        print("unwind: {} ds={} OVL from total supply".format(
            "Minting" if ds > 0 else "Burning",
            ds,
        ))
        self.model.supply += ds

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
        # TODO: Fix for px, py sensitivity constant updates! => In practice, use TWAP from Sushi/Uni OVLETH pool for px and TWAP of underlying oracle fetch for p
        # Calculate the TWAP over previous sample
        idx = self.model.schedule.steps
        if (idx % self.model.sampling_interval != 0) or (idx-self.model.sampling_interval < 0) or (idx == self.last_funding_idx):
            return

        # Calculate twap of oracle feed ... each step is value 1 in time weight
        cum_price_feed = np.sum(np.array(
            self.model.sims[self.unique_id][idx-self.model.sampling_interval:idx]
        ))
        print("fund: Paying out funding for {}".format(self.unique_id))
        print("fund: cum_price_feed", cum_price_feed)
        print("fund: sampling_interval", self.model.sampling_interval)
        twap_feed = cum_price_feed / self.model.sampling_interval
        print("fund: twap_feed", twap_feed)

        # Calculate twap of market ... update cum price value first
        self._update_cum_price()
        print("fund: cum_price", self.cum_price)
        print("fund: last_cum_price", self.last_cum_price)
        twap_market = (self.cum_price - self.last_cum_price) / self.model.sampling_interval
        self.last_cum_price = self.cum_price
        print("fund: twap_market", twap_market)

        # Calculate twa open interest for each side over sampling interval
        self._update_cum_locked_long()
        print("fund: nx", self.nx)
        print("fund: px", self.px)
        print("fund: x", self.x)
        print("fund: locked_long", self.locked_long)
        print("fund: cum_locked_long", self.cum_locked_long)
        print("fund: last_cum_locked_long", self.last_cum_locked_long)
        twao_long = (self.cum_locked_long - self.last_cum_locked_long) / self.model.sampling_interval
        print("fund: twao_long", twao_long)
        self.last_cum_locked_long = self.cum_locked_long

        self._update_cum_locked_short()
        print("fund: ny", self.ny)
        print("fund: py", self.py)
        print("fund: y", self.y)
        print("fund: locked_short", self.locked_short)
        print("fund: cum_locked_short", self.cum_locked_short)
        print("fund: last_cum_locked_short", self.last_cum_locked_short)
        twao_short = (self.cum_locked_short - self.last_cum_locked_short) / self.model.sampling_interval
        print("fund: twao_short", twao_short)
        self.last_cum_locked_short = self.cum_locked_short

        # Mark the last funding idx as now
        self.last_funding_idx = idx

        # Mint/burn funding
        funding = (twap_market - twap_feed) / twap_feed
        print("fund: funding % -> {}%".format(funding*100.0))
        if funding == 0.0:
            return
        elif funding > 0.0:
            funding = min(funding, 1.0)
            print("fund: Adding ds={} OVL to total supply".format(funding*(twao_short - twao_long)))
            self.model.supply += funding*(twao_short - twao_long)
            print("fund: Adding ds={} OVL to longs".format(twao_long*(-funding)))
            self.locked_long -= twao_long*funding
            print("fund: Adding ds={} OVL to shorts".format(twao_short*(funding)))
            self.locked_short += twao_short*funding
        else:
            funding = max(funding, -1.0)
            print("fund: Adding ds={} OVL to total supply".format(funding*(twao_long - twao_short)))
            self.model.supply += funding*(twao_long - twao_short)
            print("fund: Adding ds={} OVL to longs".format(twao_long*(funding)))
            self.locked_long += twao_long*funding
            print("fund: Adding ds={} OVL to shorts".format(twao_short*(-funding)))
            self.locked_short -= twao_short*funding

        # Update virtual liquidity reserves
        # p_market = n_x*p_x/(n_y*p_y) = x/y; nx + ny = L/n (ignoring weighting, but maintain price ratio); px*nx = x, py*ny = y;\
        # n_y = (1/p_y)*(n_x*p_x)/(p_market) ... nx + n_x*(p_x/p_y)(1/p_market) = L/n
        # n_x = L/n * (1/(1 + (p_x/p_y)*(1/p_market)))
        print("fund: Adjusting virtual liquidity constants for {}".format(self.unique_id))
        print("fund: nx (prior)", self.nx)
        print("fund: ny (prior)", self.ny)
        print("fund: x (prior)", self.x)
        print("fund: y (prior)", self.y)
        print("fund: price (prior)", self.price())
        liquidity = self.model.liquidity # TODO: use liquidity_supply_emission ...
        liq_scale_factor = liquidity/self.last_liquidity
        print("fund: last_liquidity", self.last_liquidity)
        print("fund: new liquidity", liquidity)
        print("fund: liquidity scale factor", liq_scale_factor)
        self.last_liquidity = liquidity
        self.nx *= liq_scale_factor
        self.ny *= liq_scale_factor
        self.x = self.nx*self.px
        self.y = self.ny*self.py
        self.k = self.x * self.y
        print("fund: nx (updated)", self.nx)
        print("fund: ny (updated)", self.ny)
        print("fund: x (updated)", self.x)
        print("fund: y (updated)", self.y)
        print("fund: price (updated... should be same)", self.price())

        # Calculate twap for ovlusd oracle feed to use in px, py adjustment
        print("fund: Adjusting price sensitivity constants for {}".format(self.unique_id))
        cum_ovlusd_feed = np.sum(np.array(
            self.model.sims["OVL-USD"][idx-self.model.sampling_interval:idx]
        ))
        print("fund: cum_price_feed", cum_ovlusd_feed)
        twap_ovlusd_feed = cum_ovlusd_feed / self.model.sampling_interval
        print("fund: twap_ovlusd_feed", twap_ovlusd_feed)
        self.px = twap_ovlusd_feed # px = n_usd/n_ovl
        self.py = twap_ovlusd_feed/twap_feed # py = px/p
        print("fund: px (updated)", self.px)
        print("fund: py (updated)", self.py)
        print("fund: price (updated... should be same)", self.price())



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

    def __init__(self, unique_id, model, fmarket, pos_max=0.24, deploy_max=0.5, slippage_max=0.02, trade_delay=4*10): # TODO: Fix constraint issues? => related to liquidity values we set ... do we need to weight liquidity based off vol?
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
        self.trade_delay = trade_delay
        self.last_trade_idx = 0
        self.positions = {} # { pos.id: MonetaryPosition }
        self.unwinding = False
        # TODO: store wealth in ETH and OVL, have feeds agent can trade on be
        # OVL/ETH (spot, futures) & TOKEN/ETH (spot, futures) .. start with futures trading only first so can
        # use sim values on underlying spot market. Then can do a buy/sell on spot as well if we want using
        # sims as off-chain price values(?)
        #
        # NOTE: Have defaults for trader be pos max of 0.25 of wealth,
        #       deploy_max of 0.5 of wealth so only two trades outstanding
        #       at a time. With delay between trades of 10 min

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
    def _unwind_positions(self):
        # For now just assume all positions unwound at once (even tho unrealistic)
        for pid, pos in self.positions.items():
            print(f"Arb._unwind_positions: Unwinding position {pid} on {self.fmarket.unique_id}")
            self.fmarket.unwind(pos.amount, pid)
            self.locked -= pos.amount
            self.last_trade_idx = self.model.schedule.steps

        self.positions = {}

    def _unwind_next_position(self):
        # Get the next position from inventory to unwind for this timestep
        if len(self.positions.keys()) == 0:
            self.unwinding = False
            return
        print('Arb._unwind_next_position: positions (prior)', self.positions)
        print('Arb._unwind_next_position: locked (prior)', self.locked)
        pid = list(self.positions.keys())[0]
        pos = self.positions[pid]
        self.fmarket.unwind(pos.amount, pid)
        self.locked -= pos.amount
        self.last_trade_idx = self.model.schedule.steps
        del self.positions[pid]
        print('Arb._unwind_next_position: positions (updated)', self.positions)
        print('Arb._unwind_next_position: locked (updated)', self.locked)

    def trade(self):
        # If market futures price > spot then short, otherwise long
        # Calc the slippage first to see if worth it
        # TODO: Check for an arb opportunity. If exists, trade it ... bet Y% of current wealth on the arb ...
        # Get ready to arb current spreads
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        fprice = self.fmarket.price()

        # TODO: Either wait for funding to unwind OR unwind once
        # reach wealth deploy_max and funding looks to be dried up?

        # Simple for now: tries to enter a pos_max amount of position if it wouldn't
        # breach the deploy_max threshold
        # TODO: make smarter, including thoughts on capturing funding (TWAP'ing it as well) => need to factor in slippage on spot (and have a spot market ...)
        # TODO: ALSO, when should arbitrageur exit their positions? For now, assume at funding they do (again, dumb) => Get dwasse comments here to make smarter
        # TODO: Add in slippage bounds for an order
        # TODO: Have arb bot determine position size dynamically needed to get price close to spot value (scale down size ...)
        # TODO: Have arb bot unwind all prior positions once deploys certain amount (or out of wealth)
        size = self.pos_max*self.wealth
        print(f"Arb.trade: Arb bot {self.unique_id} has {self.wealth-self.locked} OVL left to deploy")
        if self.locked + size < self.deploy_max*self.wealth:
            if sprice > fprice:
                print("Arb.trade: Checking if long position on {} is profitable after slippage ....".format(self.fmarket.unique_id))
                fees = self.fmarket.fees(size, build=True, long=True, leverage=1.0)
                slippage = self.fmarket.slippage(size-fees, build=True, long=True, leverage=1.0)
                print("Arb.trade: fees -> {}".format(fees))
                print("Arb.trade: slippage -> {}".format(slippage))
                if self.slippage_max > abs(slippage) and sprice > fprice * (1+slippage):
                    # enter the trade to arb
                    pos = self.fmarket.build(size, long=True, leverage=1.0)
                    print("Arb.trade: Entered long arb trade w pos params ...")
                    print(f"Arb.trade: pos.amount -> {pos.amount}")
                    print(f"Arb.trade: pos.long -> {pos.long}")
                    print(f"Arb.trade: pos.leverage -> {pos.leverage}")
                    print(f"Arb.trade: pos.lock_price -> {pos.lock_price}")
                    self.positions[pos.id] = pos
                    self.locked += pos.amount
                    self.last_trade_idx = idx
            elif sprice < fprice:
                print("Arb.trade: Checking if short position on {} is profitable after slippage ....".format(self.fmarket.unique_id))
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
                    self.last_trade_idx = idx
        else:
            # TODO: remove but try this here => dumb logic but want to see
            # what happens to currency supply if end up unwinding before each new trade (so only 1 pos per arb)
            self._unwind_positions()

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        idx = self.model.schedule.steps
        # Allow only one trader to trade on a market per block.
        # Add in a trade delay to simulate cooldown due to gas.
        if self.fmarket.last_trade_idx != idx and (self.last_trade_idx == 0 or (idx - self.last_trade_idx) > self.trade_delay):
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
        liquidity_supply_emission,
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

        # Markets: Assume OVL-USD is in here and only have X-USD pairs for now ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        # if x + y = L/n and x/y = p; nx = (L/2n), ny = (L/2n), x*y = k = (px*L/2n)*(py*L/2n)
        n = len(sims.keys())
        prices_ovlusd = self.sims["OVL-USD"]
        print(f"OVL-USD first sim price: {prices_ovlusd[0]}")
        liquidity_weight = {
            list(sims.keys())[i]: 1
            for i in range(n)
        }
        print("liquidity_weight", liquidity_weight)
        self.fmarkets = {
            ticker: MonetaryFMarket(
                unique_id=ticker,
                nx=(self.liquidity/(2*n))*liquidity_weight[ticker],
                ny=(self.liquidity/(2*n))*liquidity_weight[ticker],
                px=prices_ovlusd[0], # px = n_usd/n_ovl
                py=prices_ovlusd[0]/prices[0], # py = px/p
                base_fee=base_market_fee,
                max_leverage=base_max_leverage,
                model=self,
            )
            for ticker, prices in sims.items()
        }

        tickers = list(self.fmarkets.keys())
        for i in range(self.num_agents):
            agent = None
            fmarket = self.fmarkets[tickers[i % len(tickers)]]
            if i < self.num_arbitraguers:
                agent = MonetaryArbitrageur(unique_id=i, model=self, fmarket=fmarket)
            elif i < self.num_arbitraguers + self.num_keepers:
                agent = MonetaryKeeper(unique_id=i, model=self, fmarket=fmarket)
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders:
                agent = MonetaryHolder(unique_id=i, model=self, fmarket=fmarket)
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders:
                agent = MonetaryTrader(unique_id=i, model=self, fmarket=fmarket)
            else:
                agent = MonetaryAgent(unique_id=i, model=self, fmarket=fmarket)

            self.schedule.add(agent)

        # data collector
        # TODO: Why are OVL-USD and ETH-USD futures markets not doing anything in terms of arb bots?
        # TODO: What happens if not enough OVL to sway the market prices on the platform? (i.e. all locked up)
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
