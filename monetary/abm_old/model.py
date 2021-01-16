# model.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from collections import namedtuple
import numpy as np
import random
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from collections import defaultdict
from data_feeds import RandomFeed
from strategies import Strategy, Enter, Exit

#Market = namedtuple('Market', ('type', 'px'))

def get_avg_px(trade1, trade2):
    return (trade1.amount*trade1.px + trade2.amount*trade2.px)/(trade1.amount + trade2.amount)

def compute_gini(model):
    agent_wealths = [agent.earned_wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)

def get_currency_supply(model):
    return model.currency_supply

def get_wealth(model):
    return sum([agent.earned_wealth for agent in model.schedule.agents])

class Trader(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(
            self,
            tid,
            model, #the Trader has access to the System through this
            wealth,
            markets,
            strategy,
            positions,
            **kwargs):
        super().__init__(tid, model)
        self.initial_wealth = wealth
        self.earned_wealth = wealth
        self.locked_wealth = 0 # use this
        self.markets = markets
        self.strategy = strategy
        self.positions = positions
        self.trades = []
        self.upl = [0] #to take initial step into account
        self.num_trades = 0
        self.num_wins = 0
        self.num_losses = 0

    def pay_fees(self, trade):
        fee = get_fee(trade.amount, self.model.free_fee)
        trade.amount = trade.amount - fee
        self.earned_wealth -= fee
        self.model.fees += fee
        #self.model.fee_pool += fee
        self.model.pool += fee
        self.model.currency_supply -= fee
        return fee, trade

    def get_return(self, trade):
        mid = trade.market
        base_trade = self.positions[mid]
        base_px = base_trade.px
        ret = base_trade.side*(trade.px/base_px - 1)
        if ret > self.model.exit_cap:
            return self.model.exit_cap
        return ret

    def get_pl(self,  trade):
        amt = trade.amount
        return amt*self.get_return(trade)

    def mark_pos(self, trade):
        amt = trade.amount
        base_px = trade.px
        current_px = self.model.markets[trade.market].history[-1]
        return amt*trade.side*(current_px/base_px - 1)

    def step(self):
        #print(self.unique_id, self.num_trades)

        for market in self.markets:
            #import ipdb ; ipdb.set_trace()
            if self.positions[market]:
                self.upl.append(self.mark_pos(self.positions[market]))
            else:
                self.upl.append(0)
            trades = self.strategy.yield_trades(market, self, self.model)
            #import ipdb ; ipdb.set_trace()
            for trade in trades:
                self.num_trades += 1
                fee, trade = self.pay_fees(trade)
                self.trades.append((self.model.step_num, trade.px, trade.amount, trade.side))
                if trade.reference:
                    base_trade = self.positions[trade.market]
                    pos1 = base_trade.px * base_trade.amount
                    pos2 = trade.px * trade.amount
                    if base_trade.side != trade.side:
                        side = base_trade.side
                        #import ipdb ; ipdb.set_trace()
                        pl = self.get_pl(trade)

                        if pl > 0:
                            self.num_wins += 1
                            the_buffer = max(self.model.max_supply - self.model.currency_supply, 0)
                            if the_buffer < pl:
                                #import ipdb ; ipdb.set_trace()
                                self.model.lost_wins += the_buffer - pl
                                pl = the_buffer
                            self.model.currency_supply += pl # self.model.base_pool
                            self.model.wins += pl

                        else:
                            if pl < -self.locked_wealth:
                                pl = -self.locked_wealth
                            self.num_losses += 1
                            self.model.losses += pl
                            self.model.currency_supply += pl #losses are negative

                        self.earned_wealth += pl
                        self.locked_wealth -= (trade.amount + fee)
                        if self.earned_wealth < 0:
                            self.earned_wealth = 0

                        #if base_trade.amount == trade.amount: #always del position becuase we trade all capital
                        del self.positions[trade.market]
                        # else:
                        #     self.positions[trade.market].amount -= trade.amount
                        #     self.positions[trade.market].px = get_avg_px(base_trade, trade)

                    else:
                        # self.earned_wealth -= (trade.amount + fee)
                        # if self.earned_wealth < 0 :
                        #      import ipdb ; ipdb.set_trace()
                    #   fee, trade = self.pay_fees(trade)
                        self.positions[trade.market].amount += trade.amount
                        self.positions[trade.market].px = get_avg_px(base_trade, trade)

                else:
#                    self.earned_wealth -= trade.amount
#                    if self.earned_wealth<0:
#                        import ipdb ; ipdb.set_trace()
                #    fee, trade = self.pay_fees(trade)
                    self.positions[trade.market] = trade
                    self.locked_wealth += trade.amount 

        #update markets
        #update strategy




def get_fee(amt, fee):
    return amt*fee


class OVLModel(Model):
    """A model with some number of agents."""
    res_model_attrs = ['currency_supply',
                                'max_supply',
                                'losses',
                                'wins',
                                'lost_wins',
                                'pool',
                                'num_steps_pool_empty',
                                'num_steps_pool_low',
                                'fees']
    res_agent_attrs = ['initial_wealth',
                                'earned_wealth',
                                'num_trades',
                                'num_wins',
                                'num_losses',
                                'locked_wealth',
                                ]
    
    res_attrs = res_model_attrs + res_agent_attrs
    
    def __init__(self,
                 num_traders,
                 num_markets,
                 base,
                 free_fee,
                 exit_cap = 1,
                 markets = {0:RandomFeed(1, 1e5, 1, dist_args=(0, 1e5/100))},
                 strategy = Strategy(Enter, Exit, .2, .2),
                 issue = 0,
                 pct = False
                 ):
        self.num_agents = num_traders
        self.num_markets = num_markets
        self.currency_supply = 1e6
        self.exit_cap = exit_cap
        self.base = base
        self.losses = 0
        self.wins = 0
        self.lost_wins = 0
        self.pool = base
        self.max_supply = 1e6 + base
        self.printed = 0 #how much was printed at each period
        self.maturity = 24 #in steps (24 steps in  va day)
        self.step_num = 0
        self.num_steps_pool_empty = 0
        self.num_steps_pool_low = 0
        self.bound_fee = 0
        self.free_fee = free_fee
        self.fees = 0
        #self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True #for BatchRunner
        self.markets = markets
        self.issue = issue
        self.pct = pct

        
        # Create agents
        for i in range(self.num_agents):
            wealth = self.get_wealth_distribution()
            markets = [0]
            positions = defaultdict(lambda: None)
            a = Trader(
                tid=i,
                model=self,
                wealth=wealth,
                markets=markets,
                positions=positions,
                strategy=strategy)
            self.schedule.add(a)


    def step(self):
        self.step_num  += 1
        self.max_supply += self.issue
        for m in self.markets.values():
            m.update(pct=self.pct)
        self.schedule.step()

    def get_wealth_distribution(self):
        return self.base/100
        while True:
            wealth = np.random.exponential(self.base)
            if wealth >= 1 and wealth <= self.base*.9:
                return wealth*.1

    def get_results(self):
        res = {}
        for attr in self.res_attrs:
            res[attr] = self._data(attr)
        return res

    def _data(self, attr):
        if attr in self.res_model_attrs:
            return getattr(self, attr)
        elif attr in self.res_agent_attrs:
            return  sorted([(a.unique_id, getattr(a, attr)) for a in self.schedule.agents])
