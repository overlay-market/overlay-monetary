import numpy as np
import typing as tp
from collections import namedtuple
#from model import Trader, Model, Tripdbaipdde, Market
class Trade:
    def __init__(self, mid, amt, side, px, step, tid, reference):
        self.market = mid
        self.amount = amt
        self.side = side
        self.px = px
        self.step = step
        self.tid = tid
        self.reference = reference





class Exit:
    #check for positions
    @staticmethod
    def has_position(mid, trader, model) -> dict:
        return trader.positions[mid]
    #check for exit points
    @staticmethod
    def exit_signal(mid, trader, model) -> bool:
       # import ipdb; ipdb.set_trace()
        if trader.earned_wealth - trader.mark_pos(trader.positions[mid]) <= 0:
            return True
        if np.random.ranf() < trader.strategy.exit_prob:
            return True
        return False#return np.random.choice([False, True])
    #get side
    @staticmethod
    def get_side(mid, trader, model) -> int:
        return trader.positions[mid].side*-1
    #see how much to exit
    @staticmethod
    def unwind_amount(mid, trader, model) -> int:
        ###return np.random.choice([.25, .5, 1, 1])*
        if trader.earned_wealth - trader.mark_pos(trader.positions[mid]) <= 0:
            return trader.locked_wealth#hack works so long as traders always trade whole wad    positions[mid].amount
        return trader.locked_wealth#positions[mid].amount

class Enter:
    
    #check if can trade
    @staticmethod
    def can_trade(mid, trader, model) -> bool:
        if trader.earned_wealth - trader.locked_wealth> 0:
            return True
        return False
    #what markets to trade
    # @staticmethod
    # def get_tradable_markets(mid, trader, model) -> list:
    #     return []               #
    #check for signals (to trade)
    @staticmethod
    def entry_signal(mid, trader, model) -> bool:
        #the trade looks at the buffer:
        if max(model.max_supply - model.currency_supply, 0) > trader.earned_wealth*.2:
            if np.random.ranf() < trader.strategy.entry_prob:#return np.random.choice([False, True])
                return True
        return False
    #how much to trade
    @staticmethod
    def entry_amount(mid, trader, model) -> int:
        return trader.earned_wealth#XS*np.random.choice([.05, .1, .25, .5, .75, 1])
    #what direction to trade
    @staticmethod
    def get_side(mid, trader, model) -> int:
        return np.random.choice([1,-1])

# class Trade():
#     pass

class Strategy():
    #perform trader actions
    def __init__(self, Enter, Exit, entry_prob, exit_prob):
        self.Enter = Enter
        self.Exit = Exit
        self.entry_prob = entry_prob
        self.exit_prob = exit_prob

    def yield_trades(self, mid, trader, model) -> dict:
        args = (mid, trader, model)
        #import ipdb ; ipdb.set_trace()
        base_trade = self.Exit.has_position(*args)
        if base_trade and self.Exit.exit_signal(*args):
            amt = self.Exit.unwind_amount(*args)
            side = self.Exit.get_side(*args)
            px = model.markets[mid].px
            tid = trader.num_trades + 1
            yield Trade(
                mid,
                amt,
                side,
                px,
                model.schedule.steps,
                tid,
                base_trade.tid
            )
        elif self.Enter.can_trade(*args) and self.Enter.entry_signal(*args) and self.Enter.entry_amount(*args):
                amt = self.Enter.entry_amount(*args)
                side = self.Enter.get_side(*args)
                px = model.markets[mid].px
                tid = trader.num_trades + 1
                yield Trade(
                    mid,
                    amt,
                    side,
                    px,
                    model.schedule.steps,
                    tid,
                    None,
                    )
        return


class LongEnter(Enter):
    @staticmethod
    def get_side(mid, trader, model) -> int:
        return np.random.choice([1, 1,-1])

class AlwaysLongEnter(Enter):
    @staticmethod
    def get_side(mid, trader, model) -> int:
        return 1

class SlowExit(Exit):
    @staticmethod
    def exit_signal(mid, trader, model) -> bool:
       # import ipdb; ipdb.set_trace()
        if trader.earned_wealth - trader.mark_pos(trader.positions[mid]) <= 0:
            return True
        if np.random.ranf() < .1:
            return True
        return False#return np.r


class LongStrategy(Strategy):
    def __init__(self, LongEnter, Exit):
        self.Enter = LongEnter
        self.Exit = Exit
