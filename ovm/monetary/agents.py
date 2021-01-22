import logging
import typing as tp
import numpy as np

from mesa import Agent

from ovm.debug_level import PERFORM_DEBUG_LOGGING, PERFORM_INFO_LOGGING
from ovm.monetary.logs import console_log

from ovm.tickers import (
    USD_TICKER,
    OVL_TICKER,
    OVL_USD_TICKER
)

# set up logging
logger = logging.getLogger(__name__)


class MonetaryAgent(Agent):
    """
    An agent ... these are the arbers with stop losses.
    Add in position hodlers as a different agent
    later (maybe also with stop losses)
    """
    from ovm.monetary.model import MonetaryModel
    from ovm.monetary.markets import MonetaryFMarket

    def __init__(
        self,
        unique_id: int,
        model: MonetaryModel,
        fmarket: MonetaryFMarket,
        inventory: tp.Dict[str, float],
        pos_max: float = 0.2,
        deploy_max: float = 0.95,
        slippage_max: float = 0.02,
        leverage_max: float = 1.0,
        trade_delay: int = 4*10,
        size_increment: float = 0.1,
        min_edge: float = 0.0,
        max_edge: float = 0.1, # max deploy at 10% edge
        funding_multiplier: float = 1.0, # applied to funding cost when considering exiting position
        min_funding_unwind: float = 0.001, # start unwind when funding reaches .1% against position
        max_funding_unwind: float = 0.02 # unwind immediately when funding reaches 2% against position
    ):  # TODO: Fix constraint issues? => related to liquidity values we set ... do we need to weight liquidity based off vol?
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        self.fmarket = fmarket  # each 'trader' focuses on one market for now
        self.wealth = model.base_wealth  # in ovl
        self.inventory = inventory
        self.locked = 0
        self.pos_max = pos_max
        self.deploy_max = deploy_max
        self.slippage_max = slippage_max
        self.leverage_max = leverage_max
        self.trade_delay = trade_delay
        self.size_increment = size_increment
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.funding_multiplier = funding_multiplier
        self.min_funding_unwind = min_funding_unwind
        self.max_funding_unwind = max_funding_unwind
        self.last_trade_idx = 0
        self.positions: tp.Dict = {}  # { pos.id: MonetaryFPosition }
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
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"Trader agent {self.unique_id} activated")

        if self.wealth > 0 and self.locked / self.wealth < self.deploy_max:
            # Assume only make one trade per step ...
            self.trade()


class MonetaryKeeper(MonetaryAgent):
    def distribute_funding(self):
        # Sends funding payments on each agent's positions and updates px, py
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


class MonetaryLiquidator(MonetaryAgent):
    def scope_liquidations(self):
        # Finds a position to liquidate, then liquidates it
        idx = self.model.schedule.steps

        for pid, pos in self.fmarket.positions.items():
            side=1 if pos.long else -1
            open_position_notional = pos.amount*pos.leverage*(1 + \
                side*(self.fmarket.price - pos.lock_price)/pos.lock_price)
            value = pos.amount*(1 + \
                pos.leverage*side*(self.fmarket.price - pos.lock_price)/pos.lock_price)
            open_leverage = open_position_notional/value
            open_margin = 1/open_leverage
            maintenance_margin = self.fmarket.maintenance/pos.leverage
            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"Checking if liquidatable ... pid {pos.id}, amount {pos.amount}, leverage {pos.leverage}, long {pos.long}, lock_price {pos.lock_price}, market_price {self.fmarket.price}, trader id {pos.trader.unique_id}")
                logger.debug(f"Open leverage {open_leverage}, leverage {pos.leverage}, open margin {open_margin}, maintenance margin {maintenance_margin}")
                logger.debug(f"Is liquidatable? {self.fmarket.liquidatable(pid)}")
            if self.fmarket.liquidatable(pid) and pos.amount > 0.0:
                if PERFORM_DEBUG_LOGGING:
                    logger.debug("Liquidating ...")
                    logger.debug(f"self.inventory['OVL'] -> {self.inventory['OVL']}")
                reward = self.fmarket.liquidate(pid)
                self.inventory[OVL_TICKER] += reward
                self.wealth += reward
                self.last_trade_idx = self.model.schedule.steps
                pos.trader.locked -= pos.amount
                pos.trader.wealth -= pos.amount
                if PERFORM_DEBUG_LOGGING:
                    logger.debug("Liquidated ...")
                    logger.debug(f"self.inventory['OVL'] -> {self.inventory['OVL']}")
                    logger.debug(f"self.wealth -> {self.wealth}")
                    logger.debug(f"pos.trader.locked -> {pos.trader.locked}")
                    logger.debug(f"pos.trader.wealth -> {pos.trader.wealth}")

                return

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        # NOTE: liquidated positions act like market orders, so
        # liquidators act like traders in a sense
        idx = self.model.schedule.steps
        # Allow only one trader to trade on a market per block.
        # Add in a trade delay to simulate cooldown due to gas.
        if (self.fmarket.last_trade_idx != idx) and \
           (self.last_trade_idx == 0 or (idx - self.last_trade_idx) > self.trade_delay):
            self.scope_liquidations()


class MonetaryArbitrageur(MonetaryAgent):
    # TODO: super().__init__() with an arb min for padding, so doesn't trade if can't make X% locked in

    def _unwind_positions(self):
        # For now just assume all positions unwound at once (even tho unrealistic)
        # TODO: rebalance inventory on unwind!
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        sprice_ovlusd = self.model.sims[OVL_USD_TICKER][idx]
        for pid, pos in self.positions.items():
            fees = self.fmarket.fees(pos.amount, build=False, long=(
                not pos.long), leverage=pos.leverage)
            _, ds = self.fmarket.unwind(pos.amount, pid)
            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"Arb._unwind_positions: Unwinding position {pid} on {self.fmarket.unique_id}")
                logger.debug(f"Unwound: ds -> {ds}")

            self.inventory[OVL_TICKER] += pos.amount + ds - fees
            self.locked -= pos.amount
            self.wealth += ds - fees
            self.last_trade_idx = self.model.schedule.steps

            # Counter the futures trade on spot to unwind the arb
            # TODO: Have the spot market counter trades wrapped in SMarket class properly (clean this up)
            if pos.long is not True:
                spot_sell_amount = pos.amount*pos.leverage*sprice_ovlusd/sprice
                spot_sell_fees = min(
                    spot_sell_amount*self.fmarket.base_fee, pos.amount)
                spot_sell_received = (spot_sell_amount - spot_sell_fees)*sprice
                # TODO: this is wrong because of the leverage! fix
                self.inventory[self.fmarket.base_currency] -= spot_sell_amount
                self.inventory[USD_TICKER] += spot_sell_received
                if PERFORM_DEBUG_LOGGING:
                    logger.debug("Arb._unwind_positions: Selling base curr on spot to unwind arb ...")
                    logger.debug(f"Arb._unwind_positions: spot sell amount (OVL) -> {pos.amount}")
                    logger.debug(f"Arb._unwind_positions: spot sell amount ({self.fmarket.base_currency}) -> {spot_sell_amount}")
                    logger.debug(f"Arb._unwind_positions: spot sell fees ({self.fmarket.base_currency}) -> {spot_sell_fees}")
                    logger.debug(f"Arb._unwind_positions: spot sell received (USD) -> {spot_sell_received}")
                    logger.debug(f"Arb._unwind_positions: inventory -> {self.inventory}")
            else:
                spot_buy_amount = pos.amount*pos.leverage*sprice_ovlusd
                spot_buy_fees = min(
                    spot_buy_amount*self.fmarket.base_fee, pos.amount)
                spot_buy_received = (spot_buy_amount - spot_buy_fees)/sprice
                self.inventory[USD_TICKER] -= spot_buy_amount
                self.inventory[self.fmarket.base_currency] += spot_buy_received
                if PERFORM_DEBUG_LOGGING:
                    logger.debug("Arb._unwind_positions: Buying base curr on spot to lock in arb ...")
                    logger.debug(f"Arb._unwind_positions: spot buy amount (OVL) -> {pos.amount}")
                    logger.debug(f"Arb._unwind_positions: spot buy amount (USD) -> {spot_buy_amount}")
                    logger.debug(f"Arb._unwind_positions: spot buy fees (USD) -> {spot_buy_fees}")
                    logger.debug(f"Arb._unwind_positions: spot buy received ({self.fmarket.base_currency}) -> {spot_buy_received}")
                    logger.debug(f"Arb._unwind_positions: inventory -> {self.inventory}")

        self.positions = {}

    def _unwind_next_position(self):
        # Get the next position from inventory to unwind for this timestep
        if len(self.positions.keys()) == 0:
            self.unwinding = False
            return
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f'Arb._unwind_next_position: positions (prior) {self.positions}')
            logger.debug(f'Arb._unwind_next_position: locked (prior) {self.locked}')

        pid = list(self.positions.keys())[0]
        pos = self.positions[pid]
        _, ds = self.fmarket.unwind(pos.amount, pid)
        self.locked -= pos.amount
        self.last_trade_idx = self.model.schedule.steps
        del self.positions[pid]
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f'Arb._unwind_next_position: positions (updated) {self.positions}')
            logger.debug(f'Arb._unwind_next_position: locked (updated) {self.locked}')

    def trade(self):
        # If market futures price > spot then short, otherwise long
        # Calc the slippage first to see if worth it
        # TODO: Check for an arb opportunity. If exists, trade it ... bet Y% of current wealth on the arb ...
        # Get ready to arb current spreads
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        sprice_ovlusd = self.model.sims[OVL_USD_TICKER][idx]
        fprice = self.fmarket.price

        # TODO: Check arbs are making money on the spot .... Implement spot USD basis

        # TODO: Either wait for funding to unwind OR unwind once
        # reach wealth deploy_max and funding looks to be dried up?

        # Simple for now: tries to enter a pos_max amount of position if it wouldn't
        # breach the deploy_max threshold
        # TODO: make smarter, including thoughts on capturing funding (TWAP'ing it as well) => need to factor in slippage on spot (and have a spot market ...)
        # TODO: ALSO, when should arbitrageur exit their positions? For now, assume at funding they do (again, dumb) => Get dwasse comments here to make smarter
        # TODO: Add in slippage bounds for an order
        # TODO: Have arb bot determine position size dynamically needed to get price close to spot value (scale down size ...)
        # TODO: Have arb bot unwind all prior positions once deploys certain amount (or out of wealth)
        amount = self.pos_max*self.wealth
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"Arb.trade: Arb bot {self.unique_id} has {self.wealth-self.locked} OVL left to deploy")

        if self.locked + amount < self.deploy_max*self.wealth:
            if sprice > fprice:
                fees = self.fmarket.fees(amount, build=True, long=True, leverage=self.leverage_max)
                slippage = self.fmarket.slippage(amount-fees,
                                                 build=True,
                                                 long=True,
                                                 leverage=self.leverage_max)

                if PERFORM_DEBUG_LOGGING:
                    logger.debug(f"Arb.trade: Checking if long position on {self.fmarket.unique_id} is profitable after slippage ....")
                    logger.debug(f"Arb.trade: fees -> {fees}")
                    logger.debug(f"Arb.trade: slippage -> {slippage}")
                    logger.debug(f"Arb.trade: arb profit opp % -> {sprice/(fprice * (1+slippage)) - 1.0}")

                if self.slippage_max > abs(slippage) and sprice > fprice * (1+slippage) \
                    and sprice/(fprice * (1+slippage)) - 1.0 > 0.005: # TODO: arb_min on the RHS here instead of hard coded 0.005 = 0.5%
                    # enter the trade to arb
                    pos = self.fmarket.build(amount, long=True, leverage=self.leverage_max, trader=self)
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug("Arb.trade: Entered long arb trade w pos params ...")
                        logger.debug(f"Arb.trade: pos.amount -> {pos.amount}")
                        logger.debug(f"Arb.trade: pos.long -> {pos.long}")
                        logger.debug(f"Arb.trade: pos.leverage -> {pos.leverage}")
                        logger.debug(f"Arb.trade: pos.lock_price -> {pos.lock_price}")

                    self.positions[pos.id] = pos
                    self.inventory[OVL_TICKER] -= pos.amount + fees
                    self.locked += pos.amount
                    self.wealth -= fees
                    self.last_trade_idx = idx

                    # Counter the futures trade on spot with sell to lock in the arb
                    # TODO: Check never goes negative and eventually implement with a spot CFMM
                    # TODO: send fees to spot market CFMM ... (amount - fees)
                    spot_sell_amount = pos.amount*pos.leverage*sprice_ovlusd/sprice
                    # assume same as futures fees
                    spot_sell_fees = min(
                        spot_sell_amount*self.fmarket.base_fee, pos.amount)
                    spot_sell_received = (
                        spot_sell_amount - spot_sell_fees)*sprice

                    self.inventory[self.fmarket.base_currency] -= spot_sell_amount
                    self.inventory[USD_TICKER] += spot_sell_received

                    if PERFORM_DEBUG_LOGGING:
                        logger.debug("Arb.trade: Selling base curr on spot to lock in arb ...")
                        logger.debug(f"Arb.trade: spot sell amount (OVL) -> {pos.amount}")
                        logger.debug(f"Arb.trade: spot sell amount ({self.fmarket.base_currency}) -> {spot_sell_amount}")
                        logger.debug(f"Arb.trade: spot sell fees ({self.fmarket.base_currency}) -> {spot_sell_fees}")
                        logger.debug(f"Arb.trade: spot sell received (USD) -> {spot_sell_received}")
                        logger.debug(f"Arb.trade: inventory -> {self.inventory}")

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = - pos.amount * (sprice_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price + pos.amount * (price_t - lock_price)/lock_price
                    #           = pos.amount * [ - (sprice_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) + (price_t/lock_price - 1) ]
                    #           ~ pos.amount * [ - price_t/s_price + price_t/lock_price ] (if sprice_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = pos.amount * price_t * [ 1/lock_price - 1/s_price ]
                    # But s_price > lock_price, so PnL (approx) > 0
                    locked_in_approx = pos.amount * pos.leverage * \
                        (sprice/pos.lock_price - 1.0)
                    # TODO: incorporate fee structure!
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                        logger.debug(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*sprice_ovlusd}")

            elif sprice < fprice:
                fees = self.fmarket.fees(
                    amount, build=True, long=False, leverage=self.leverage_max)
                # should be negative ...
                slippage = self.fmarket.slippage(
                    amount-fees, build=True, long=False, leverage=self.leverage_max)
                if PERFORM_DEBUG_LOGGING:
                    logger.debug(f"Arb.trade: Checking if short position on {self.fmarket.unique_id} is profitable after slippage ....")
                    logger.debug(f"Arb.trade: fees -> {fees}")
                    logger.debug(f"Arb.trade: slippage -> {slippage}")
                    logger.debug(f"Arb.trade: arb profit opp % -> {1.0 - sprice/(fprice * (1+slippage))}")
                if self.slippage_max > abs(slippage) and sprice < fprice * (1+slippage) \
                    and 1.0 - sprice/(fprice * (1+slippage)) > 0.005: # TODO: arb_min on the RHS here instead of hard coded 0.005 = 0.5%
                    # enter the trade to arb
                    pos = self.fmarket.build(amount, long=False, leverage=self.leverage_max, trader=self)
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug("Arb.trade: Entered short arb trade w pos params ...")
                        logger.debug(f"Arb.trade: pos.amount -> {pos.amount}")
                        logger.debug(f"Arb.trade: pos.long -> {pos.long}")
                        logger.debug(f"Arb.trade: pos.leverage -> {pos.leverage}")
                        logger.debug(f"Arb.trade: pos.lock_price -> {pos.lock_price}")

                    self.positions[pos.id] = pos
                    self.inventory[OVL_TICKER] -= pos.amount + fees
                    self.locked += pos.amount
                    self.wealth -= fees
                    self.last_trade_idx = idx

                    # Counter the futures trade on spot with buy to lock in the arb
                    # TODO: Check never goes negative and eventually implement with a spot CFMM
                    # TODO: send fees to spot market CFMM ...
                    # TODO: FIX THIS FOR LEVERAGE SINCE OWING DEBT ON SPOT (and not accounting for it properly) -> Fine with counter unwind ultimately in long run
                    spot_buy_amount = pos.amount*pos.leverage*sprice_ovlusd
                    spot_buy_fees = min(
                        spot_buy_amount*self.fmarket.base_fee, pos.amount)
                    spot_buy_received = (
                        spot_buy_amount - spot_buy_fees)/sprice
                    self.inventory[USD_TICKER] -= spot_buy_amount
                    self.inventory[self.fmarket.base_currency] += spot_buy_received

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = pos.amount * (sprice_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price - pos.amount * (price_t - lock_price)/lock_price
                    #           = pos.amount * [ (sprice_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) - (price_t/lock_price - 1) ]
                    #           ~ pos.amount * [ price_t/s_price - price_t/lock_price ] (if sprice_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = pos.amount * price_t * [ 1/s_price - 1/lock_price ]
                    # But s_price < lock_price, so PnL (approx) > 0
                    locked_in_approx = pos.amount * pos.leverage * \
                        (1.0 - sprice/pos.lock_price)
                    # TODO: incorporate fee structure!
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug("Arb.trade: Buying base curr on spot to lock in arb ...")
                        logger.debug(f"Arb.trade: spot buy amount (OVL) -> {pos.amount}")
                        logger.debug(f"Arb.trade: spot buy amount (USD) -> {spot_buy_amount}")
                        logger.debug(f"Arb.trade: spot buy fees (USD) -> {spot_buy_fees}")
                        logger.debug(f"Arb.trade: spot buy received ({self.fmarket.base_currency}) -> {spot_buy_received}")
                        logger.debug(f"Arb.trade: inventory -> {self.inventory}")
                        logger.debug(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                        logger.debug(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*sprice_ovlusd}")
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
        if (self.fmarket.last_trade_idx != idx) and \
           (self.last_trade_idx == 0 or (idx - self.last_trade_idx) > self.trade_delay):
            self.trade()


class MonetaryTrader(MonetaryAgent):
    def trade(self):
        pass


class MonetaryHolder(MonetaryAgent):
    def trade(self):
        pass


# Only attempts for opportunities above a certain edge threshold, and linearly scales into position according to the amount of edge available.
# Unwinds position according to edge adjusted for current funding rate * funding_multiplier (ideally, this would be predicted funding rate).
class MonetarySniper(MonetaryAgent):

    def _get_unwind_amount(self, funding_rate, current_size, long):
        # Assume negative funding favors longs
        if long and funding_rate > self.min_funding_unwind:
            effective_rate = min(funding_rate, self.max_funding_unwind)
            return current_size * funding_rate / effective_rate
        if -funding_rate > self.min_funding_unwind:
            effective_rate = min(-funding_rate, self.max_funding_unwind)
            return current_size * (-funding_rate) / effective_rate
        return 0.0

    def _unwind_positions(self):
        # TODO: rebalance inventory on unwind!
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        sprice_ovlusd = self.model.sims[OVL_USD_TICKER][idx]
        unwound_pids = []
        for pid, pos in self.positions.items():
            unwind_amount = self._get_unwind_amount(self.fmarket.funding(), pos.amount, pos.long)
            unwind_amount = min(pos.amount, unwind_amount)
            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"Arb._unwind_positions: Unwinding position {pid} on {self.fmarket.unique_id}; unwind amount {unwind_amount}")

            if unwind_amount == 0.0:
                continue

            fees = self.fmarket.fees(unwind_amount, build=False, long=(
                not pos.long), leverage=pos.leverage)
            pos, ds = self.fmarket.unwind(unwind_amount, pid)
            if pos == None:
                unwound_pids.append(pid)
                continue
            self.inventory[OVL_TICKER] += unwind_amount + ds - fees
            self.locked -= unwind_amount
            self.wealth += ds - fees
            self.last_trade_idx = self.model.schedule.steps

            # Counter the futures trade on spot to unwind the arb
            # TODO: Have the spot market counter trades wrapped in SMarket class properly (clean this up)
            if pos.long is not True:
                spot_sell_amount = unwind_amount*pos.leverage*sprice_ovlusd/sprice
                spot_sell_fees = min(
                    spot_sell_amount*self.fmarket.base_fee, unwind_amount)
                spot_sell_received = (spot_sell_amount - spot_sell_fees)*sprice

                # TODO: this is wrong because of the leverage! fix
                self.inventory[self.fmarket.base_currency] -= spot_sell_amount
                self.inventory[USD_TICKER] += spot_sell_received
                if PERFORM_DEBUG_LOGGING:
                    logger.debug("Arb._unwind_positions: Selling base curr on spot to unwind arb ...")
                    logger.debug(f"Arb._unwind_positions: spot sell amount (OVL) -> {unwind_amount}")
                    logger.debug(f"Arb._unwind_positions: spot sell amount ({self.fmarket.base_currency}) -> {spot_sell_amount}")
                    logger.debug(f"Arb._unwind_positions: spot sell fees ({self.fmarket.base_currency}) -> {spot_sell_fees}")
                    logger.debug(f"Arb._unwind_positions: spot sell received (USD) -> {spot_sell_received}")
                    logger.debug(f"Arb._unwind_positions: inventory -> {self.inventory}")
            else:
                spot_buy_amount = unwind_amount*pos.leverage*sprice_ovlusd
                spot_buy_fees = min(
                    spot_buy_amount*self.fmarket.base_fee, unwind_amount)
                spot_buy_received = (spot_buy_amount - spot_buy_fees)/sprice
                self.inventory[USD_TICKER] -= spot_buy_amount
                self.inventory[self.fmarket.base_currency] += spot_buy_received
                if PERFORM_DEBUG_LOGGING:
                    logger.debug("Arb._unwind_positions: Buying base curr on spot to lock in arb ...")
                    logger.debug(f"Arb._unwind_positions: spot buy amount (OVL) -> {unwind_amount}")
                    logger.debug(f"Arb._unwind_positions: spot buy amount (USD) -> {spot_buy_amount}")
                    logger.debug(f"Arb._unwind_positions: spot buy fees (USD) -> {spot_buy_fees}")
                    logger.debug(f"Arb._unwind_positions: spot buy received ({self.fmarket.base_currency}) -> {spot_buy_received}")
                    logger.debug(f"Arb._unwind_positions: inventory -> {self.inventory}")

            if pos.amount == unwind_amount:
                unwound_pids.append(pid)

        for pid in unwound_pids:
            self.positions.pop(pid)

    def _get_filled_price(self, price, amount, long):
        fees = self.fmarket.fees(amount, build=True, long=True, leverage=self.leverage_max)
        slippage = self.fmarket.slippage(amount-fees,
                                            build=True,
                                            long=True,
                                            leverage=self.leverage_max)

        fee_perc = fees/amount
        if long:
            return price * (1 + slippage + fee_perc)
        return price * (1 - slippage - fee_perc)

    def _get_size(self, sprice, fprice, max_size, long):
        # Assume min size is zero
        sizes = np.arange(self.size_increment*self.wealth, max_size, self.size_increment*self.wealth)
        edge_map = {}
        for size in sizes:
            filled_price = self._get_filled_price(fprice, size, long)
            if long:
                edge_map[self._get_effective_edge(sprice - filled_price, self.fmarket.funding(), long)] = size
            else:
                edge_map[self._get_effective_edge(filled_price - sprice, self.fmarket.funding(), long)] = size
        if len(edge_map.keys()) == 0:
            return 0.0
        best_size = edge_map[max(edge_map.keys())]
        return best_size

    def _get_effective_edge(self, raw_edge, funding_rate, long):
        # Assume negative funding favors longs
        funding_edge = raw_edge
        if long:
            funding_edge -= funding_rate * self.funding_multiplier
            return min(funding_edge, self.max_edge)
        funding_edge += funding_rate * self.funding_multiplier
        return min(funding_edge, self.max_edge)

    def trade(self):
        # If market futures price > spot then short, otherwise long
        # Calc the slippage first to see if worth it
        # TODO: Check for an arb opportunity. If exists, trade it ... bet Y% of current wealth on the arb ...
        # Get ready to arb current spreads
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        sprice_ovlusd = self.model.sims[OVL_USD_TICKER][idx]
        fprice = self.fmarket.price

        # TODO: Check arbs are making money on the spot .... Implement spot USD basis

        # TODO: Either wait for funding to unwind OR unwind once
        # reach wealth deploy_max and funding looks to be dried up?

        # Simple for now: tries to enter a pos_max amount of position if it wouldn't
        # breach the deploy_max threshold
        # TODO: make smarter, including thoughts on capturing funding (TWAP'ing it as well) => need to factor in slippage on spot (and have a spot market ...)
        # TODO: ALSO, when should arbitrageur exit their positions? For now, assume at funding they do (again, dumb) => Get dwasse comments here to make smarter
        # TODO: Add in slippage bounds for an order
        # TODO: Have arb bot determine position size dynamically needed to get price close to spot value (scale down size ...)
        # TODO: Have arb bot unwind all prior positions once deploys certain amount (or out of wealth)
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"Arb.trade: Arb bot {self.unique_id} has {self.wealth-self.locked} OVL left to deploy")

        available_size = self.wealth - self.locked
        if available_size > 0:
            if sprice > fprice:
                if PERFORM_DEBUG_LOGGING:
                    logger.debug(f"Arb.trade: Checking if long position on {self.fmarket.unique_id} is profitable after slippage ....")
                amount = self._get_size(sprice, fprice, available_size, True)
                # NOTE: this is hacky
                if amount == 0.0:
                    return self._unwind_positions()

                fees = self.fmarket.fees(amount, build=True, long=True, leverage=self.leverage_max)
                slippage = self.fmarket.slippage(amount-fees,
                                                 build=True,
                                                 long=True,
                                                 leverage=self.leverage_max)
                # This has a good amount of duplicate work; already have fees, slippage, edge calculated
                fill_price = self._get_filled_price(fprice, amount, True) # TODO: size of trade here
                edge = sprice - fill_price
                effective_edge = self._get_effective_edge(edge, self.fmarket.funding(), True) # TODO: current funding rate estimate goes here
                if effective_edge > self.min_edge:
                    deploy_fraction = effective_edge / self.max_edge
                    amount = deploy_fraction * available_size

                    # enter the trade to arb
                    pos = self.fmarket.build(amount, long=True, leverage=self.leverage_max, trader=self)
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug(f"Arb.trade: fees: {fees}; slippage: {slippage}; deploy fraction: {deploy_fraction}; amount: {amount}; fill price {fill_price}; edge {edge}; edge surplus {edge - self.min_edge}")
                        logger.debug("Arb.trade: Entered long arb trade w pos params ...")
                        logger.debug(f"Arb.trade: pos.amount -> {pos.amount}")
                        logger.debug(f"Arb.trade: pos.long -> {pos.long}")
                        logger.debug(f"Arb.trade: pos.leverage -> {pos.leverage}")
                        logger.debug(f"Arb.trade: pos.lock_price -> {pos.lock_price}")

                    self.positions[pos.id] = pos
                    self.inventory[OVL_TICKER] -= pos.amount + fees
                    self.locked += pos.amount
                    self.wealth -= fees
                    self.last_trade_idx = idx
                    # Counter the futures trade on spot with sell to lock in the arb
                    # TODO: Check never goes negative and eventually implement with a spot CFMM
                    # TODO: send fees to spot market CFMM ... (amount - fees)
                    spot_sell_amount = pos.amount*pos.leverage*sprice_ovlusd/sprice
                    # assume same as futures fees
                    spot_sell_fees = min(
                        spot_sell_amount*self.fmarket.base_fee, pos.amount)
                    spot_sell_received = (
                        spot_sell_amount - spot_sell_fees)*sprice

                    self.inventory[self.fmarket.base_currency] -= spot_sell_amount
                    self.inventory[USD_TICKER] += spot_sell_received
                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = - pos.amount * (sprice_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price + pos.amount * (price_t - lock_price)/lock_price
                    #           = pos.amount * [ - (sprice_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) + (price_t/lock_price - 1) ]
                    #           ~ pos.amount * [ - price_t/s_price + price_t/lock_price ] (if sprice_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = pos.amount * price_t * [ 1/lock_price - 1/s_price ]
                    # But s_price > lock_price, so PnL (approx) > 0
                    locked_in_approx = pos.amount * pos.leverage * \
                        (sprice/pos.lock_price - 1.0)
                    # TODO: incorporate fee structure!
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug("Arb.trade: Selling base curr on spot to lock in arb ...")
                        logger.debug(f"Arb.trade: spot sell amount (OVL) -> {pos.amount}")
                        logger.debug(f"Arb.trade: spot sell amount ({self.fmarket.base_currency}) -> {spot_sell_amount}")
                        logger.debug(f"Arb.trade: spot sell fees ({self.fmarket.base_currency}) -> {spot_sell_fees}")
                        logger.debug(f"Arb.trade: spot sell received (USD) -> {spot_sell_received}")
                        logger.debug(f"Arb.trade: inventory -> {self.inventory}")
                        logger.debug(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                        logger.debug(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*sprice_ovlusd}")
            elif sprice < fprice:
                if PERFORM_DEBUG_LOGGING:
                    logger.debug(f"Arb.trade: Checking if short position on {self.fmarket.unique_id} is profitable after slippage ....")

                amount = self._get_size(sprice, fprice, available_size, False)
                # NOTE: this is hacky
                if amount == 0.0:
                    return self._unwind_positions()

                fees = self.fmarket.fees(
                    amount, build=True, long=False, leverage=self.leverage_max)
                # should be negative ...
                slippage = self.fmarket.slippage(
                    amount-fees, build=True, long=False, leverage=self.leverage_max)
                # This has a good amount of duplicate work; already have fees, slippage, edge calculated
                fill_price = self._get_filled_price(fprice, amount, False)
                edge = fill_price - sprice
                effective_edge = self._get_effective_edge(edge, self.fmarket.funding(), False)
                if effective_edge > self.min_edge:
                    deploy_fraction = effective_edge / self.max_edge
                    amount = deploy_fraction * available_size
                    # enter the trade to arb
                    pos = self.fmarket.build(amount, long=False, leverage=self.leverage_max, trader=self)
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug(f"Arb.trade: fees: {fees}; slippage: {slippage}; deploy fraction: {deploy_fraction}; amount: {amount}; fill price {fill_price}; edge {edge}; edge surplus {edge - self.min_edge}")
                        logger.debug("Arb.trade: Entered short arb trade w pos params ...")
                        logger.debug(f"Arb.trade: pos.amount -> {pos.amount}")
                        logger.debug(f"Arb.trade: pos.long -> {pos.long}")
                        logger.debug(f"Arb.trade: pos.leverage -> {pos.leverage}")
                        logger.debug(f"Arb.trade: pos.lock_price -> {pos.lock_price}")

                    self.positions[pos.id] = pos
                    self.inventory[OVL_TICKER] -= pos.amount + fees
                    self.locked += pos.amount
                    self.wealth -= fees
                    self.last_trade_idx = idx

                    # Counter the futures trade on spot with buy to lock in the arb
                    # TODO: Check never goes negative and eventually implement with a spot CFMM
                    # TODO: send fees to spot market CFMM ...
                    # TODO: FIX THIS FOR LEVERAGE SINCE OWING DEBT ON SPOT (and not accounting for it properly) -> Fine with counter unwind ultimately in long run
                    spot_buy_amount = pos.amount*pos.leverage*sprice_ovlusd
                    spot_buy_fees = min(
                        spot_buy_amount*self.fmarket.base_fee, pos.amount)
                    spot_buy_received = (
                        spot_buy_amount - spot_buy_fees)/sprice
                    self.inventory[USD_TICKER] -= spot_buy_amount
                    self.inventory[self.fmarket.base_currency] += spot_buy_received

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = pos.amount * (sprice_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price - pos.amount * (price_t - lock_price)/lock_price
                    #           = pos.amount * [ (sprice_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) - (price_t/lock_price - 1) ]
                    #           ~ pos.amount * [ price_t/s_price - price_t/lock_price ] (if sprice_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = pos.amount * price_t * [ 1/s_price - 1/lock_price ]
                    # But s_price < lock_price, so PnL (approx) > 0
                    locked_in_approx = pos.amount * pos.leverage * \
                        (1.0 - sprice/pos.lock_price)
                    # TODO: incorporate fee structure!
                    if PERFORM_DEBUG_LOGGING:
                        logger.debug("Arb.trade: Buying base curr on spot to lock in arb ...")
                        logger.debug(f"Arb.trade: spot buy amount (OVL) -> {pos.amount}")
                        logger.debug(f"Arb.trade: spot buy amount (USD) -> {spot_buy_amount}")
                        logger.debug(f"Arb.trade: spot buy fees (USD) -> {spot_buy_fees}")
                        logger.debug(f"Arb.trade: spot buy received ({self.fmarket.base_currency}) -> {spot_buy_received}")
                        logger.debug(f"Arb.trade: inventory -> {self.inventory}")
                        logger.debug(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                        logger.debug(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*sprice_ovlusd}")
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
        if (self.fmarket.last_trade_idx != idx) and \
           (self.last_trade_idx == 0 or (idx - self.last_trade_idx) > self.trade_delay):
            self.trade()
