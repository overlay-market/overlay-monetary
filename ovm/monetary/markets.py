import logging
from typing import Any
from dataclasses import dataclass
import numpy as np
import uuid
from collections import deque

from ovm.debug_level import PERFORM_DEBUG_LOGGING

# set up logging
logger = logging.getLogger(__name__)

# nx, ny: OVL locked in x and y token
# dn: amount added to either bucket for the long/short position


@dataclass(frozen=False)
class MonetaryFPosition:
    fmarket_ticker: str
    lock_price: float = 0.0
    amount: float = 0.0
    long: bool = True
    leverage: float = 1.0
    trader: Any = None
    # TODO: Link this to the trader who entered otherwise no way to know how wealth changes after liquidate

    def __post_init__(self):
        self.id = uuid.uuid4()

    @property
    def directional_size(self):
        if self.long:
            return self.amount * self.leverage
        else:
            return -self.amount * self.leverage


@dataclass(frozen=True)
class MonetaryMarketObservation:
    timestamp: int
    cum_price: float
    cum_locked_long: float = None
    cum_locked_short: float = None


class MonetaryFMarket:
    from ovm.monetary.model import MonetaryModel

    def __init__(self,
                 unique_id: str,
                 nx: float,
                 ny: float,
                 px: float,
                 py: float,
                 base_fee: float,
                 max_leverage: float,
                 liquidate_reward: float,
                 maintenance: float, # this is times initial margin (i.e. 1/leverage); 0.0 < maintenance < 1.0,
                 trade_limit: int, # number of trades allowed per idx
                 model: MonetaryModel):
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
        self.liquidate_reward = liquidate_reward
        self.maintenance = maintenance
        self.model = model
        self.positions = {}  # { id: [MonetaryFPosition] }
        self.base_currency = unique_id[:-len(f"-{model.quote_ticker}")]
        self.outstanding_amount_long = 0.0 # pos amounts (credits)
        self.outstanding_amount_short = 0.0 # pos amounts (credits)
        self.locked_long = 0.0  # Total OVL locked in long positions
        self.locked_short = 0.0  # Total OVL locked in short positions
        self.cum_locked_long = 0.0
        self.cum_locked_long_idx = 0
        # Used for time-weighted open interest on a side within sampling period
        self.cum_locked_short = 0.0
        self.cum_locked_short_idx = 0
        self.last_cum_locked_long = 0.0
        self.last_cum_locked_short = 0.0
        self.cum_price = 0.0
        self.cum_price_idx = 0
        # Used for sliding twaps: will be an array in Solidity with push() only for gas cost (see Uniswap v2 sliding impl)
        self.sliding_period_size = int(self.model.sampling_interval / self.model.sampling_twap_granularity) + 1
        self.sliding_fobservations = deque([
            MonetaryMarketObservation(
                timestamp=0,
                cum_price=0.0,
                cum_locked_long=0.0,
                cum_locked_short=0.0,
            )
            for i in range(self.sliding_period_size)
        ])
        self.sliding_sobservations = deque([
            MonetaryMarketObservation(timestamp=0, cum_price=0.0)
            for i in range(self.sliding_period_size)
        ])
        self.last_sliding_observation_idx = 0
        self.last_liquidity = model.liquidity  # For liquidity adjustments
        self.last_block_price = self.x / self.y
        self.last_fund_cum_price = 0.0
        self.last_twap_market = self.x / self.y
        self.last_funding_idx = 0
        self.last_trade_idx = 0
        self.trade_limit = trade_limit
        self.trades_in_idx = 0
        self.cum_funding_pay_long = 0.0
        self.cum_funding_pay_short = 0.0
        self.cum_funding_ds = 0.0
        self.last_funding_rate = 0
        if True: # PERFORM_DEBUG_LOGGING:
            logger.debug(f"Init'ing FMarket {self.unique_id}")
            logger.debug(f"FMarket {self.unique_id} has x = {self.x}")
            logger.debug(f"FMarket {self.unique_id} has nx={self.nx} OVL")
            logger.debug(f"FMarket {self.unique_id} has px={self.px}")
            logger.debug(f"FMarket {self.unique_id} has y={self.y}")
            logger.debug(f"FMarket {self.unique_id} has ny={self.ny} OVL")
            logger.debug(f"FMarket {self.unique_id} has py={self.py}")
            logger.debug(f"FMarket {self.unique_id} has k={self.k}")
            logger.debug(f"FMarket {self.unique_id} has px/py={self.px/self.py}")
            logger.debug(f"FMarket {self.unique_id} has price=x/y={self.price}")

    @property
    def price(self) -> float:
        return self.x / self.y

    @property
    def sliding_twap(self) -> float:
        oldest_obs = self.sliding_fobservations[0]
        newest_obs = self.sliding_fobservations[-1]

        # TODO: In solidity implementation this could cause problems in the first hour => solution could be leverage = 1.0 for first hour as cap
        if oldest_obs.timestamp == 0:
            return 0.0
        return (newest_obs.cum_price - oldest_obs.cum_price) / (newest_obs.timestamp - oldest_obs.timestamp)

    @property
    def sliding_twao_long(self) -> float:
        oldest_obs = self.sliding_fobservations[0]
        newest_obs = self.sliding_fobservations[-1]

        # TODO: In solidity implementation this could cause problems in the first hour => solution could be leverage = 1.0 for first hour as cap
        if oldest_obs.timestamp == 0:
            return 0.0
        return (newest_obs.cum_locked_long - oldest_obs.cum_locked_long) / (newest_obs.timestamp - oldest_obs.timestamp)

    @property
    def sliding_twao_short(self) -> float:
        oldest_obs = self.sliding_fobservations[0]
        newest_obs = self.sliding_fobservations[-1]

        # TODO: In solidity implementation this could cause problems in the first hour => solution could be leverage = 1.0 for first hour as cap
        if oldest_obs.timestamp == 0:
            return 0.0
        return (newest_obs.cum_locked_short - oldest_obs.cum_locked_short) / (newest_obs.timestamp - oldest_obs.timestamp)

    @property
    def sliding_twap_spot(self) -> float:
        oldest_obs = self.sliding_sobservations[0]
        newest_obs = self.sliding_sobservations[-1]

        # TODO: In solidity implementation this could cause problems in the first hour => solution could be leverage = 1.0 for first hour as cap
        if oldest_obs.timestamp == 0:
            return 0.0
        return (newest_obs.cum_price - oldest_obs.cum_price) / (newest_obs.timestamp - oldest_obs.timestamp)

    def _update_cum_price(self):
        # TODO: cum_price and time_elapsed setters ...
        # TODO: Need to check that this is the last swap for given timestep ... (slightly different than Uniswap in practice)
        idx = self.model.schedule.steps
        if idx > self.cum_price_idx:  # and last swap for idx ...
            self.cum_price += (idx - self.cum_price_idx) * self.price
            self.cum_price_idx = idx

    def _update_cum_locked_long(self):
        idx = self.model.schedule.steps
        if idx > self.cum_locked_long_idx:
            self.cum_locked_long += (idx
                                     - self.cum_locked_long_idx) * self.locked_long
            self.cum_locked_long_idx = idx

    def _update_cum_locked_short(self):
        idx = self.model.schedule.steps
        if idx > self.cum_locked_short_idx:
            self.cum_locked_short += (idx
                                      - self.cum_locked_short_idx) * self.locked_short
            self.cum_locked_short_idx = idx

    def _update_cum_locked(self):
        self._update_cum_locked_long()
        self._update_cum_locked_short()

    def _update_locked_amount(self, dn: float, build: bool, long: bool):
        if build and long:
            self.locked_long += dn
        elif build and not long:
            self.locked_short += dn
        elif not build and long:
            self.locked_long -= min(self.locked_long, dn)
        else:
            self.locked_short -= min(self.locked_short, dn)

    def _update_outstanding_amount(self, amount: float, build: bool, long: bool):
        if build and long:
            self.outstanding_amount_long += amount
        elif build and not long:
            self.outstanding_amount_short += amount
        elif not build and long:
            self.outstanding_amount_long -= min(amount, self.outstanding_amount_long)
        else:
            self.outstanding_amount_short -= min(amount, self.outstanding_amount_short)

    def _update_trades_in_idx(self):
        idx = self.model.schedule.steps
        if idx == self.last_trade_idx:
            self.trades_in_idx += 1
        else:
            self.last_trade_idx = idx
            self.trades_in_idx = 1

    def _update_sliding_observations(self):
        # Pop oldest obs of top of queue then append newest
        idx = self.model.schedule.steps
        if idx - self.last_sliding_observation_idx >= self.model.sampling_twap_granularity:
            # Update futures
            self.sliding_fobservations.popleft()
            self.sliding_fobservations.append(
                MonetaryMarketObservation(
                    timestamp=idx,
                    cum_price=self.cum_price,
                    cum_locked_long=self.cum_locked_long,
                    cum_locked_short=self.cum_locked_short,
                )
            )

            # Update spot
            self.sliding_sobservations.popleft()

            # Calc newest cum price
            # TODO: In Solidity, simply fetch cum price from spot (use our feed contract as a proxy)
            sobs_last = self.sliding_sobservations[-1]
            sprice = self.model.sims[self.unique_id][idx]
            scum_price = sobs_last.cum_price + \
                (idx - sobs_last.timestamp) * sprice
            self.sliding_sobservations.append(
                MonetaryMarketObservation(timestamp=idx,
                                           cum_price=scum_price)
            )

            self.last_sliding_observation_idx = idx

    def _sliding_observations_window(self) -> (int, int):
        return (
            self.sliding_fobservations[0].timestamp,
            self.sliding_fobservations[-1].timestamp
        )

    def _has_empty_sliding_observations(self):
        return self.sliding_fobservations[0].timestamp == 0

    def _impose_fees(self,
                     dn: float,
                     build: float,
                     long: float,
                     leverage: float):
        # Impose fees, burns portion, and transfers rest to treasury
        size = dn*leverage
        fees = min(size*self.base_fee, dn)
        assert fees >= 0.0, f"fees should be positive but are {fees} on build={build}"

        # Burn 50% and other 50% send to treasury
        self.model.supply -= 0.5*fees
        self.model.treasury += 0.5*fees

        return dn - fees

    def max_allowed_leverage(self, long: bool, lock_price: float):
        # Accounts for use of TWAP in liquidatable check ...
        # Anything above this amount should get liquidated immediately
        max_allowed = self.max_leverage
        if self.sliding_twap == 0.0:
            return max_allowed

        if long and self.sliding_twap < lock_price:
            max_allowed = max(min(
                self.max_leverage,
                self.maintenance + (1 - self.maintenance) / (1 - self.sliding_twap/lock_price)
            ), 1)
        elif not long and self.sliding_twap > lock_price:
            max_allowed = max(min(
                self.max_leverage,
                self.maintenance + (1 - self.maintenance) / (self.sliding_twap/lock_price - 1)
            ), 1)

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"FMarket.max_allowed_leverage: long -> {long}")
            logger.debug(f"FMarket.max_allowed_leverage: lock_price -> {lock_price}")
            logger.debug(f"FMarket.max_allowed_leverage: sliding_twap -> {self.sliding_twap}")
            logger.debug(f"FMarket.max_allowed_leverage: price -> {self.price}")
            logger.debug(f"FMarket.max_allowed_leverage: max_allowed_leverage -> {max_allowed}")

        return max_allowed

    def fees(self,
             dn: float,
             build: bool,
             long: bool,
             leverage: float):
        size = dn*leverage
        return min(size*self.base_fee, dn)

    def peek_price(self,
                   dn: float,
                   build: bool,
                   long: bool,
                   leverage: float,
                   fees: bool = True):
        amount = dn
        if fees:
            amount -= self.fees(dn, build, long, leverage)

        return (1 + self.slippage(amount, build, long, leverage)) * self.price

    def slippage(self,
                 dn: float,
                 build: bool,
                 long: bool,
                 leverage: float):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        slippage = 0.0
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"FMarket.slippage: market -> {self.unique_id}")
            logger.debug(f"FMarket.slippage: margin (OVL) -> {dn}")
            logger.debug(f"FMarket.slippage: leverage -> {leverage}")
            logger.debug(f"FMarket.slippage: is long? -> {long}")
            logger.debug(f"FMarket.slippage: build? -> {build}")

        if (build and long) or (not build and not long):
            if PERFORM_DEBUG_LOGGING:
                logger.debug("FMarket.slippage: dn = +px*dx; (x+dx)*(y-dy) = k")
                logger.debug(f"FMarket.slippage: px -> {self.px}")
                logger.debug(f"FMarket.slippage: py -> {self.py}")

            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)

            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"FMarket.slippage: reserves (Quote: x) -> {self.x}")
                logger.debug(f"FMarket.slippage: position impact (Quote: dx) -> {dx}")
                logger.debug(f"FMarket.slippage: position impact % (Quote: dx/x) -> {dx/self.x}")
                logger.debug(f"FMarket.slippage: reserves (Base: y) -> {self.y}")
                logger.debug(f"FMarket.slippage: position impact (Base: dy) -> {dy}")
                logger.debug(f"FMarket.slippage: position impact % (Base: dy/y) -> {dy/self.y}")

            assert dy < self.y, "slippage: Not enough liquidity in self.y for swap"
            slippage = ((self.x+dx)/(self.y-dy) - self.price) / self.price

            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"FMarket.slippage: price before -> {self.price}")
                logger.debug(f"FMarket.slippage: price after -> {(self.x+dx)/(self.y-dy)}")
                logger.debug(f"FMarket.slippage: slippage -> {slippage}")
        else:
            if PERFORM_DEBUG_LOGGING:
                logger.debug("FMarket.slippage: dn = -px*dx; (x-dx)*(y+dy) = k")
                logger.debug(f"FMarket.slippage: px -> {self.px}")
                logger.debug(f"FMarket.slippage: py -> {self.py}")

            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)

            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"FMarket.slippage: reserves (Quote: x) -> {self.x}")
                logger.debug(f"FMarket.slippage: position impact (Quote: dx) -> {dx}")
                logger.debug(f"FMarket.slippage: position impact % (Quote: dx/x) -> {dx/self.x}")
                logger.debug(f"FMarket.slippage: reserves (Base: y) -> {self.y}")
                logger.debug(f"FMarket.slippage: position impact (Base: dy) -> {dy}")
                logger.debug(f"FMarket.slippage: position impact % (Base: dy/y) -> {dy/self.y}")

            assert dx < self.x, "slippage: Not enough liquidity in self.x for swap"
            slippage = ((self.x-dx)/(self.y+dy) - self.price) / self.price

            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"FMarket.slippage: price before -> {self.price}")
                logger.debug(f"FMarket.slippage: price after -> {(self.x-dx)/(self.y+dy)}")
                logger.debug(f"FMarket.slippage: slippage -> {slippage}")

        return slippage

    def can_trade(self):
        idx = self.model.schedule.steps
        return (
            idx != self.last_trade_idx or
            self.trades_in_idx <= self.trade_limit
        )

    def _swap(self,
              dn: float,
              build: bool,
              long: bool,
              leverage: float):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        # TODO: dynamic k upon funding based off OVLETH liquidity changes
        avg_price = 0.0
        if (build and long) or (not build and not long):
            # print("dn = +px*dx")
            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)

            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"_swap: position size (OVL) -> {dn*leverage}")
                logger.debug(f"_swap: position impact (Quote: dx) -> {dx}")
                logger.debug(f"_swap: position impact (Base: dy) -> {dy}")

            assert dy < self.y, "_swap: Not enough liquidity in self.y for swap"
            assert dy/self.py < self.ny, "_swap: Not enough liquidity in self.ny for swap"
            avg_price = self.k / (self.x * (self.x+dx))
            self.x += dx
            self.nx += dx/self.px
            self.y -= dy
            self.ny -= dy/self.py
        else:
            # print("dn = -px*dx")
            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)

            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"_swap: position size (OVL) -> {dn*leverage}")
                logger.debug(f"_swap: position impact (Quote: dx) -> {dx}")
                logger.debug(f"_swap: position impact (Base: dy) -> {dy}")

            assert dx < self.x, "_swap: Not enough liquidity in self.x for swap"
            assert dx/self.px < self.nx, "_swap: Not enough liquidity in self.nx for swap"
            avg_price = self.k / (self.x * (self.x-dx))
            self.y += dy
            self.ny += dy/self.py
            self.x -= dx
            self.nx -= dx/self.px

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"_swap: {'Built' if build else 'Unwound'} {'long' if long else 'short'} position on {self.unique_id} of size {dn*leverage} OVL at avg price of {1/avg_price}, with lock price {self.price}")
            logger.debug(f"_swap: Percent diff bw avg and lock price is {100*(1/avg_price - self.price)/self.price}%")
            logger.debug(f"_swap: locked_long -> {self.locked_long} OVL")
            logger.debug(f"_swap: nx -> {self.nx}")
            logger.debug(f"_swap: x -> {self.x}")
            logger.debug(f"_swap: locked_short -> {self.locked_short} OVL")
            logger.debug(f"_swap: ny -> {self.ny}")
            logger.debug(f"_swap: y -> {self.y}")

        # Market cache updates
        self._update_cum_price()
        self._update_sliding_observations()
        self._update_trades_in_idx()

        return self.price

    def build(self,
              dn: float,
              long: bool,
              leverage: float,
              trader: Any = None):
        assert leverage <= self.max_leverage, "build: leverage exceeds max_allowed_leverage"
        amount = self._impose_fees(
            dn, build=True, long=long, leverage=leverage)

        # Update locked amount
        self._update_locked_amount(amount, build=True, long=long)

        # Update cumulative cache
        self._update_cum_locked()

        # Do the swap
        price = self._swap(amount, build=True, long=long, leverage=leverage)
        pos = MonetaryFPosition(fmarket_ticker=self.unique_id,
                                lock_price=price,
                                amount=amount,
                                long=long,
                                leverage=leverage,
                                trader=trader)
        self.positions[pos.id] = pos

        # Update outstanding pos amounts
        self._update_outstanding_amount(amount, build=True, long=long)

        return pos

    # TODO: Fix for open interest share of locked_long/short; pro rata share
    # of locked! (do we need to change nx/ny on funding pay given reserve skew?)
    #  => do this first, then assess funding stability and whether to shift
    def unwind(self,
               dn: float,
               pid: uuid.UUID):
        pos = self.positions.get(pid)
        if pos is None:
            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"No position with pid {pid} exists on market {self.unique_id}")
            return None, 0.0
        elif pos.amount < dn:
            if PERFORM_DEBUG_LOGGING:
                logger.debug(f"Unwind amount {dn} is too large for locked position with pid {pid} amount {pos.amount}")
            return None, 0.0

        # TODO: Account for pro-rata share of funding!
        # Unlock from long/short pool first
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"unwind: dn = {dn}")
            logger.debug(f"unwind: pos = {pos.id}")
            logger.debug(f"unwind: locked_long = {self.locked_long}")
            logger.debug(f"unwind: locked_short = {self.locked_short}")

        # Update outstanding pos amounts
        self._update_outstanding_amount(dn, build=False, long=pos.long)

        # TODO: Fix for funding pro-rata logic .... for now just min it ...
        #print(f"unwind: Getting pro rata amount locked ...")
        #print(f"unwind: dn = {dn}")
        #print(f"unwind: long = {pos.long}")
        #print(f"unwind: outstanding_amount_long = {self.outstanding_amount_long}")
        #print(f"unwind: outstanding_amount_short = {self.outstanding_amount_short}")
        #print(f"unwind: locked_long = {self.locked_long}")
        #print(f"unwind: locked_short = {self.locked_short}")

        # TODO: Fix this logic => screwing it up ...
        #amount = self.pro_rata_amount_locked(dn, long=pos.long)
        #print(f"unwind: amount = {amount}")
        amount = dn

        # Update locked amount
        self._update_locked_amount(amount, build=False, long=pos.long)

        # Update cumulative cache
        self._update_cum_locked()

        # Do the swap
        amount=self._impose_fees(
            amount, build = False, long = pos.long, leverage = pos.leverage)
        price=self._swap(amount, build = False,
                         long = pos.long, leverage = pos.leverage)
        side=1 if pos.long else -1

        # Mint/burn from total supply the profits/losses
        ds=amount * pos.leverage * side * \
            (price - pos.lock_price)/pos.lock_price

        # Cap to make sure system doesn't burn more than locked amount in pos
        if ds < 0:
            ds = max(ds, -amount)
        # print(f"unwind: {'Minting' if ds > 0 else 'Burning'} ds={ds} OVL from total supply")
        self.model.supply += ds

        # Adjust position amounts stored
        if dn == pos.amount:
            del self.positions[pid]
            pos=None
        else:
            pos.amount -= amount
            self.positions[pid]=pos

        return pos, ds

    # TODO: adjust liquidity for nx, ny based on funding ...
    # def _update_liquidity

    def funding(self):
        # View for current estimate of the funding rate over current sampling period
        idx=self.model.schedule.steps
        start_idx, end_idx = self._sliding_observations_window()
        dt = end_idx - start_idx
        if dt == 0 or self.sliding_twap_spot == 0.0:
            return 0.0

        # Estimate for twap
        return (self.sliding_twap - self.sliding_twap_spot) / self.sliding_twap_spot

    def fund(self):
        # Pay out funding to each respective pool based on underlying market
        # oracle fetch
        # TODO: funding reward for those that call this function! build into swap fee structure ...
        idx=self.model.schedule.steps

        # First update cache quantities for freshness
        self._update_cum_price()
        self._update_sliding_observations()

        start_idx, end_idx = self._sliding_observations_window()
        if (idx % self.model.sampling_interval != 0) or \
           (idx-self.model.sampling_interval < 0) or \
           self._has_empty_sliding_observations() or \
           start_idx == end_idx or \
           (idx == self.last_funding_idx):
            return

        # Calculate twap of oracle feed using timestamps from sliding observations
        if PERFORM_DEBUG_LOGGING:
            print(f"fund: start_idx => {start_idx}")
            print(f"fund: end_idx => {end_idx}")
            print(f"fund: sampling_interval => {self.model.sampling_interval}")
            print(f"fund: idx => {idx}")
            print(f"fund: last_funding_idx (prior) => {self.last_funding_idx}")
            print(f"fund: sliding fobservations => {self.sliding_fobservations}")

        # Mark the last funding idx as now
        self.last_funding_idx=idx
        funding = (self.sliding_twap - self.sliding_twap_spot) / self.sliding_twap_spot

        if PERFORM_DEBUG_LOGGING:
            print(f"fund: funding % (sliding) => {funding*100.0}%")
            print(f"fund: last_funding_idx (updated) => {self.last_funding_idx}")

        self.last_funding_rate=funding

        # Update virtual liquidity reserves
        # p_market = n_x*p_x/(n_y*p_y) = x/y; nx + ny = L/n (ignoring weighting, but maintain price ratio); px*nx = x, py*ny = y;\
        # n_y = (1/p_y)*(n_x*p_x)/(p_market) ... nx + n_x*(p_x/p_y)(1/p_market) = L/n
        # n_x = L/n * (1/(1 + (p_x/p_y)*(1/p_market)))
        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: Adjusting virtual liquidity constants for {self.unique_id}")
        #    logger.debug(f"fund: nx (prior) = {self.nx}")
        #    logger.debug(f"fund: ny (prior) = {self.ny}")
        #    logger.debug(f"fund: x (prior) = {self.x}")
        #    logger.debug(f"fund: y (prior) = {self.y}")
        #    logger.debug(f"fund: price (prior) = {self.price}")

        # TODO: use liquidity_supply_emission ...
        #liquidity=self.model.liquidity
        #liq_scale_factor=liquidity / self.last_liquidity

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: last_liquidity = {self.last_liquidity}")
        #    logger.debug(f"fund: new liquidity = {liquidity}")
        #    logger.debug(f"fund: liquidity scale factor = {liq_scale_factor}")

        #self.last_liquidity=liquidity
        #self.nx *= liq_scale_factor
        #self.ny *= liq_scale_factor
        #self.x=self.nx*self.px
        #self.y=self.ny*self.py
        #self.k=self.x * self.y

        # Fetch OVL-ETH FMarket and get twap_spot of feed to use in px, py adjustment
        ovl_quote_fmarket = self.model.fmarkets[self.model.ovl_quote_ticker]

        # Use twap for OVL-ETH and spot feed to reset price ("funding")
        # TODO: spin off into separate _update function for sensitivity coeffs
        twap_ovl_quote_market = ovl_quote_fmarket.sliding_twap
        twap_ovl_quote_feed = ovl_quote_fmarket.sliding_twap_spot
        twap_feed = self.sliding_twap_spot
        twap_market = self.sliding_twap

        #print(f"fund: twap_ovl_quote_feed (spot) = {twap_ovl_quote_feed}")
        #print(f"fund: twap_ovl_quote_market (futures) = {twap_ovl_quote_market}")
        #print(f"fund: twap_feed (spot) = {twap_feed}")
        #print(f"fund: twap_market (futures) = {twap_market}")

        # I can still do this twap futures fanciness, but

        # NOTE: these below only update when trading is happening
        if twap_ovl_quote_market != 0.0 and twap_ovl_quote_feed != 0.0 and \
           twap_market != 0.0 and twap_feed != 0.0:
            #print(f"fund: Adjusting price sensitivity constants for {self.unique_id}")
            #print(f"fund: Step idx {idx}")
            # print(f"fund: twap_ovl_quote_feed = {twap_ovl_quote_feed}")
            # print(f"fund: twap_feed = {twap_feed}")
            # print(f"fund: ovl_quote_feed = {self.model.sims[ovl_quote_fmarket.unique_id][idx]}")
            # print(f"fund: feed = {self.model.sims[self.unique_id][idx]}")

            #print(f"fund: px (prior) = {self.px}")
            #print(f"fund: py (prior) = {self.py}")
            #print(f"fund: nx (prior) = {self.nx}")
            #print(f"fund: ny (prior) = {self.ny}")
            #print(f"fund: x (prior) = {self.x}")
            #print(f"fund: y (prior) = {self.y}")
            #print(f"fund: k (prior) = {self.k}")
            #print(f"fund: price (prior) = {self.price}")
            price_prior = self.price
            p = self.px/self.py

            #print(f"fund: px (before) = {self.px}")
            #print(f"fund: py (before) = {self.py}")
            #print(f"fund: p = px/py (before) = {p}")

            # Adjust px by diff bw OVL-QUOTE spot twap and futures twap
            # self.px = self.px_old + ((self.px_new - self.px_old) / self.px_old) * self.px_old
            # NOTE: Do we actually want to do it this way or simply set it to the spot?
            ovl_quote_diff = (twap_ovl_quote_feed - twap_ovl_quote_market) / twap_ovl_quote_market
            #print(f"fund: twap_ovl_quote_feed (spot) = {twap_ovl_quote_feed}")
            #print(f"fund: twap_ovl_quote_market (futures) = {twap_ovl_quote_market}")
            #print(f"fund: ovl_quote_diff = {ovl_quote_diff}")
            self.px *= (1 + ovl_quote_diff)

            market_diff = (twap_feed - twap_market) / twap_market
            #print(f"fund: twap_feed (spot) = {twap_feed}")
            #print(f"fund: twap_market (futures) = {twap_market}")
            #print(f"fund: market_diff = {market_diff}")
            p *= (1 + market_diff)
            self.py = self.px/p

            #print(f"fund: px (after) = {self.px}")
            #print(f"fund: py (after) = {self.py}")
            #print(f"fund: p = px/py (after) = {p}")

            # NOTE: next two lines are OLD (this is TWAP spot direct resetting)
            #self.px=twap_ovl_quote_feed  # px = n_quote/n_ovl
            #self.py=twap_ovl_quote_feed/twap_feed  # py = px/p

            # This is funding: Have p = px/py => spot twap by resetting nx = ny
            # Choose midpoint between nx, ny to limit liquidity bump ups over time: think about it more in terms of placement on x*y=k curve
            n_mid = (self.nx + self.ny) / 2.0
            self.nx = n_mid
            self.ny = n_mid
            self.x = self.nx * self.px
            self.y = self.ny * self.py
            self.k = self.x*self.y

            # print(f"fund: px (updated) = {self.px}")
            # print(f"fund: py (updated) = {self.py}")
            #print(f"fund: nx (updated) = {self.nx}")
            #print(f"fund: ny (updated) = {self.ny}")
            #print(f"fund: x (updated) = {self.x}")
            #print(f"fund: y (updated) = {self.y}")
            #print(f"fund: k (updated) = {self.k}")
            #print(f"fund: price (updated) = {self.price}")
            #print(f"fund: price diff (%) = {(self.price - price_prior)/price_prior}")

    def liquidatable(self, pid: uuid.UUID) -> bool:
        pos = self.positions.get(pid)
        if pos is None or (pos.long and pos.leverage == 1.0) or self.sliding_twap == 0.0:
            return False

        # position initial margin fraction = 1/leverage
        # Use TWAP over last sampling interval so resistant to flash loans
        side=1 if pos.long else -1
        open_position_notional = pos.amount*pos.leverage*(1 + \
            side*(self.sliding_twap - pos.lock_price)/pos.lock_price)
        value = pos.amount*(1 + \
            pos.leverage*side*(self.sliding_twap - pos.lock_price)/pos.lock_price)
        open_leverage = open_position_notional/value
        open_margin = 1/open_leverage
        maintenance_margin = self.maintenance/pos.leverage

        if open_margin < maintenance_margin and PERFORM_DEBUG_LOGGING:
            logger.debug(f"liquidatable: pos {pid} is liquidatable ...")
            logger.debug(f"liquidatable: pos.fmarket_ticker {pos.fmarket_ticker}")
            logger.debug(f"liquidatable: pos.lock_price {pos.lock_price}")
            logger.debug(f"liquidatable: pos.amount {pos.amount}")
            logger.debug(f"liquidatable: pos.long {pos.long}")
            logger.debug(f"liquidatable: pos.leverage {pos.leverage}")
            logger.debug(f"liquidatable: pos.trader {pos.trader}")
            logger.debug(f"liquidatable: fmarket.price {self.price}")
            logger.debug(f"liquidatable: fmarket.sliding_twap {self.sliding_twap}")
            logger.debug(f"liquidatable: open_position_notional {open_position_notional}")
            logger.debug(f"liquidatable: value {value}")
            logger.debug(f"liquidatable: open_leverage {open_leverage}")
            logger.debug(f"liquidatable: open_margin {open_margin}")
            logger.debug(f"liquidatable: maintenance_margin {maintenance_margin}")
            logger.debug(f"liquidatable: open_margin < maintenance_margin {open_margin < maintenance_margin}")

        return open_margin < maintenance_margin

    def reward_to_liquidate(self, pid: uuid.UUID) -> float:
        if not self.liquidatable(pid):
            return 0.0

        pos = self.positions.get(pid)
        fees = self.fees(dn=pos.amount,
                         build=False,
                         long=pos.long,
                         leverage=pos.leverage)
        amount = pos.amount-fees
        unwind_price = self.peek_price(dn=amount,
                                       build=False,
                                       long=pos.long,
                                       leverage=pos.leverage,
                                       fees=False)

        # Mint/burn from total supply the profits/losses
        side = 1 if pos.long else -1
        ds = amount * pos.leverage * side * \
            (unwind_price - pos.lock_price)/pos.lock_price

        if ds > 0.0:
            return 0.0

        return abs(ds) * self.liquidate_reward

    def liquidate(self, pid: uuid.UUID) -> float:
        can = self.liquidatable(pid)
        pos = self.positions.get(pid)
        if pos is None or not can:
            return 0.0

        # Unwind but change supply back to original before unwind to factor in
        # reward to liquidator (then modify supply again) -> this is hacky
        _, ds = self.unwind(pos.amount, pid)
        if PERFORM_DEBUG_LOGGING:
            logger.debug("liquidated pos.amount", pos.amount)
            logger.debug("liqudated unwind ds", ds)
        self.model.supply -= ds

        # NOTE: ds should be negative
        assert ds <= 0.0, f"liquidate: position liquidation should result in burn of amount, ds={ds}"
        reward = abs(ds) * self.liquidate_reward

        # TODO: rethink this calculation so it's not a percent of the loss? or at least not as large

        # Anything left over after the burn is pos.amount - abs(ds) (leftover margin) ... split this
        margin = max(pos.amount - abs(ds), 0)
        assert margin >= 0.0, f"margin should be positive but are {margin} on liquidate"

        # Any maintenance margin should be split between burn and treasury
        self.model.treasury += 0.5 * margin
        self.model.supply -= (abs(ds) - reward + 0.5*margin)
        return reward


class MonetarySMarket:
    def __init__(self,
                 unique_id: str,
                 x: float,
                 y: float,
                 k: float):
        self.unique_id = unique_id
        self.x = x
        self.y = y
        self.k = k

    @property
    def price(self) -> float:
        return self.x / self.y

    def swap(self,
             dn: float,
             buy: bool = True) -> float:
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        if buy:
            # print("dn = +dx")
            dx = dn
            dy = self.y - self.k/(self.x + dx)
            self.x += dx
            self.y -= dy
        else:
            # print("dn = -dx")
            dy = dn
            dx = self.x - self.k/(self.y + dy)
            self.y += dy
            self.x -= dx
        return self.price
