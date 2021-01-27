import logging
from typing import Any
from dataclasses import dataclass
import numpy as np
import uuid

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
                 maintenance: float, # this is times initial margin (i.e. 1/leverage); 0.0 < maintenance < 1.0
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
        self.locked_long = 0.0  # Total OVL locked in long positions
        self.locked_short = 0.0  # Total OVL locked in short positions
        self.cum_locked_long = 0.0
        self.cum_locked_long_idx = 0
        # Used for time-weighted open interest on a side within sampling period
        self.cum_locked_short = 0.0
        self.cum_locked_short_idx = 0
        self.last_cum_locked_long = 0.0
        self.last_cum_locked_short = 0.0
        self.cum_price = self.x / self.y
        self.cum_price_idx = 0
        self.last_cum_price = self.x / self.y
        self.last_liquidity = model.liquidity  # For liquidity adjustments
        self.last_funding_idx = 0
        self.last_trade_idx = 0
        self.last_funding_rate = 0
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"Init'ing FMarket {self.unique_id}")
            logger.debug(f"FMarket {self.unique_id} has x = {self.x}")
            logger.debug(f"FMarket {self.unique_id} has nx={self.nx} OVL")
            logger.debug(f"FMarket {self.unique_id} has px={self.px}")
            logger.debug(f"FMarket {self.unique_id} has y={self.y}")
            logger.debug(f"FMarket {self.unique_id} has ny={self.ny} OVL")
            logger.debug(f"FMarket {self.unique_id} has px={self.py}")
            logger.debug(f"FMarket {self.unique_id} has k={self.k}")

    @property
    def price(self) -> float:
        return self.x / self.y

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

    def fees(self,
             dn: float,
             build: bool,
             long: bool,
             leverage: float):
        size = dn*leverage
        return min(size*self.base_fee, dn)

    def slippage(self,
                 dn: float,
                 build: bool,
                 long: bool,
                 leverage: float):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        assert leverage < self.max_leverage, "slippage: leverage exceeds max_leverage"
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

    def _swap(self,
              dn: float,
              build: bool,
              long: bool,
              leverage: float):
        # k = (x + dx) * (y - dy)
        # dy = y - k/(x+dx)
        # TODO: dynamic k upon funding based off OVLETH liquidity changes
        assert leverage < self.max_leverage, "_swap: leverage exceeds max_leverage"
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

        self._update_cum_price()
        idx = self.model.schedule.steps
        self.last_trade_idx = idx
        return self.price

    def build(self,
              dn: float,
              long: bool,
              leverage: float,
              trader: Any = None):
        # TODO: Factor in shares of lock pools for funding payment portions to work
        amount = self._impose_fees(
            dn, build=True, long=long, leverage=leverage)
        price = self._swap(amount, build=True, long=long, leverage=leverage)
        pos = MonetaryFPosition(fmarket_ticker=self.unique_id,
                                lock_price=price,
                                amount=amount,
                                long=long,
                                leverage=leverage,
                                trader=trader)
        self.positions[pos.id] = pos

        # Lock into long/short pool last
        if long:
            self.locked_long += amount
            self._update_cum_locked_long()
        else:
            self.locked_short += amount
            self._update_cum_locked_short()
        return pos

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
        # TODO: Fix this! something's wrong and I'm getting negative reserve amounts upon unwind :(
        # TODO: Locked long seems to go negative which is wrong. Why here?

        # Unlock from long/short pool first
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"unwind: dn = {dn}")
            logger.debug(f"unwind: pos = {pos.id}")
            logger.debug(f"unwind: locked_long = {self.locked_long}")
            logger.debug(f"unwind: locked_short = {self.locked_short}")

        # TODO: Fix for funding pro-rata logic .... for now just min it ...
        if pos.long:
            dn=min(dn, self.locked_long)
            assert dn <= self.locked_long, "unwind: Not enough locked in self.locked_long for unwind"
            self.locked_long -= dn
            self._update_cum_locked_long()
        else:
            dn=min(dn, self.locked_short)
            assert dn <= self.locked_short, "unwind: Not enough locked in self.locked_short for unwind"
            self.locked_short -= dn
            self._update_cum_locked_short()

        amount=self._impose_fees(
            dn, build = False, long = pos.long, leverage = pos.leverage)
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

    def funding(self):
        # View for current estimate of the funding rate over current sampling period
        idx=self.model.schedule.steps
        dt=idx - self.last_funding_idx
        if dt == 0 or self.cum_price_idx == self.last_funding_idx:
            return 0.0

        # Calculate twap of oracle feed ... each step is value 1 in time weight
        cum_price_feed=np.sum(self.model.sims[self.unique_id][self.last_funding_idx:idx])
        twap_feed=cum_price_feed / dt

        # Calculate twap of market ... update cum price value first
        #twap_market = (self.cum_price - self.last_cum_price) / dt
        # TODO: twap market instead of actual price below since this is bad (but just for testing sniper for now)
        twap_market=self.price
        funding=(twap_market - twap_feed) / twap_feed

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"funding: Checking funding for {self.unique_id}")
            logger.debug(f"funding: cum_price_feed = {cum_price_feed}")
            logger.debug(f"funding: Time since last funding (dt) = {dt}")
            logger.debug(f"funding: twap_feed = {twap_feed}")
            logger.debug(f"funding: cum_price = {self.cum_price}")
            logger.debug(f"funding: last_cum_price = {self.last_cum_price}")
            logger.debug(f"funding: twap_market = {twap_market}")

        return funding

    def fund(self):
        # Pay out funding to each respective pool based on underlying market
        # oracle fetch
        # Calculate the TWAP over previous sample
        idx=self.model.schedule.steps
        if (idx % self.model.sampling_interval != 0) or (idx-self.model.sampling_interval < 0) or (idx == self.last_funding_idx):
            return

        # Calculate twap of oracle feed ... each step is value 1 in time weight
        cum_price_feed = \
            np.sum(self.model.sims[self.unique_id][idx - self.model.sampling_interval:idx])

        twap_feed=cum_price_feed / self.model.sampling_interval

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: Paying out funding for {self.unique_id}")
        #    logger.debug(f"fund: cum_price_feed = {cum_price_feed}")
        #    logger.debug(f"fund: sampling_interval = {self.model.sampling_interval}")
        #    logger.debug(f"fund: twap_feed = {twap_feed}")

        # Calculate twap of market ... update cum price value first
        #self._update_cum_price()

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: cum_price = {self.cum_price}")
        #    logger.debug(f"fund: last_cum_price = {self.last_cum_price}")

        #twap_market=(self.cum_price - self.last_cum_price) / \
        #             self.model.sampling_interval
        #self.last_cum_price=self.cum_price

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: twap_market = {twap_market}")

        # Calculate twa open interest for each side over sampling interval
        #self._update_cum_locked_long()

        #twao_long=(self.cum_locked_long - self.last_cum_locked_long) / \
        #           self.model.sampling_interval

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: nx={self.nx}")
        #    logger.debug(f"fund: px={self.px}")
        #    logger.debug(f"fund: x={self.x}")
        #    logger.debug(f"fund: locked_long={self.locked_long}")
        #    logger.debug(f"fund: cum_locked_long={self.cum_locked_long}")
        #    logger.debug(f"fund: last_cum_locked_long={self.last_cum_locked_long}")
        #    logger.debug(f"fund: twao_long={twao_long}")

        #self.last_cum_locked_long=self.cum_locked_long

        #self._update_cum_locked_short()

        #twao_short=(self.cum_locked_short - \
        #            self.last_cum_locked_short) / self.model.sampling_interval

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: ny={self.ny}")
        #    logger.debug(f"fund: py={self.py}")
        #    logger.debug(f"fund: y={self.y}")
        #    logger.debug(f"fund: locked_short={self.locked_short}")
        #    logger.debug(f"fund: cum_locked_short={self.cum_locked_short}")
        #    logger.debug(f"fund: last_cum_locked_short={self.last_cum_locked_short}")
        #    logger.debug(f"fund: twao_short={twao_short}")

        #self.last_cum_locked_short=self.cum_locked_short

        # Mark the last funding idx as now
        self.last_funding_idx=idx

        # Mint/burn funding
        #funding=(twap_market - twap_feed) / twap_feed
        #self.last_funding_rate=funding
        # print(f"fund: funding % -> {funding*100.0}%")
        #if funding == 0.0:
        #    return
        #elif funding > 0.0:
        #    funding=min(funding, 1.0)
        #    # can't have negative locked long
        #    funding_long=min(twao_long*funding, self.locked_long)
        #    funding_short=twao_short*funding
        #    self.model.supply += funding_short - funding_long
        #    self.locked_long -= funding_long
        #    self.locked_short += funding_short
        #    if PERFORM_DEBUG_LOGGING:
        #        logger.debug(f"fund: Adding ds={funding_short - funding_long} OVL to total supply")
        #        logger.debug(f"fund: Adding ds={-funding_long} OVL to longs")
        #        logger.debug(f"fund: Adding ds={funding_short} OVL to shorts")
        #else:
        #    funding=max(funding, -1.0)
        #    funding_long=abs(twao_long*funding)
        #    # can't have negative locked short
        #    funding_short=min(abs(twao_short*funding), self.locked_short)
        #    # print(f"fund: Adding ds={funding_long - funding_short} OVL to total supply")
        #    self.model.supply += funding_long - funding_short
        #    # print(f"fund: Adding ds={funding_long} OVL to longs")
        #    self.locked_long += funding_long
        #    # print(f"fund: Adding ds={-funding_short} OVL to shorts")
        #    self.locked_short -= funding_short
        #    if PERFORM_DEBUG_LOGGING:
        #        logger.debug("fund: Adding ds={funding_long - funding_short} OVL to total supply")
        #        logger.debug(f"fund: Adding ds={funding_long} OVL to longs")
        #        logger.debug(f"fund: Adding ds={-funding_short} OVL to shorts")

        # Update virtual liquidity reserves
        # p_market = n_x*p_x/(n_y*p_y) = x/y; nx + ny = L/n (ignoring weighting, but maintain price ratio); px*nx = x, py*ny = y;\
        # n_y = (1/p_y)*(n_x*p_x)/(p_market) ... nx + n_x*(p_x/p_y)(1/p_market) = L/n
        # n_x = L/n * (1/(1 + (p_x/p_y)*(1/p_market)))
        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"fund: Adjusting virtual liquidity constants for {self.unique_id}")
            logger.debug(f"fund: nx (prior) = {self.nx}")
            logger.debug(f"fund: ny (prior) = {self.ny}")
            logger.debug(f"fund: x (prior) = {self.x}")
            logger.debug(f"fund: y (prior) = {self.y}")
            logger.debug(f"fund: price (prior) = {self.price}")

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

        #if PERFORM_DEBUG_LOGGING:
        #    logger.debug(f"fund: nx (updated) = {self.nx}")
        #    logger.debug(f"fund: ny (updated) = {self.ny}")
        #    logger.debug(f"fund: x (updated) = {self.x}")
        #    logger.debug(f"fund: y (updated) = {self.y}")
        #    logger.debug(f"fund: price (updated... should be same) = {self.price}")

        # Calculate twap for ovl_quote oracle feed to use in px, py adjustment
        cum_ovl_quote_feed = \
            np.sum(self.model.sims[self.model.ovl_quote_ticker][idx-self.model.sampling_interval:idx])

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"fund: px (prior) = {self.px}")
            logger.debug(f"fund: py (prior) = {self.py}")
            logger.debug(f"fund: price (prior) = {self.price}")

        twap_ovl_quote_feed=cum_ovl_quote_feed / self.model.sampling_interval

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"fund: twap_ovl_quote_feed = {twap_ovl_quote_feed}")
            logger.debug(f"fund: twap_feed = {twap_feed}")

        self.px=twap_ovl_quote_feed  # px = n_quote/n_ovl
        self.py=twap_ovl_quote_feed/twap_feed  # py = px/p

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"fund: px (updated) = {self.px}")
            logger.debug(f"fund: py (updated) = {self.py}")
            logger.debug(f"fund: price (updated) = {self.price}")

        if PERFORM_DEBUG_LOGGING:
            logger.debug(f"fund: Adjusting price sensitivity constants for {self.unique_id}")
            logger.debug(f"fund: cum_price_feed = {cum_ovl_quote_feed}")
            logger.debug(f"fund: twap_ovl_quote_feed = {twap_ovl_quote_feed}")
            logger.debug(f"fund: px (updated) = {self.px}")
            logger.debug(f"fund: py (updated) = {self.py}")
            logger.debug(f"fund: price (updated... should be same) = {self.price}")

    def liquidatable(self, pid: uuid.UUID) -> bool:
        pos = self.positions.get(pid)
        if pos is None or (pos.long and pos.leverage == 1.0):
            return False

        # position initial margin fraction = 1/leverage
        side=1 if pos.long else -1
        open_position_notional = pos.amount*pos.leverage*(1 + \
            side*(self.price - pos.lock_price)/pos.lock_price)
        value = pos.amount*(1 + \
            pos.leverage*side*(self.price - pos.lock_price)/pos.lock_price)
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
            logger.debug(f"liquidatable: open_position_notional {open_position_notional}")
            logger.debug(f"liquidatable: value {value}")
            logger.debug(f"liquidatable: open_leverage {open_leverage}")
            logger.debug(f"liquidatable: open_margin {open_margin}")
            logger.debug(f"liquidatable: maintenance_margin {maintenance_margin}")
            logger.debug(f"liquidatable: open_margin < maintenance_margin {open_margin < maintenance_margin}")

        return open_margin < maintenance_margin

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

        # Anything left over after the burn is pos.amount - abs(ds) (leftover margin) ... split this
        margin = max(pos.amount - abs(ds), 0)

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
