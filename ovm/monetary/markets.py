from dataclasses import dataclass
import logging
import typing as tp
import uuid

import numpy as np

# nx, ny: OVL locked in x and y token
# dn: amount added to either bucket for the long/short position

from ovm.debug_level import DEBUG_LEVEL

# set up logging
logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class MonetaryFPosition:
    futures_market_ticker: str
    lock_price: float = 0.0
    amount_of_ovl_locked: float = 0.0  # this is non-negative
    long: bool = True
    leverage: float = 1.0

    def __post_init__(self):
        self.id = uuid.uuid4()

    @property
    def directional_size(self):
        if self.long:
            return self.amount_of_ovl_locked * self.leverage
        else:
            return -self.amount_of_ovl_locked * self.leverage


class MonetaryFMarket:  # This is Overlay
    from ovm.monetary.model import MonetaryModel

    def __init__(self,
                 unique_id: str,
                 nx: float,
                 ny: float,
                 px: float,
                 py: float,
                 base_fee: float,
                 max_leverage: float,
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
        self.model = model
        self.positions: tp.Dict[tp.Any, MonetaryFPosition] = {} # { id: [MonetaryFPosition] }
        self.base_currency = unique_id[:-len("-USD")]
        self.locked_long = 0.0  # Total OVL locked in long positions
        self.locked_short = 0.0  # Total OVL locked in short positions
        self.cum_locked_long = 0.0
        self.cum_locked_long_time_step = 0
        self.cum_locked_short = 0.0  # Used for time-weighted open interest on a side within sampling period
        self.cum_locked_short_time_step = 0
        self.last_cum_locked_long = 0.0
        self.last_cum_locked_short = 0.0
        self.cum_price = self.x / self.y
        self.cum_price_time_step = 0
        self.last_cum_price = self.x / self.y
        self.last_liquidity = model.liquidity # For liquidity adjustments
        self.last_funding_time_step = 0
        self.last_trade_time_step = 0
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"Init'ing FMarket {self.unique_id}")
            logger.debug(f"FMarket {self.unique_id} has x = {self.x}")
            logger.debug(f"FMarket {self.unique_id} has nx={self.nx} OVL")
            logger.debug(f"FMarket {self.unique_id} has y={self.y}")
            logger.debug(f"FMarket {self.unique_id} has ny={self.ny} OVL")
            logger.debug(f"FMarket {self.unique_id} has k={self.k}")

    @property
    def price(self) -> float:
        return self.x / self.y

    def _update_cum_price(self):
        # TODO: cum_price and time_elapsed setters ...
        # TODO: Need to check that this is the last swap for given timestep ... (slightly different than Uniswap in practice)
        current_time_step = self.model.schedule.steps
        if current_time_step > self.cum_price_time_step:  # and last swap for current_time_step ...
            self.cum_price += (current_time_step - self.cum_price_time_step) * self.price
            self.cum_price_time_step = current_time_step

    def _update_cum_locked_long(self):
        current_time_step = self.model.schedule.steps
        if current_time_step > self.cum_locked_long_time_step:
            self.cum_locked_long += (current_time_step - self.cum_locked_long_time_step) * self.locked_long
            self.cum_locked_long_time_step = current_time_step

    def _update_cum_locked_short(self):
        current_time_step = self.model.schedule.steps
        if current_time_step > self.cum_locked_short_time_step:
            self.cum_locked_short += (current_time_step - self.cum_locked_short_time_step) * self.locked_short
            self.cum_locked_short_time_step = current_time_step

    def _impose_fees(self,
                     dn: float,
                     build: float,
                     long: float,
                     leverage: float):
        # Impose fees, burns portion, and transfers rest to treasury
        size = dn*leverage
        fees = min(size*self.base_fee, dn)

        # Burn 50% and other 50% send to treasury
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"Burning ds={0.5*fees} OVL from total supply")
        self.model.supply_of_ovl -= 0.5 * fees
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
        if (build and long) or (not build and not long):
            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)
            assert dy < self.y, "slippage: Not enough liquidity in self.y for swap"
            slippage = ((self.x+dx)/(self.y-dy) - self.price) / self.price
        else:
            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)
            assert dx < self.x, "slippage: Not enough liquidity in self.x for swap"
            slippage = ((self.x-dx)/(self.y+dy) - self.price) / self.price
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
        if (build and long) or (not build and not long):
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug("dn = +px*dx")
            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)
            assert dy < self.y, "_swap: Not enough liquidity in self.y for swap"
            assert dy/self.py < self.ny, "_swap: Not enough liquidity in self.ny for swap"
            avg_price = self.k / (self.x * (self.x+dx))
            self.x += dx
            self.nx += dx/self.px
            self.y -= dy
            self.ny -= dy/self.py
        else:
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug("dn = -px*dx")
            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)
            assert dx < self.x, "_swap: Not enough liquidity in self.x for swap"
            assert dx/self.px < self.nx, "_swap: Not enough liquidity in self.nx for swap"
            avg_price = self.k / (self.x * (self.x-dx))
            self.y += dy
            self.ny += dy/self.py
            self.x -= dx
            self.nx -= dx/self.px

        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"_swap: {'Built' if build else 'Unwound'} {'long' if long else 'short'} "
                         f"position on {self.unique_id} of size {dn*leverage} OVL "
                         f"at avg price of {1/avg_price}, with lock price {self.price}")

            logger.debug(f"_swap: Percent diff bw avg and lock price is "
                         f"{100*(1/avg_price - self.price)/self.price}%")

            logger.debug(f"_swap: locked_long -> {self.locked_long} OVL")
            logger.debug(f"_swap: nx -> {self.nx}")
            logger.debug(f"_swap: x -> {self.x}")
            logger.debug(f"_swap: locked_short -> {self.locked_short} OVL")
            logger.debug(f"_swap: ny -> {self.ny}")
            logger.debug(f"_swap: y -> {self.y}")
        self._update_cum_price()
        self.last_trade_time_step = self.model.schedule.steps
        return self.price

    def build(self,
              dn: float,
              long: bool,
              leverage: float) -> MonetaryFPosition:
        # TODO: Factor in shares of lock pools for funding payment portions to work
        amount_of_ovl_locked = self._impose_fees(dn, build=True, long=long, leverage=leverage)
        price = self._swap(amount_of_ovl_locked, build=True, long=long, leverage=leverage)
        position = \
            MonetaryFPosition(futures_market_ticker=self.unique_id,
                              lock_price=price,
                              amount_of_ovl_locked=amount_of_ovl_locked,
                              long=long,
                              leverage=leverage)
        self.positions[position.id] = position

        # Lock into long/short pool last
        if long:
            self.locked_long += amount_of_ovl_locked
            self._update_cum_locked_long()
        else:
            self.locked_short += amount_of_ovl_locked
            self._update_cum_locked_short()
        return position

    def unwind(self,
               dn: float,
               position_id: uuid.UUID) -> tp.Tuple[MonetaryFPosition, float]:
        """
        Reduce or eliminate an existing position

        Args:
            dn: amount in OVL to reduce the position by
            position_id: position id

        Returns:

        """
        position = self.positions.get(position_id)
        if position is None:
            raise ValueError(f"No position with position_id={position_id} exists on market {self.unique_id}")
        elif position.amount_of_ovl_locked < dn:
            raise ValueError(f"Unwind amount {dn} is too large for locked position with position_id={position_id} "
                             f"amount {position.amount_of_ovl_locked}")

        # TODO: Account for pro-rata share of funding!
        # TODO: Fix this! something's wrong and I'm getting negative reserve amounts upon unwind :(
        # TODO: Locked long seems to go negative which is wrong. Why here?

        # Unlock from long/short pool first
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"unwind: dn = {dn}")
            logger.debug(f"unwind: position = {position.id}")
            logger.debug(f"unwind: locked_long = {self.locked_long}")
            logger.debug(f"unwind: locked_short = {self.locked_short}")
        # TODO: Fix for funding pro-rata logic .... for now just min it ...
        if position.long:
            dn = min(dn, self.locked_long)
            assert dn <= self.locked_long, "unwind: Not enough locked in self.locked_long for unwind"
            self.locked_long -= dn
            self._update_cum_locked_long()
        else:
            dn = min(dn, self.locked_short)
            assert dn <= self.locked_short, "unwind: Not enough locked in self.locked_short for unwind"
            self.locked_short -= dn
            self._update_cum_locked_short()

        amount = self._impose_fees(dn, build=False, long=position.long, leverage=position.leverage)
        price = self._swap(amount, build=False, long=position.long, leverage=position.leverage)
        side = 1 if position.long else -1

        # Mint/burn from total supply the profits/losses
        ds = amount * position.leverage * side * (price - position.lock_price) / position.lock_price
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"unwind: {'Minting' if ds > 0 else 'Burning'} ds={ds} OVL from total supply")
        self.model.supply_of_ovl += ds

        # Adjust position amounts stored
        if dn == position.amount_of_ovl_locked:
            del self.positions[position_id]
            position = None
        else:
            # Here the instance is mutated. Hence MonetaryFPosition cannot be frozen
            position.amount_of_ovl_locked -= amount
            self.positions[position_id] = position

        return position, ds

    def fund(self):
        # Pay out funding to each respective pool based on underlying market
        # oracle fetch
        # TODO: Fix for px, py sensitivity constant updates! => In practice, use TWAP from Sushi/Uni OVLETH pool for px and TWAP of underlying oracle fetch for p
        # Calculate the TWAP over previous sample
        current_time_step = self.model.schedule.steps
        if (current_time_step % self.model.sampling_interval != 0) or (current_time_step-self.model.sampling_interval < 0) or (current_time_step == self.last_funding_time_step):
            return

        # Calculate twap of oracle feed ... each step is value 1 in time weight
        cum_price_feed = np.sum(np.array(
            self.model.ticker_to_time_series_of_prices_map[self.unique_id][current_time_step - self.model.sampling_interval:current_time_step]
        ))
        twap_feed = cum_price_feed / self.model.sampling_interval
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"fund: Paying out funding for {self.unique_id}")
            logger.debug(f"fund: cum_price_feed = {cum_price_feed}")
            logger.debug(f"fund: sampling_interval = {self.model.sampling_interval}")
            logger.debug(f"fund: twap_feed = {twap_feed}")

        # Calculate twap of market ... update cum price value first
        self._update_cum_price()
        twap_market = (self.cum_price - self.last_cum_price) / self.model.sampling_interval
        self.last_cum_price = self.cum_price
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"fund: cum_price = {self.cum_price}")
            logger.debug(f"fund: last_cum_price = {self.last_cum_price}")
            logger.debug(f"fund: twap_market = {twap_market}")

        # Calculate twa open interest for each side over sampling interval
        self._update_cum_locked_long()
        twao_long = (self.cum_locked_long - self.last_cum_locked_long) / self.model.sampling_interval
        self.last_cum_locked_long = self.cum_locked_long
        self._update_cum_locked_short()
        twao_short = (self.cum_locked_short - self.last_cum_locked_short) / self.model.sampling_interval
        self.last_cum_locked_short = self.cum_locked_short

        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"fund: nx={self.nx}")
            logger.debug(f"fund: px={self.px}")
            logger.debug(f"fund: x={self.x}")
            logger.debug(f"fund: locked_long={self.locked_long}")
            logger.debug(f"fund: cum_locked_long={self.cum_locked_long}")
            logger.debug(f"fund: last_cum_locked_long={self.last_cum_locked_long}")
            logger.debug(f"fund: twao_long={twao_long}")
            logger.debug(f"fund: ny={self.ny}")
            logger.debug(f"fund: py={self.py}")
            logger.debug(f"fund: y={self.y}")
            logger.debug(f"fund: locked_short={self.locked_short}")
            logger.debug(f"fund: cum_locked_short={self.cum_locked_short}")
            logger.debug(f"fund: last_cum_locked_short={self.last_cum_locked_short}")
            logger.debug(f"fund: twao_short={twao_short}")

        # Mark the last funding current_time_step as now
        self.last_funding_time_step = current_time_step

        # Mint/burn funding
        funding = (twap_market - twap_feed) / twap_feed
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"fund: funding % -> {funding*100.0}%")
        if funding == 0.0:
            return
        elif funding > 0.0:
            funding = min(funding, 1.0)
            funding_long = min(twao_long*funding, self.locked_long) # can't have negative locked long
            funding_short = twao_short*funding
            self.model.supply_of_ovl += funding_short - funding_long
            self.locked_long -= funding_long
            self.locked_short += funding_short
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug(f"fund: Adding ds={funding_short - funding_long} OVL to total supply")
                logger.debug(f"fund: Adding ds={-funding_long} OVL to longs")
                logger.debug(f"fund: Adding ds={funding_short} OVL to shorts")
        else:
            funding = max(funding, -1.0)
            funding_long = abs(twao_long*funding)
            funding_short = min(abs(twao_short*funding), self.locked_short) # can't have negative locked short
            self.model.supply_of_ovl += funding_long - funding_short
            self.locked_long += funding_long
            self.locked_short -= funding_short
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug(f"fund: Adding ds={funding_long - funding_short} OVL to total supply")
                logger.debug(f"fund: Adding ds={funding_long} OVL to longs")
                logger.debug(f"fund: Adding ds={-funding_short} OVL to shorts")

        # Update virtual liquidity reserves
        # p_market = n_x*p_x/(n_y*p_y) = x/y; nx + ny = L/n (ignoring weighting, but maintain price ratio); px*nx = x, py*ny = y;\
        # n_y = (1/p_y)*(n_x*p_x)/(p_market) ... nx + n_x*(p_x/p_y)(1/p_market) = L/n
        # n_x = L/n * (1/(1 + (p_x/p_y)*(1/p_market)))
        liquidity = self.model.liquidity  # TODO: use liquidity_supply_emission ...
        liq_scale_factor = liquidity / self.last_liquidity
        self.last_liquidity = liquidity
        self.nx *= liq_scale_factor
        self.ny *= liq_scale_factor
        self.x = self.nx*self.px
        self.y = self.ny*self.py
        self.k = self.x * self.y
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"fund: Adjusting virtual liquidity constants for {self.unique_id}")
            logger.debug(f"fund: nx (prior) = {self.nx}")
            logger.debug(f"fund: ny (prior) = {self.ny}")
            logger.debug(f"fund: x (prior) = {self.x}")
            logger.debug(f"fund: y (prior) = {self.y}")
            logger.debug(f"fund: price (prior) = {self.price}")
            logger.debug(f"fund: last_liquidity = {self.last_liquidity}")
            logger.debug(f"fund: new liquidity = {liquidity}")
            logger.debug(f"fund: liquidity scale factor = {liq_scale_factor}")
            logger.debug(f"fund: nx (updated) = {self.nx}")
            logger.debug(f"fund: ny (updated) = {self.ny}")
            logger.debug(f"fund: x (updated) = {self.x}")
            logger.debug(f"fund: y (updated) = {self.y}")
            logger.debug(f"fund: price (updated... should be same) = {self.price}")

        # Calculate twap for ovlusd oracle feed to use in px, py adjustment
        cum_ovlusd_feed = np.sum(np.array(
            self.model.ticker_to_time_series_of_prices_map["OVL-USD"][current_time_step - self.model.sampling_interval:current_time_step]
        ))
        twap_ovlusd_feed = cum_ovlusd_feed / self.model.sampling_interval
        self.px = twap_ovlusd_feed # px = n_usd/n_ovl
        self.py = twap_ovlusd_feed/twap_feed # py = px/p
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"fund: Adjusting price sensitivity constants for {self.unique_id}")
            logger.debug(f"fund: cum_price_feed = {cum_ovlusd_feed}")
            logger.debug(f"fund: twap_ovlusd_feed = {twap_ovlusd_feed}")
            logger.debug(f"fund: px (updated) = {self.px}")
            logger.debug(f"fund: py (updated) = {self.py}")
            logger.debug(f"fund: price (updated... should be same) = {self.price}")


# ToDo: This code is not used yet. Do not delete but ignore for now.
class MonetarySMarket:  # This is UniSwap
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
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug("dn = +dx")
            dx = dn
            dy = self.y - self.k/(self.x + dx)
            self.x += dx
            self.y -= dy
        else:
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug("dn = -dx")
            dy = dn
            dx = self.x - self.k/(self.y + dy)
            self.y += dy
            self.x -= dx
        return self.price
