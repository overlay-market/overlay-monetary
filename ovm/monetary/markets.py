import logging
from typing import Any
from dataclasses import dataclass
import numpy as np
import uuid

from ovm.monetary.logs import console_log

from ovm.tickers import (
    USD_TICKER,
    OVL_TICKER,
    OVL_USD_TICKER
)

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
    from model import MonetaryModel

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
        self.base_currency = unique_id[:-len("-USD")]
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
        console_log(logger, [
            f"Init'ing FMarket {self.unique_id}",
            f"FMarket {self.unique_id} has x = {self.x}",
            f"FMarket {self.unique_id} has nx={self.nx} OVL",
            f"FMarket {self.unique_id} has y={self.y}",
            f"FMarket {self.unique_id} has ny={self.ny} OVL",
            f"FMarket {self.unique_id} has k={self.k}",
        ])

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
        console_log(logger, [
            f"FMarket.slippage: market -> {self.unique_id}",
            f"FMarket.slippage: margin (OVL) -> {dn}",
            f"FMarket.slippage: leverage -> {leverage}",
            f"FMarket.slippage: is long? -> {long}",
            f"FMarket.slippage: build? -> {build}",
        ])
        if (build and long) or (not build and not long):
            console_log(logger, [
                "FMarket.slippage: dn = +px*dx; (x+dx)*(y-dy) = k",
                f"FMarket.slippage: px -> {self.px}",
                f"FMarket.slippage: py -> {self.py}"
            ])

            dx = self.px*dn*leverage
            dy = self.y - self.k/(self.x + dx)

            console_log(logger, [
                f"FMarket.slippage: reserves (Quote: x) -> {self.x}",
                f"FMarket.slippage: position impact (Quote: dx) -> {dx}",
                f"FMarket.slippage: position impact % (Quote: dx/x) -> {dx/self.x}",
                f"FMarket.slippage: reserves (Base: y) -> {self.y}",
                f"FMarket.slippage: position impact (Base: dy) -> {dy}",
                f"FMarket.slippage: position impact % (Base: dy/y) -> {dy/self.y}"
            ])

            assert dy < self.y, "slippage: Not enough liquidity in self.y for swap"
            slippage = ((self.x+dx)/(self.y-dy) - self.price) / self.price

            console_log(logger, [
                f"FMarket.slippage: price before -> {self.price}",
                f"FMarket.slippage: price after -> {(self.x+dx)/(self.y-dy)}",
                f"FMarket.slippage: slippage -> {slippage}"
            ])
        else:
            console_log(logger, [
                "FMarket.slippage: dn = -px*dx; (x-dx)*(y+dy) = k",
                f"FMarket.slippage: px -> {self.px}",
                f"FMarket.slippage: py -> {self.py}",
            ])

            dy = self.py*dn*leverage
            dx = self.x - self.k/(self.y + dy)

            console_log(logger, [
                f"FMarket.slippage: reserves (Quote: x) -> {self.x}",
                f"FMarket.slippage: position impact (Quote: dx) -> {dx}",
                f"FMarket.slippage: position impact % (Quote: dx/x) -> {dx/self.x}",
                f"FMarket.slippage: reserves (Base: y) -> {self.y}",
                f"FMarket.slippage: position impact (Base: dy) -> {dy}",
                f"FMarket.slippage: position impact % (Base: dy/y) -> {dy/self.y}",
            ])

            assert dx < self.x, "slippage: Not enough liquidity in self.x for swap"
            slippage = ((self.x-dx)/(self.y+dy) - self.price) / self.price

            console_log(logger, [
                f"FMarket.slippage: price before -> {self.price}",
                f"FMarket.slippage: price after -> {(self.x-dx)/(self.y+dy)}",
                f"FMarket.slippage: slippage -> {slippage}"
            ])

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

            console_log(logger, [
                f"_swap: position size (OVL) -> {dn*leverage}",
                f"_swap: position impact (Quote: dx) -> {dx}",
                f"_swap: position impact (Base: dy) -> {dy}"
            ])

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

            console_log(logger, [
                f"_swap: position size (OVL) -> {dn*leverage}",
                f"_swap: position impact (Quote: dx) -> {dx}",
                f"_swap: position impact (Base: dy) -> {dy}"
            ])

            assert dx < self.x, "_swap: Not enough liquidity in self.x for swap"
            assert dx/self.px < self.nx, "_swap: Not enough liquidity in self.nx for swap"
            avg_price = self.k / (self.x * (self.x-dx))
            self.y += dy
            self.ny += dy/self.py
            self.x -= dx
            self.nx -= dx/self.px

        console_log(logger, [
            f"_swap: {'Built' if build else 'Unwound'} {'long' if long else 'short'} position on {self.unique_id} of size {dn*leverage} OVL at avg price of {1/avg_price}, with lock price {self.price}",
            f"_swap: Percent diff bw avg and lock price is {100*(1/avg_price - self.price)/self.price}%",
            f"_swap: locked_long -> {self.locked_long} OVL",
            f"_swap: nx -> {self.nx}",
            f"_swap: x -> {self.x}",
            f"_swap: locked_short -> {self.locked_short} OVL",
            f"_swap: ny -> {self.ny}",
            f"_swap: y -> {self.y}",
        ])

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
            console_log(logger, [
                f"No position with pid {pid} exists on market {self.unique_id}"
            ])
            return None, 0.0
        elif pos.amount < dn:
            console_log(logger, [
                f"Unwind amount {dn} is too large for locked position with pid {pid} amount {pos.amount}"
            ])
            return None, 0.0

        # TODO: Account for pro-rata share of funding!
        # TODO: Fix this! something's wrong and I'm getting negative reserve amounts upon unwind :(
        # TODO: Locked long seems to go negative which is wrong. Why here?

        # Unlock from long/short pool first
        console_log(logger, [
            f"unwind: dn = {dn}",
            f"unwind: pos = {pos.id}",
            f"unwind: locked_long = {self.locked_long}",
            f"unwind: locked_short = {self.locked_short}"
        ])

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
        cum_price_feed=np.sum(np.array(
            self.model.sims[self.unique_id][self.last_funding_idx:idx]
        ))
        twap_feed=cum_price_feed / dt

        # Calculate twap of market ... update cum price value first
        #twap_market = (self.cum_price - self.last_cum_price) / dt
        # TODO: twap market instead of actual price below since this is bad (but just for testing sniper for now)
        twap_market=self.price
        funding=(twap_market - twap_feed) / twap_feed

        console_log(logger, [
            f"funding: Checking funding for {self.unique_id}",
            f"funding: cum_price_feed = {cum_price_feed}",
            f"funding: Time since last funding (dt) = {dt}",
            f"funding: twap_feed = {twap_feed}",
            f"funding: cum_price = {self.cum_price}",
            f"funding: last_cum_price = {self.last_cum_price}",
            f"funding: twap_market = {twap_market}",
        ])

        return funding

    def fund(self):
        # Pay out funding to each respective pool based on underlying market
        # oracle fetch
        # Calculate the TWAP over previous sample
        idx=self.model.schedule.steps
        if (idx % self.model.sampling_interval != 0) or (idx-self.model.sampling_interval < 0) or (idx == self.last_funding_idx):
            return

        # Calculate twap of oracle feed ... each step is value 1 in time weight
        cum_price_feed=np.sum(np.array(
            self.model.sims[self.unique_id][idx - \
                self.model.sampling_interval:idx]
        ))

        twap_feed=cum_price_feed / self.model.sampling_interval

        console_log(logger, [
            f"fund: Paying out funding for {self.unique_id}",
            f"fund: cum_price_feed = {cum_price_feed}",
            f"fund: sampling_interval = {self.model.sampling_interval}",
            f"fund: twap_feed = {twap_feed}",
        ])

        # Calculate twap of market ... update cum price value first
        self._update_cum_price()

        console_log(logger, [
            f"fund: cum_price = {self.cum_price}",
            f"fund: last_cum_price = {self.last_cum_price}",
        ])

        twap_market=(self.cum_price - self.last_cum_price) / \
                     self.model.sampling_interval
        self.last_cum_price=self.cum_price

        console_log(logger, [f"fund: twap_market = {twap_market}"])

        # Calculate twa open interest for each side over sampling interval
        self._update_cum_locked_long()

        twao_long=(self.cum_locked_long - self.last_cum_locked_long) / \
                   self.model.sampling_interval

        console_log(logger, [
            f"fund: nx={self.nx}",
            f"fund: px={self.px}",
            f"fund: x={self.x}",
            f"fund: locked_long={self.locked_long}",
            f"fund: cum_locked_long={self.cum_locked_long}",
            f"fund: last_cum_locked_long={self.last_cum_locked_long}",
            f"fund: twao_long={twao_long}",
        ])

        self.last_cum_locked_long=self.cum_locked_long

        self._update_cum_locked_short()

        twao_short=(self.cum_locked_short - \
                    self.last_cum_locked_short) / self.model.sampling_interval

        console_log(logger, [
            f"fund: ny={self.ny}",
            f"fund: py={self.py}",
            f"fund: y={self.y}",
            f"fund: locked_short={self.locked_short}",
            f"fund: cum_locked_short={self.cum_locked_short}",
            f"fund: last_cum_locked_short={self.last_cum_locked_short}",
            f"fund: twao_short={twao_short}",
        ])

        self.last_cum_locked_short=self.cum_locked_short

        # Mark the last funding idx as now
        self.last_funding_idx=idx

        # Mint/burn funding
        funding=(twap_market - twap_feed) / twap_feed
        self.last_funding_rate=funding
        # print(f"fund: funding % -> {funding*100.0}%")
        if funding == 0.0:
            return
        elif funding > 0.0:
            funding=min(funding, 1.0)
            # can't have negative locked long
            funding_long=min(twao_long*funding, self.locked_long)
            funding_short=twao_short*funding
            self.model.supply += funding_short - funding_long
            self.locked_long -= funding_long
            self.locked_short += funding_short
            console_log(logger, [
                f"fund: Adding ds={funding_short - funding_long} OVL to total supply",
                f"fund: Adding ds={-funding_long} OVL to longs",
                f"fund: Adding ds={funding_short} OVL to shorts",
            ])
        else:
            funding=max(funding, -1.0)
            funding_long=abs(twao_long*funding)
            # can't have negative locked short
            funding_short=min(abs(twao_short*funding), self.locked_short)
            # print(f"fund: Adding ds={funding_long - funding_short} OVL to total supply")
            self.model.supply += funding_long - funding_short
            # print(f"fund: Adding ds={funding_long} OVL to longs")
            self.locked_long += funding_long
            # print(f"fund: Adding ds={-funding_short} OVL to shorts")
            self.locked_short -= funding_short
            console_log(logger, [
                f"fund: Adding ds={funding_long - funding_short} OVL to total supply",
                f"fund: Adding ds={funding_long} OVL to longs",
                f"fund: Adding ds={-funding_short} OVL to shorts",
            ])

        # Update virtual liquidity reserves
        # p_market = n_x*p_x/(n_y*p_y) = x/y; nx + ny = L/n (ignoring weighting, but maintain price ratio); px*nx = x, py*ny = y;\
        # n_y = (1/p_y)*(n_x*p_x)/(p_market) ... nx + n_x*(p_x/p_y)(1/p_market) = L/n
        # n_x = L/n * (1/(1 + (p_x/p_y)*(1/p_market)))
        console_log(logger, [
            f"fund: Adjusting virtual liquidity constants for {self.unique_id}",
            f"fund: nx (prior) = {self.nx}",
            f"fund: ny (prior) = {self.ny}",
            f"fund: x (prior) = {self.x}",
            f"fund: y (prior) = {self.y}",
            f"fund: price (prior) = {self.price}",
        ])

        # TODO: use liquidity_supply_emission ...
        liquidity=self.model.liquidity
        liq_scale_factor=liquidity / self.last_liquidity

        console_log(logger, [
            f"fund: last_liquidity = {self.last_liquidity}",
            f"fund: new liquidity = {liquidity}",
            f"fund: liquidity scale factor = {liq_scale_factor}"
        ])

        self.last_liquidity=liquidity
        self.nx *= liq_scale_factor
        self.ny *= liq_scale_factor
        self.x=self.nx*self.px
        self.y=self.ny*self.py
        self.k=self.x * self.y

        console_log(logger, [
            f"fund: nx (updated) = {self.nx}",
            f"fund: ny (updated) = {self.ny}",
            f"fund: x (updated) = {self.x}",
            f"fund: y (updated) = {self.y}",
            f"fund: price (updated... should be same) = {self.price}",
        ])

        # Calculate twap for ovlusd oracle feed to use in px, py adjustment
        cum_ovlusd_feed=np.sum(np.array(
            self.model.sims[OVL_USD_TICKER][idx-self.model.sampling_interval:idx]
        ))
        twap_ovlusd_feed=cum_ovlusd_feed / self.model.sampling_interval
        self.px=twap_ovlusd_feed  # px = n_usd/n_ovl
        self.py=twap_ovlusd_feed/twap_feed  # py = px/p

        console_log(logger, [
            f"fund: Adjusting price sensitivity constants for {self.unique_id}",
            f"fund: cum_price_feed = {cum_ovlusd_feed}",
            f"fund: twap_ovlusd_feed = {twap_ovlusd_feed}",
            f"fund: px (updated) = {self.px}",
            f"fund: py (updated) = {self.py}",
            f"fund: price (updated... should be same) = {self.price}",
        ])

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
        return open_margin < maintenance_margin

    def liquidate(self, pid: uuid.UUID) -> float:
        can = self.liquidatable(pid)
        pos = self.positions.get(pid)
        if pos is None or not can:
            return 0.0

        # Unwind but change supply back to original before unwind to factor in
        # reward to liquidator (then modify supply again) -> this is hacky
        _, ds = self.unwind(pos.amount, pid)
        self.model.supply -= ds

        # NOTE: ds should be negative
        assert ds < 0, "liquidate: position liquidation should result in burn of amount"
        reward = abs(ds) * self.liquidate_reward

        # Anything left over after the burn is pos.amount - abs(ds) (leftover margin) ... split this
        margin = min(pos.amount - abs(ds), 0)

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
