import logging
import typing as tp

from mesa import Agent

from ovm.debug_level import DEBUG_LEVEL

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
        futures_market: MonetaryFMarket,  # the overlay market this agent is assigned to
        inventory: tp.Dict[str, float],
        pos_max: float = 0.9,
        deploy_max: float = 0.95,
        slippage_max: float = 0.02,
        leverage_max: float = 1.0,
        trade_delay: int = 4*10
    ):  # TODO: Fix constraint issues? => related to liquidity values we set ... do we need to weight liquidity based off vol?
        """
        Customize the agent
        """
        self.unique_id = unique_id
        super().__init__(unique_id, model)
        # each 'trader' focuses on one market for now
        self.futures_market = futures_market
        self.wealth = model.base_wealth  # in ovl
        self.inventory = inventory
        self.locked = 0
        self.pos_max = pos_max
        self.deploy_max = deploy_max
        self.slippage_max = slippage_max
        self.leverage_max = leverage_max

        # this is the number of steps the agent cannot trade for since the last trade
        self.trade_delay = trade_delay

        # the time step at which this agent last traded
        self.last_trade_time_step = 0
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
        # logger.debug(f"Trader agent {self.unique_id} activated")
        if self.wealth > 0 and self.locked / self.wealth < self.deploy_max:
            # Assume only make one trade per step ...
            self.trade()


class MonetaryArbitrageur(MonetaryAgent):
    # TODO: super().__init__() with an arb min for padding, so doesn't trade if can't make X% locked in

    def _unwind_positions(self):
        # For now just assume all positions unwound at once (even tho unrealistic)
        # TODO: rebalance inventory on unwind!
        current_time_step = self.model.schedule.steps
        spot_price = self.model.ticker_to_time_series_of_prices_map[self.futures_market.unique_id][current_time_step]
        spot_price_ovlusd = self.model.ticker_to_time_series_of_prices_map["OVL-USD"][current_time_step]
        for position_id, position in self.positions.items():
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug(f"Arb._unwind_positions: Unwinding position {position_id} on {self.futures_market.unique_id}")
            fees = self.futures_market.fees(position.amount_of_ovl_locked,
                                            build=False, long=(not position.long),
                                            leverage=position.leverage)
            _, ds = self.futures_market.unwind(position.amount_of_ovl_locked, position_id)
            self.inventory["OVL"] += position.amount_of_ovl_locked + ds - fees
            self.locked -= position.amount_of_ovl_locked
            self.wealth += ds - fees
            self.last_trade_time_step = self.model.schedule.steps

            # Counter the futures trade on spot to unwind the arb
            # TODO: Have the spot market counter trades wrapped in SMarket class properly (clean this up)
            if position.long is not True:
                spot_sell_amount = position.amount_of_ovl_locked * position.leverage * spot_price_ovlusd / spot_price
                spot_sell_fees = min(
                    spot_sell_amount*self.futures_market.base_fee, position.amount_of_ovl_locked)
                spot_sell_received = (spot_sell_amount - spot_sell_fees)*spot_price
                # TODO: this is wrong because of the leverage! fix
                self.inventory[self.futures_market.base_currency] -= spot_sell_amount
                self.inventory["USD"] += spot_sell_received

                if logging.root.level <= DEBUG_LEVEL:
                    logger.debug("Arb._unwind_positions: Selling base curr on spot to unwind arb ...")
                    logger.debug(f"Arb._unwind_positions: spot sell amount (OVL) "
                                 f"-> {position.amount_of_ovl_locked}")
                    logger.debug(f"Arb._unwind_positions: spot sell amount "
                                 f"({self.futures_market.base_currency}) -> {spot_sell_amount}")
                    logger.debug(f"Arb._unwind_positions: spot sell fees "
                                 f"({self.futures_market.base_currency}) -> {spot_sell_fees}")
                    logger.debug(f"Arb._unwind_positions: spot sell received (USD) "
                                 f"-> {spot_sell_received}")
                    logger.debug(f"Arb._unwind_positions: inventory -> {self.inventory}")
            else:
                spot_buy_amount = position.amount_of_ovl_locked * position.leverage * spot_price_ovlusd
                spot_buy_fees = min(
                    spot_buy_amount*self.futures_market.base_fee, position.amount_of_ovl_locked)
                spot_buy_received = (spot_buy_amount - spot_buy_fees)/spot_price
                self.inventory["USD"] -= spot_buy_amount
                self.inventory[self.futures_market.base_currency] += spot_buy_received
                if logging.root.level <= DEBUG_LEVEL:
                    logger.debug("Arb._unwind_positions: Buying base curr on spot to lock in arb ...")
                    logger.debug(f"Arb._unwind_positions: spot buy amount (OVL) -> "
                                 f"{position.amount_of_ovl_locked}")
                    logger.debug(f"Arb._unwind_positions: spot buy amount (USD) "
                                 f"-> {spot_buy_amount}")
                    logger.debug(f"Arb._unwind_positions: spot buy fees (USD) -> {spot_buy_fees}")
                    logger.debug(f"Arb._unwind_positions: spot buy received "
                                 f"({self.futures_market.base_currency}) -> {spot_buy_received}")
                    logger.debug(f"Arb._unwind_positions: inventory -> {self.inventory}")

        self.positions = {}

    def _unwind_next_position(self):
        # Get the next position from inventory to unwind for this timestep
        if len(self.positions.keys()) == 0:
            self.unwinding = False
            return
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug('Arb._unwind_next_position: positions (prior)', self.positions)
            logger.debug('Arb._unwind_next_position: locked (prior)', self.locked)
        position_id = list(self.positions.keys())[0]
        position = self.positions[position_id]
        _, ds = self.futures_market.unwind(position.amount_of_ovl_locked, position_id)
        self.locked -= position.amount_of_ovl_locked
        self.last_trade_time_step = self.model.schedule.steps
        del self.positions[position_id]
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug('Arb._unwind_next_position: positions (updated)', self.positions)
            logger.debug('Arb._unwind_next_position: locked (updated)', self.locked)

    def trade(self):
        # If market futures price > spot then short, otherwise long
        # Calc the slippage first to see if worth it
        # TODO: Check for an arb opportunity. If exists, trade it ... bet Y% of current wealth on the arb ...
        # Get ready to arb current spreads
        current_time_step = self.model.schedule.steps
        spot_price = self.model.ticker_to_time_series_of_prices_map[self.futures_market.unique_id][current_time_step]
        spot_price_ovlusd = self.model.ticker_to_time_series_of_prices_map["OVL-USD"][current_time_step]
        futures_price = self.futures_market.price

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
        amount = self.pos_max * self.wealth
        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"Arb.trade: Arb bot {self.unique_id} has {self.wealth - self.locked} OVL left to deploy")
        if self.locked + amount < self.deploy_max*self.wealth:
            if spot_price > futures_price:
                if logging.root.level <= DEBUG_LEVEL:
                    logger.debug(f"Arb.trade: Checking if long position on "
                                 f"{self.futures_market.unique_id} is profitable after slippage ....")

                fees = self.futures_market.fees(amount, build=True, long=True, leverage=self.leverage_max)
                slippage = self.futures_market.slippage(amount - fees,
                                                        build=True,
                                                        long=True,
                                                        leverage=self.leverage_max)

                if logging.root.level <= DEBUG_LEVEL:
                    logger.debug(f"Arb.trade: fees -> {fees}")
                    logger.debug(f"Arb.trade: slippage -> {slippage}")
                    logger.debug(f"Arb.trade: arb profit opp % -> "
                                 f"{spot_price/(futures_price * (1+slippage)) - 1.0 - 2*self.futures_market.base_fee}")

                if self.slippage_max > abs(slippage) and spot_price > futures_price * (1+slippage) \
                    and spot_price/(futures_price * (1+slippage)) - 1.0 - 2*self.futures_market.base_fee > 0.0025: # TODO: arb_min on the RHS here instead of hard coded 0.0025 = 0.25%
                    # enter the trade to arb
                    position = self.futures_market.build(amount, long=True, leverage=self.leverage_max)
                    if logging.root.level <= DEBUG_LEVEL:
                        logger.debug("Arb.trade: Entered long arb trade w position params ...")
                        logger.debug(f"Arb.trade: position amount -> {position.amount_of_ovl_locked}")
                        logger.debug(f"Arb.trade: position long -> {position.long}")
                        logger.debug(f"Arb.trade: position leverage -> {position.leverage}")
                        logger.debug(f"Arb.trade: position lock_price -> {position.lock_price}")
                    self.positions[position.id] = position
                    self.inventory["OVL"] -= position.amount_of_ovl_locked + fees
                    self.locked += position.amount_of_ovl_locked
                    self.wealth -= fees
                    self.last_trade_time_step = current_time_step

                    # Counter the futures trade on spot with sell to lock in the arb
                    # TODO: Check never goes negative and eventually implement with a spot CFMM
                    # TODO: send fees to spot market CFMM ... (amount - fees)
                    spot_sell_amount = position.amount_of_ovl_locked * position.leverage * spot_price_ovlusd / spot_price
                    # assume same as futures fees
                    spot_sell_fees = min(
                        spot_sell_amount*self.futures_market.base_fee, position.amount_of_ovl_locked)
                    spot_sell_received = (
                        spot_sell_amount - spot_sell_fees)*spot_price
                    self.inventory[self.futures_market.base_currency] -= spot_sell_amount
                    self.inventory["USD"] += spot_sell_received
                    if logging.root.level <= DEBUG_LEVEL:
                        logger.debug("Arb.trade: Selling base curr on spot to lock in arb ...")
                        logger.debug(f"Arb.trade: spot sell amount (OVL) -> {position.amount_of_ovl_locked}")
                        logger.debug(f"Arb.trade: spot sell amount ({self.futures_market.base_currency})"
                              f" -> {spot_sell_amount}")
                        logger.debug(f"Arb.trade: spot sell fees ({self.futures_market.base_currency})"
                              f" -> {spot_sell_fees}")
                        logger.debug(f"Arb.trade: spot sell received (USD)"
                              f" -> {spot_sell_received}")
                        logger.debug(f"Arb.trade: inventory -> {self.inventory}")

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = - position.amount * (spot_price_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price + position.amount * (price_t - lock_price)/lock_price
                    #           = position.amount * [ - (spot_price_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) + (price_t/lock_price - 1) ]
                    #           ~ position.amount * [ - price_t/s_price + price_t/lock_price ] (if spot_price_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = position.amount * price_t * [ 1/lock_price - 1/s_price ]
                    # But s_price > lock_price, so PnL (approx) > 0
                    locked_in_approx = position.amount_of_ovl_locked * position.leverage * \
                                       (spot_price/position.lock_price - 1.0)
                    # TODO: incorporate fee structure!
                    if logging.root.level <= DEBUG_LEVEL:
                        logger.debug(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                        logger.debug(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*spot_price_ovlusd}")

            elif spot_price < futures_price:
                if logging.root.level <= DEBUG_LEVEL:
                    logger.debug(f"Arb.trade: Checking if short position on {self.futures_market.unique_id} "
                                 f"is profitable after slippage ....")

                fees = self.futures_market.fees(
                    amount, build=True, long=False, leverage=self.leverage_max)
                # should be negative ...
                slippage = self.futures_market.slippage(
                    amount-fees, build=True, long=False, leverage=self.leverage_max)
                if logging.root.level <= DEBUG_LEVEL:
                    logger.debug(f"Arb.trade: fees -> {fees}")
                    logger.debug(f"Arb.trade: slippage -> {slippage}")
                    logger.debug(f"Arb.trade: arb profit opp % -> "
                      f"{1.0 - spot_price/(futures_price * (1+slippage)) - 2*self.futures_market.base_fee}")
                if self.slippage_max > abs(slippage) and spot_price < futures_price * (1+slippage) \
                    and 1.0 - spot_price/(futures_price * (1+slippage)) - 2*self.futures_market.base_fee > 0.0025: # TODO: arb_min on the RHS here instead of hard coded 0.0025 = 0.25%
                    # enter the trade to arb
                    position = self.futures_market.build(
                        amount, long=False, leverage=self.leverage_max)
                    if logging.root.level <= DEBUG_LEVEL:
                        logger.debug("Arb.trade: Entered short arb trade w position params ...")
                        logger.debug(f"Arb.trade: position.amount -> {position.amount_of_ovl_locked}")
                        logger.debug(f"Arb.trade: position.long -> {position.long}")
                        logger.debug(f"Arb.trade: position.leverage -> {position.leverage}")
                        logger.debug(f"Arb.trade: position.lock_price -> {position.lock_price}")
                    self.positions[position.id] = position
                    self.inventory["OVL"] -= position.amount_of_ovl_locked + fees
                    self.locked += position.amount_of_ovl_locked
                    self.wealth -= fees
                    self.last_trade_time_step = current_time_step

                    # Counter the futures trade on spot with buy to lock in the arb
                    # TODO: Check never goes negative and eventually implement with a spot CFMM
                    # TODO: send fees to spot market CFMM ...
                    # TODO: FIX THIS FOR LEVERAGE SINCE OWING DEBT ON SPOT (and not accounting for it properly) -> Fine with counter unwind ultimately in long run
                    spot_buy_amount = position.amount_of_ovl_locked * position.leverage * spot_price_ovlusd
                    spot_buy_fees = min(
                        spot_buy_amount*self.futures_market.base_fee, position.amount_of_ovl_locked)
                    spot_buy_received = (
                        spot_buy_amount - spot_buy_fees)/spot_price
                    self.inventory["USD"] -= spot_buy_amount
                    self.inventory[self.futures_market.base_currency] += spot_buy_received
                    if logging.root.level <= DEBUG_LEVEL:
                        logger.debug("Arb.trade: Buying base curr on spot to lock in arb ...")
                        logger.debug(f"Arb.trade: spot buy amount (OVL) -> {position.amount_of_ovl_locked}")
                        logger.debug(f"Arb.trade: spot buy amount (USD) -> {spot_buy_amount}")
                        logger.debug(f"Arb.trade: spot buy fees (USD) -> {spot_buy_fees}")
                        logger.debug(f"Arb.trade: spot buy received ({self.futures_market.base_currency})"
                              f" -> {spot_buy_received}")
                        logger.debug(f"Arb.trade: inventory -> {self.inventory}")

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = position.amount * (spot_price_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price - position.amount * (price_t - lock_price)/lock_price
                    #           = position.amount * [ (spot_price_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) - (price_t/lock_price - 1) ]
                    #           ~ position.amount * [ price_t/s_price - price_t/lock_price ] (if spot_price_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = position.amount * price_t * [ 1/s_price - 1/lock_price ]
                    # But s_price < lock_price, so PnL (approx) > 0
                    locked_in_approx = position.amount_of_ovl_locked * position.leverage * \
                                       (1.0 - spot_price/position.lock_price)
                    # TODO: incorporate fee structure!
                    if logging.root.level <= DEBUG_LEVEL:
                        logger.debug(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                        logger.debug(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*spot_price_ovlusd}")
        else:
            # TODO: remove but try this here => dumb logic but want to see
            # what happens to currency supply if end up unwinding before each new trade (so only 1 position per arb)
            self._unwind_positions()

    def step(self):
        """
        Modify this method to change what an individual agent will do during each step.
        Can include logic based on neighbors states.
        """
        current_time_step = self.model.schedule.steps
        # Allow only one trader to trade on a market per block.
        # Add in a trade delay to simulate cooldown due to gas.
        if (self.futures_market.last_trade_time_step != current_time_step) and \
           (self.last_trade_time_step == 0 or (current_time_step - self.last_trade_time_step) > self.trade_delay):
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
        self.futures_market.fund()

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
