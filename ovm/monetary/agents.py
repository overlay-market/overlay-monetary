import typing as tp

from mesa import Agent


class MonetaryAgent(Agent):
    """
    An agent ... these are the arbers with stop losses.
    Add in position hodlers as a different agent
    later (maybe also with stop losses)
    """
    from model import MonetaryModel
    from markets import MonetaryFMarket

    def __init__(
        self,
        unique_id: int,
        model: MonetaryModel,
        fmarket: MonetaryFMarket,
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
        self.fmarket = fmarket  # each 'trader' focuses on one market for now
        self.wealth = model.base_wealth  # in ovl
        self.inventory = inventory
        self.locked = 0
        self.pos_max = pos_max
        self.deploy_max = deploy_max
        self.slippage_max = slippage_max
        self.leverage_max = leverage_max
        self.trade_delay = trade_delay
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
        # print(f"Trader agent {self.unique_id} activated")
        if self.wealth > 0 and self.locked / self.wealth < self.deploy_max:
            # Assume only make one trade per step ...
            self.trade()


class MonetaryArbitrageur(MonetaryAgent):
    # TODO: super().__init__() with an arb min for padding, so doesn't trade if can't make X% locked in

    def _unwind_positions(self):
        # For now just assume all positions unwound at once (even tho unrealistic)
        # TODO: rebalance inventory on unwind!
        idx = self.model.schedule.steps
        sprice = self.model.sims[self.fmarket.unique_id][idx]
        sprice_ovlusd = self.model.sims["OVL-USD"][idx]
        for pid, pos in self.positions.items():
            print(
                f"Arb._unwind_positions: Unwinding position {pid} on {self.fmarket.unique_id}")
            fees = self.fmarket.fees(pos.amount, build=False, long=(
                not pos.long), leverage=pos.leverage)
            _, ds = self.fmarket.unwind(pos.amount, pid)
            self.inventory["OVL"] += pos.amount + ds - fees
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
                print("Arb._unwind_positions: Selling base curr on spot to unwind arb ...")
                print(f"Arb._unwind_positions: spot sell amount (OVL) -> {pos.amount}")
                print(f"Arb._unwind_positions: spot sell amount ({self.fmarket.base_currency})"
                      f" -> {spot_sell_amount}")

                print(f"Arb._unwind_positions: spot sell fees ({self.fmarket.base_currency})"
                      f" -> {spot_sell_fees}")

                print(f"Arb._unwind_positions: spot sell received (USD) -> {spot_sell_received}")
                # TODO: this is wrong because of the leverage! fix
                self.inventory[self.fmarket.base_currency] -= spot_sell_amount
                self.inventory["USD"] += spot_sell_received
                print(f"Arb._unwind_positions: inventory -> {self.inventory}")
            else:
                spot_buy_amount = pos.amount*pos.leverage*sprice_ovlusd
                spot_buy_fees = min(
                    spot_buy_amount*self.fmarket.base_fee, pos.amount)
                spot_buy_received = (spot_buy_amount - spot_buy_fees)/sprice
                print("Arb._unwind_positions: Buying base curr on spot to lock in arb ...")
                print(f"Arb._unwind_positions: spot buy amount (OVL) -> {pos.amount}")
                print(f"Arb._unwind_positions: spot buy amount (USD) -> {spot_buy_amount}")
                print(f"Arb._unwind_positions: spot buy fees (USD) -> {spot_buy_fees}")
                print(f"Arb._unwind_positions: spot buy received ({self.fmarket.base_currency})"
                      f" -> {spot_buy_received}")

                self.inventory["USD"] -= spot_buy_amount
                self.inventory[self.fmarket.base_currency] += spot_buy_received
                print(f"Arb._unwind_positions: inventory -> {self.inventory}")

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
        _, ds = self.fmarket.unwind(pos.amount, pid)
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
        sprice_ovlusd = self.model.sims["OVL-USD"][idx]
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
        print(f"Arb.trade: Arb bot {self.unique_id} has {self.wealth-self.locked} OVL left to deploy")
        if self.locked + amount < self.deploy_max*self.wealth:
            if sprice > fprice:
                print(f"Arb.trade: Checking if long position on {self.fmarket.unique_id} "
                      f"is profitable after slippage ....")

                fees = self.fmarket.fees(amount, build=True, long=True, leverage=self.leverage_max)
                slippage = self.fmarket.slippage(amount-fees,
                                                 build=True,
                                                 long=True,
                                                 leverage=self.leverage_max)

                print(f"Arb.trade: fees -> {fees}")
                print(f"Arb.trade: slippage -> {slippage}")
                print(f"Arb.trade: arb profit opp % -> "
                      f"{sprice/(fprice * (1+slippage)) - 1.0 - 2*self.fmarket.base_fee}")

                if self.slippage_max > abs(slippage) and sprice > fprice * (1+slippage) \
                    and sprice/(fprice * (1+slippage)) - 1.0 - 2*self.fmarket.base_fee > 0.0025: # TODO: arb_min on the RHS here instead of hard coded 0.0025 = 0.25%
                    # enter the trade to arb
                    pos = self.fmarket.build(amount, long=True, leverage=self.leverage_max)
                    print("Arb.trade: Entered long arb trade w pos params ...")
                    print(f"Arb.trade: pos.amount -> {pos.amount}")
                    print(f"Arb.trade: pos.long -> {pos.long}")
                    print(f"Arb.trade: pos.leverage -> {pos.leverage}")
                    print(f"Arb.trade: pos.lock_price -> {pos.lock_price}")
                    self.positions[pos.id] = pos
                    self.inventory["OVL"] -= pos.amount + fees
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
                    print("Arb.trade: Selling base curr on spot to lock in arb ...")
                    print(f"Arb.trade: spot sell amount (OVL) -> {pos.amount}")
                    print(f"Arb.trade: spot sell amount ({self.fmarket.base_currency})"
                          f" -> {spot_sell_amount}")
                    print(f"Arb.trade: spot sell fees ({self.fmarket.base_currency})"
                          f" -> {spot_sell_fees}")
                    print(f"Arb.trade: spot sell received (USD)"
                          f" -> {spot_sell_received}")
                    self.inventory[self.fmarket.base_currency] -= spot_sell_amount
                    self.inventory["USD"] += spot_sell_received
                    print(f"Arb.trade: inventory -> {self.inventory}")

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = - pos.amount * (sprice_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price + pos.amount * (price_t - lock_price)/lock_price
                    #           = pos.amount * [ - (sprice_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) + (price_t/lock_price - 1) ]
                    #           ~ pos.amount * [ - price_t/s_price + price_t/lock_price ] (if sprice_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = pos.amount * price_t * [ 1/lock_price - 1/s_price ]
                    # But s_price > lock_price, so PnL (approx) > 0
                    locked_in_approx = pos.amount * pos.leverage * \
                        (sprice/pos.lock_price - 1.0)
                    # TODO: incorporate fee structure!
                    print(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                    print(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*sprice_ovlusd}")

            elif sprice < fprice:
                print(f"Arb.trade: Checking if short position on {self.fmarket.unique_id} "
                      f"is profitable after slippage ....")

                fees = self.fmarket.fees(
                    amount, build=True, long=False, leverage=self.leverage_max)
                # should be negative ...
                slippage = self.fmarket.slippage(
                    amount-fees, build=True, long=False, leverage=self.leverage_max)
                print(f"Arb.trade: fees -> {fees}")
                print(f"Arb.trade: slippage -> {slippage}")
                print(f"Arb.trade: arb profit opp % -> "
                      f"{1.0 - sprice/(fprice * (1+slippage)) - 2*self.fmarket.base_fee}")
                if self.slippage_max > abs(slippage) and sprice < fprice * (1+slippage) \
                    and 1.0 - sprice/(fprice * (1+slippage)) - 2*self.fmarket.base_fee > 0.0025: # TODO: arb_min on the RHS here instead of hard coded 0.0025 = 0.25%
                    # enter the trade to arb
                    pos = self.fmarket.build(
                        amount, long=False, leverage=self.leverage_max)
                    print("Arb.trade: Entered short arb trade w pos params ...")
                    print(f"Arb.trade: pos.amount -> {pos.amount}")
                    print(f"Arb.trade: pos.long -> {pos.long}")
                    print(f"Arb.trade: pos.leverage -> {pos.leverage}")
                    print(f"Arb.trade: pos.lock_price -> {pos.lock_price}")
                    self.positions[pos.id] = pos
                    self.inventory["OVL"] -= pos.amount + fees
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
                    print("Arb.trade: Buying base curr on spot to lock in arb ...")
                    print(f"Arb.trade: spot buy amount (OVL) -> {pos.amount}")
                    print(f"Arb.trade: spot buy amount (USD) -> {spot_buy_amount}")
                    print(f"Arb.trade: spot buy fees (USD) -> {spot_buy_fees}")
                    print(f"Arb.trade: spot buy received ({self.fmarket.base_currency})"
                          f" -> {spot_buy_received}")
                    self.inventory["USD"] -= spot_buy_amount
                    self.inventory[self.fmarket.base_currency] += spot_buy_received
                    print(f"Arb.trade: inventory -> {self.inventory}")

                    # Calculate amount profit locked in in OVL and USD terms ... (This is rough for now since not accounting for OVL exposure and actual PnL forms ... and assuming spot/futures converge with funding doing it)
                    # PnL (OVL) = pos.amount * (sprice_ovlusd/sprice_ovlusd_t) * (price_t - s_price)/s_price - pos.amount * (price_t - lock_price)/lock_price
                    #           = pos.amount * [ (sprice_ovlusd/sprice_ovlusd_t) * (price_t/s_price - 1 ) - (price_t/lock_price - 1) ]
                    #           ~ pos.amount * [ price_t/s_price - price_t/lock_price ] (if sprice_ovlusd/sprice_ovlusd_t ~ 1 over trade entry/exit time period)
                    #           = pos.amount * price_t * [ 1/s_price - 1/lock_price ]
                    # But s_price < lock_price, so PnL (approx) > 0
                    locked_in_approx = pos.amount * pos.leverage * \
                        (1.0 - sprice/pos.lock_price)
                    # TODO: incorporate fee structure!
                    print(f"Arb.trade: arb profit locked in (OVL) = {locked_in_approx}")
                    print(f"Arb.trade: arb profit locked in (USD) = {locked_in_approx*sprice_ovlusd}")
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
