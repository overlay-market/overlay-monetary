import logging
from functools import partial
import typing as tp
from random import randint

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from tqdm import tqdm

from ovm.debug_level import (
    PERFORM_DEBUG_LOGGING,
    PERFORM_INFO_LOGGING
)

from ovm.tickers import OVL_TICKER

from ovm.monetary.data_collection import (
    DataCollectionOptions,
    HDF5DataCollector
)

from ovm.monetary.data_io import AgentBasedSimulationInputData

from ovm.monetary.plot_labels import (
    price_deviation_label,
    spot_price_label,
    futures_price_label,
    skew_label,
    reserve_skew_relative_label,
    open_positions_label,
    inventory_wealth_ovl_label,
    inventory_wealth_quote_label,
    agent_wealth_ovl_label,
    GINI_LABEL,
    SUPPLY_LABEL,
    TREASURY_LABEL,
    LIQUIDITY_LABEL
)

from ovm.time_resolution import TimeResolution

# set up logging
logger = logging.getLogger(__name__)


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
        num_arbitrageurs: int,
        num_keepers: int,
        num_traders: int,
        num_holders: int,
        num_snipers: int,
        num_long_apes: int,
        num_short_apes: int,
        num_liquidators: int,
        # sims: tp.Dict[str, np.ndarray],
        input_data: AgentBasedSimulationInputData,
        quote_ticker: str,
        ovl_quote_ticker: str,
        base_wealth: float,
        base_market_fee: float,
        base_max_leverage: float,
        base_maintenance: float,
        base_liquidate_reward: float,
        liquidity: float,
        liquidity_supply_emission: tp.List[float],
        treasury: float,
        sampling_interval: int,
        sampling_twap_granularity: int,
        trade_limit: int,
        time_resolution: TimeResolution,
        data_collection_options: DataCollectionOptions = DataCollectionOptions(),
        seed: tp.Optional[int] = None
    ):
        from ovm.monetary.agents import (
            MonetaryArbitrageur,
            MonetaryKeeper,
            MonetaryHolder,
            MonetaryTrader,
            MonetarySniper,
            MonetaryLiquidator,
            MonetaryApe,
        )

        from ovm.monetary.markets import MonetaryFMarket

        from ovm.monetary.reporters import (
            GiniReporter,
            PriceDifferenceReporter,
            FuturesPriceReporter,
            SpotPriceReporter,
            SupplyReporter,
            LiquidityReporter,
            TreasuryReporter,
            AggregateWealthForAgentTypeReporter,
            AggregateInventoryWealthForAgentTypeReporter,
            SkewReporter,
            ReserveSkewRelativeReporter,
            OpenPositionReporter,
            AgentWealthReporter
        )

        super().__init__(seed=seed)
        self.num_agents = num_arbitrageurs + num_keepers + \
            num_traders + num_holders + num_snipers + num_liquidators + \
            num_long_apes + num_short_apes
        self.num_arbitraguers = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_traders = num_traders
        self.num_holders = num_holders
        self.num_snipers = num_snipers
        self.num_long_apes = num_long_apes
        self.num_short_apes = num_short_apes
        self.num_liquidators = num_liquidators
        self.base_wealth = base_wealth
        self.base_market_fee = base_market_fee
        self.base_max_leverage = base_max_leverage
        self.base_maintenance = base_maintenance
        self.liquidity = liquidity
        self.treasury = treasury
        self.data_collection_options = data_collection_options
        self.sampling_interval = sampling_interval
        self.sampling_twap_granularity = sampling_twap_granularity
        self.supply = base_wealth * self.num_agents + liquidity
        self.schedule = RandomActivation(self)
        # self.sims = sims  # { k: [ prices ] }
        self.input_data = input_data
        self.quote_ticker = quote_ticker
        self.ovl_quote_ticker = ovl_quote_ticker
        self.trade_limit = trade_limit
        self.time_resolution = time_resolution

        if PERFORM_INFO_LOGGING:
            print("Model kwargs for initial conditions of sim:")
            print(f"quote_ticker = {quote_ticker}")
            print(f"ovl_quote_ticker = {ovl_quote_ticker}")
            print(f"num_arbitrageurs = {num_arbitrageurs}")
            print(f"num_snipers = {num_snipers}")
            print(f"num_long_apes = {num_long_apes}")
            print(f"num_short_apes = {num_short_apes}")
            print(f"num_keepers = {num_keepers}")
            print(f"num_traders = {num_traders}")
            print(f"num_holders = {num_holders}")
            print(f"num_liquidators = {num_liquidators}")
            print(f"base_wealth = {base_wealth}")
            print(f"total_supply = {self.supply}")
            print(f"sampling_interval = {self.sampling_interval}")
            print(f"sampling_twap_granularity = {self.sampling_twap_granularity}")
            print(f"trade_limit = {self.trade_limit}")
            print(
                f"num_agents * base_wealth + liquidity = {self.num_agents*self.base_wealth + self.liquidity}")

        # Markets: Assume OVL-QUOTE is in here and only have X-QUOTE pairs for now ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        # if x + y = L/n and x/y = p; nx = (L/2n), ny = (L/2n), x*y = k = (px*L/2n)*(py*L/2n)
        n = len(self.sims.keys())
        prices_ovl_quote = self.sims[self.ovl_quote_ticker]
        liquidity_weight = {
            list(self.sims.keys())[i]: 1
            for i in range(n)
        }
        self.fmarkets = {
            ticker: MonetaryFMarket(
                unique_id=ticker,
                nx=(self.liquidity/(2*n))*liquidity_weight[ticker],
                ny=(self.liquidity/(2*n))*liquidity_weight[ticker],
                px=prices_ovl_quote[0],  # px = n_quote/n_ovl
                py=prices_ovl_quote[0]/prices[0],  # py = px/p
                base_fee=base_market_fee,
                max_leverage=base_max_leverage,
                liquidate_reward=base_liquidate_reward,
                maintenance=base_maintenance,
                trade_limit=trade_limit,
                model=self,
            )
            for ticker, prices in self.sims.items()
        }

        tickers = list(self.fmarkets.keys())
        for i in range(self.num_agents):
            agent = None
            fmarket = self.fmarkets[tickers[i % len(tickers)]]
            base_curr = fmarket.unique_id[:-len(f"-{quote_ticker}")]
            base_quote_price = self.sims[fmarket.unique_id][0]
            inventory: tp.Dict[str, float] = {}
            if base_curr != OVL_TICKER:
                inventory = {
                    OVL_TICKER: self.base_wealth,
                    quote_ticker: self.base_wealth*prices_ovl_quote[0],
                    base_curr: self.base_wealth*prices_ovl_quote[0]/base_quote_price,
                }  # 50/50 inventory of base and quote curr (3x base_wealth for total in OVL)
            else:
                inventory = {
                    OVL_TICKER: self.base_wealth*2,  # 2x since using for both spot and futures
                    quote_ticker: self.base_wealth*prices_ovl_quote[0]
                }
            # For leverage max, pick an integer between 1.0 & 5.0 (vary by agent)
            leverage_max = randint(1, 3)
            init_delay = 0  # randint(0, sampling_interval)
            if i < self.num_arbitraguers:
                agent = MonetaryArbitrageur(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth*0.25,
                    leverage_max=leverage_max,
                    init_delay=init_delay,
                    trade_delay=5,
                    min_edge=0.05, # NOTE: intense here sort of emulates high gas costs (sort of)
                )
            elif i < self.num_arbitraguers + self.num_keepers:
                agent = MonetaryKeeper(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth*0.25,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders:
                agent = MonetaryHolder(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth*0.25,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders:
                agent = MonetaryTrader(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth*0.25,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders + self.num_snipers:
                # sniper_leverage_max = randint(1, 2)
                agent = MonetarySniper(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth*0.5,
                    leverage_max=leverage_max,
                    size_increment=0.2,
                    init_delay=init_delay,
                    trade_delay=5,
                    min_edge=0.05, # NOTE: intense here sort of emulates high gas costs (sort of)
                    max_edge=0.1,  # max deploy at 10% edge
                    funding_multiplier=1.0,  # applied to funding cost when considering exiting position
                    min_funding_unwind=0.001,  # start unwind when funding reaches .1% against position
                    max_funding_unwind=0.02  # unwind immediately when funding reaches 2% against position
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders + self.num_snipers + self.num_liquidators:
                agent = MonetaryLiquidator(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders + self.num_snipers + self.num_liquidators + self.num_long_apes:
                #ape_leverage_max = randint(4, 6)
                unwind_delay = randint(
                    sampling_interval*24*1, sampling_interval*24*7)
                agent = MonetaryApe(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth,
                    side=1,
                    leverage_max=leverage_max,
                    init_delay=init_delay,
                    trade_delay=5,
                    unwind_delay=unwind_delay,
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders + self.num_snipers + self.num_liquidators + self.num_long_apes + self.num_short_apes:
                #ape_leverage_max = randint(4, 6)
                unwind_delay = randint(
                    sampling_interval*24*1, sampling_interval*24*7)
                agent = MonetaryApe(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    pos_amount=self.base_wealth,
                    side=-1,
                    leverage_max=leverage_max,
                    init_delay=init_delay,
                    trade_delay=5,
                    unwind_delay=unwind_delay,
                )
            else:
                from ovm.monetary.agents import MonetaryAgent
                agent = MonetaryAgent(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )

            self.schedule.add(agent)

        if self.data_collection_options.perform_data_collection:
            model_reporters = {
                # price_deviation_label(ticker): partial(compute_price_difference, ticker=ticker)
                price_deviation_label(ticker): PriceDifferenceReporter(ticker=ticker)
                for ticker in tickers
            }
            model_reporters.update({
                # spot_price_label(ticker): partial(compute_spot_price, ticker=ticker)
                spot_price_label(ticker): SpotPriceReporter(ticker=ticker)
                for ticker in tickers
            })
            model_reporters.update({
                # futures_price_label(ticker): partial(compute_futures_price, ticker=ticker)
                futures_price_label(ticker): FuturesPriceReporter(ticker=ticker)
                for ticker in tickers
            })
            model_reporters.update({
                # skew_label(ticker): partial(compute_positional_imbalance_by_market, ticker=ticker)
                skew_label(ticker): SkewReporter(ticker=ticker)
                for ticker in tickers
            })
            model_reporters.update({
                # skew_label(ticker): partial(compute_positional_imbalance_by_market, ticker=ticker)
                reserve_skew_relative_label(ticker): ReserveSkewRelativeReporter(ticker=ticker)
                for ticker in tickers
            })
            model_reporters.update({
                # open_positions_label(ticker): partial(compute_open_positions_per_market, ticker=ticker)
               open_positions_label(ticker): OpenPositionReporter(ticker=ticker)
                for ticker in tickers
            })

            if self.data_collection_options.compute_gini_coefficient:
                model_reporters.update({
                    GINI_LABEL: GiniReporter(),
                })

            model_reporters.update({
                # SUPPLY_LABEL: compute_supply,
                # TREASURY_LABEL: compute_treasury,
                # LIQUIDITY_LABEL: compute_liquidity
                SUPPLY_LABEL: SupplyReporter(),
                TREASURY_LABEL: TreasuryReporter(),
                LIQUIDITY_LABEL: LiquidityReporter()
            })

            if self.data_collection_options.compute_wealth:
                model_reporters.update({
                    # "Agent": partial(compute_wealth_for_agent_type, agent_type=None)
                    "Agent": partial(AggregateWealthForAgentTypeReporter())
                })

            for agent_type_name, agent_type in [("Arbitrageurs", MonetaryArbitrageur),
                                                ("Keepers", MonetaryKeeper),
                                                ("Traders", MonetaryTrader),
                                                ("Holders", MonetaryHolder),
                                                ("Liquidators", MonetaryLiquidator),
                                                ("Snipers", MonetarySniper),
                                                ("Apes", MonetaryApe)]:
                if self.data_collection_options.compute_wealth:
                    model_reporters[agent_wealth_ovl_label(agent_type_name)] = \
                        AggregateWealthForAgentTypeReporter(agent_type=agent_type)
                        # partial(compute_wealth_for_agent_type, agent_type=agent_type)

                if self.data_collection_options.compute_inventory_wealth:
                    model_reporters.update({
                        inventory_wealth_ovl_label(agent_type_name):
                            AggregateInventoryWealthForAgentTypeReporter(agent_type=agent_type),
                            # partial(compute_inventory_wealth_for_agent_type, agent_type=agent_type),

                        inventory_wealth_quote_label(agent_type_name, self.quote_ticker):
                            AggregateInventoryWealthForAgentTypeReporter(agent_type=agent_type, in_quote=True)
                            # partial(compute_inventory_wealth_for_agent_type, agent_type=agent_type, in_quote=True)
                    })

            save_interval = \
                int(24 * 60 * 60 /
                    data_collection_options.data_collection_interval /
                    time_resolution.in_seconds)

            if data_collection_options.use_hdf5:
                self.data_collector = \
                    HDF5DataCollector(
                        model=self,
                        save_interval=save_interval,
                        model_reporters=model_reporters,
                        agent_reporters={"Wealth": AgentWealthReporter()})
            else:
                self.data_collector = DataCollector(
                    model_reporters=model_reporters,
                    agent_reporters={"Wealth": AgentWealthReporter()},
                )

        self.running = True
        if self.data_collection_options.perform_data_collection and \
           not self.data_collection_options.use_hdf5:
            self.data_collector.collect(self)

    @property
    def name(self) -> str:
        return 'MonetaryModel'

    @property
    def number_of_markets(self) -> int:
        return len(self.sims)

    @property
    def sims(self) -> tp.Dict[str, np.ndarray]:
        return self.input_data.ticker_to_series_of_prices_map

    def step(self):
        """
        A model step. Used for collecting simulation and advancing the schedule
        """
        from ovm.monetary.agents import (
            MonetaryArbitrageur,
            MonetarySniper,
            MonetaryLiquidator,
            MonetaryApe,
        )
        if self.data_collection_options.perform_data_collection and \
           self.schedule.steps % self.data_collection_options.data_collection_interval == 0:
            self.data_collector.collect(self)

        if self.schedule.steps % (24 * self.sampling_interval) == 0: # logger.getEffectiveLevel() <= 10
            # Snipers
            top_10_snipers = sorted(
                [a for a in self.schedule.agents if type(a) == MonetarySniper],
                key=lambda item: item.wealth,
                reverse=True
            )[:10]
            bottom_10_snipers = sorted(
                [a for a in self.schedule.agents if type(a) == MonetarySniper],
                key=lambda item: item.wealth
            )[:10]
            top_10_snipers_wealth = {
                a.unique_id: a.wealth
                for a in top_10_snipers
            }
            bottom_10_snipers_wealth = {
                a.unique_id: a.wealth
                for a in bottom_10_snipers
            }
            if PERFORM_INFO_LOGGING:
                print("========================================")
                print(
                    f"Model.step: Sniper wealths top 10 -> {top_10_snipers_wealth}")
                print(
                    f"Model.step: Sniper wealths bottom 10 -> {bottom_10_snipers_wealth}")

            # Arbs
            top_10_arbs = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryArbitrageur],
                key=lambda item: item.wealth,
                reverse=True
            )[:10]
            bottom_10_arbs = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryArbitrageur],
                key=lambda item: item.wealth
            )[:10]
            top_10_arbs_wealth = {
                a.unique_id: a.wealth
                for a in top_10_arbs
            }
            bottom_10_arbs_wealth = {
                a.unique_id: a.wealth
                for a in bottom_10_arbs
            }
            if PERFORM_INFO_LOGGING:
                print("========================================")
                print(
                    f"Model.step: Arb wealths top 10 -> {top_10_arbs_wealth}")
                print(
                    f"Model.step: Arb wealths bottom 10 -> {bottom_10_arbs_wealth}")

            # Liquidators
            top_10_liqs = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryLiquidator],
                key=lambda item: item.wealth,
                reverse=True
            )[:10]
            bottom_10_liqs = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryLiquidator],
                key=lambda item: item.wealth
            )[:10]
            top_10_liqs_wealth = {
                a.unique_id: a.wealth
                for a in top_10_liqs
            }
            bottom_10_liqs_wealth = {
                a.unique_id: a.wealth
                for a in bottom_10_liqs
            }
            if PERFORM_INFO_LOGGING:
                print("========================================")
                print(
                    f"Model.step: Liq wealths top 10 -> {top_10_liqs_wealth}")
                print(
                    f"Model.step: Liq wealths bottom 10 -> {bottom_10_liqs_wealth}")

            # Apes
            top_10_long_apes = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryApe and a.side == 1],
                key=lambda item: item.wealth,
                reverse=True
            )[:10]
            bottom_10_long_apes = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryApe and a.side == 1],
                key=lambda item: item.wealth
            )[:10]
            top_10_short_apes = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryApe and a.side == -1],
                key=lambda item: item.wealth,
                reverse=True
            )[:10]
            bottom_10_short_apes = sorted(
                [a for a in self.schedule.agents if type(
                    a) == MonetaryApe and a.side == -1],
                key=lambda item: item.wealth
            )[:10]
            top_10_long_apes_wealth = {
                a.unique_id: a.wealth
                for a in top_10_long_apes
            }
            bottom_10_long_apes_wealth = {
                a.unique_id: a.wealth
                for a in bottom_10_long_apes
            }
            top_10_short_apes_wealth = {
                a.unique_id: a.wealth
                for a in top_10_short_apes
            }
            bottom_10_short_apes_wealth = {
                a.unique_id: a.wealth
                for a in bottom_10_short_apes
            }
            if PERFORM_INFO_LOGGING:
                print("========================================")
                print(
                    f"Model.step: Ape long wealths top 10 -> {top_10_long_apes_wealth}")
                print(
                    f"Model.step: Ape long wealths bottom 10 -> {bottom_10_long_apes_wealth}")
                print(
                    f"Model.step: Ape short wealths top 10 -> {top_10_short_apes_wealth}")
                print(
                    f"Model.step: Ape short wealths bottom 10 -> {bottom_10_short_apes_wealth}")

        from ovm.monetary.reporters import (
            compute_supply,
            compute_treasury,
            compute_price_difference,
            compute_spot_price,
            compute_skew_for_market,
            compute_open_positions_per_market,
            compute_reserve_skew_for_market,
        )
        if PERFORM_DEBUG_LOGGING:
            print(f"step {self.schedule.steps}")
            print(f"supply {compute_supply(self)}")
            print(f"treasury {compute_treasury(self)}")
            if self.schedule.steps % (24 * self.sampling_interval) == 0:
                for ticker, fmarket in self.fmarkets.items():
                    print(f"fmarket: ticker {ticker}")
                    print(f"fmarket: nx {fmarket.nx}")
                    print(f"fmarket: px {fmarket.px}")
                    print(f"fmarket: ny {fmarket.ny}")
                    print(f"fmarket: py {fmarket.py}")
                    print(f"fmarket: x {fmarket.x}")
                    print(f"fmarket: y {fmarket.y}")
                    print(f"fmarket: k {fmarket.k}")
                    print(
                        f"fmarket: locked_long (OVL) {fmarket.locked_long}")
                    print(
                        f"fmarket: locked_short (OVL) {fmarket.locked_short}")

                    print(f"fmarket: futures price {fmarket.price}")
                    print(f"fmarket: futures sliding TWAP {fmarket.sliding_twap}")
                    print(
                        f"fmarket: spot price {compute_spot_price(self, ticker)}")
                    print(f"fmarket: price_diff bw f/s "
                                 f"{compute_price_difference(self, ticker)}")
                    print(f"fmarket: positional imbalance "
                                 f"{compute_skew_for_market(self, ticker)}")
                    print(f"fmarket: reserve skew "
                                 f"{compute_reserve_skew_for_market(self, ticker, relative=False)}")
                    print(f"fmarket: relative reserve skew "
                                 f"{compute_reserve_skew_for_market(self, ticker, relative=True)}")
                    print(f"fmarket: open positions "
                                 f"{compute_open_positions_per_market(self, ticker)}")

        self.schedule.step()

    def run_steps(self, number_of_steps_to_simulate: int, use_tqdm: bool = True):
        run_range = range(number_of_steps_to_simulate + 1)
        if use_tqdm:
            run_range = tqdm(run_range)
        try:
            for _ in run_range:
                self.step()
        finally:
            if self.data_collection_options.use_hdf5:
                # ToDo: Flush buffer to HDF5 file
                self.data_collector.flush()
