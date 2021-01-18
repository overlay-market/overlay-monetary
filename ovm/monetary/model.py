from functools import partial
import logging
import typing as tp

import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from ovm.debug_level import DEBUG_LEVEL
from ovm.monetary.options import DataCollectionOptions
from ovm.monetary.plot_labels import (
    price_deviation_label,
    spot_price_label,
    futures_price_label,
    skew_label,
    inventory_wealth_ovl_label,
    inventory_wealth_usd_label,
    GINI_LABEL,
    GINI_ARBITRAGEURS_LABEL,
    SUPPLY_LABEL,
    TREASURY_LABEL,
    LIQUIDITY_LABEL
)

from ovm.tickers import (
    USD_TICKER,
    OVL_TICKER,
    OVL_USD_TICKER
)

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
        ticker_to_time_series_of_prices_map: tp.Dict[str, np.ndarray],
        base_wealth: float,
        base_market_fee: float,
        base_max_leverage: float,
        liquidity: float,
        liquidity_supply_emission: tp.List[float],
        treasury: float,
        sampling_interval: int,
        data_collection_options: DataCollectionOptions = DataCollectionOptions()
    ):
        from ovm.monetary.agents import (
            MonetaryArbitrageur,
            MonetaryKeeper,
            MonetaryHolder,
            MonetaryTrader
        )

        from ovm.monetary.markets import MonetaryFMarket

        from ovm.monetary.reporters import (
            compute_gini,
            compute_price_difference,
            compute_futures_price,
            compute_spot_price,
            compute_supply,
            compute_liquidity,
            compute_treasury,
            compute_wealth_for_agent_type,
            compute_inventory_wealth_for_agent_type,
            compute_positional_imbalance_by_market
        )

        super().__init__()
        self.num_agents = num_arbitrageurs + num_keepers + num_traders + num_holders
        self.num_arbitrageurs = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_traders = num_traders
        self.num_holders = num_holders
        self.base_wealth = base_wealth
        self.base_market_fee = base_market_fee
        self.base_max_leverage = base_max_leverage
        self.liquidity = liquidity
        self.treasury = treasury
        # the interval of oracle updates and funding rate recalculations
        self.sampling_interval = sampling_interval
        self.data_collection_options = data_collection_options
        self.supply_of_ovl = base_wealth * self.num_agents + liquidity
        self.schedule = RandomActivation(self)
        self.ticker_to_time_series_of_prices_map = ticker_to_time_series_of_prices_map  # { k: [ time_series_of_prices ] }

        # Markets: Assume OVL-USD is in here and only have X-USD pairs for now ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        # if x + y = L/n and x/y = p; nx = (L/2n), ny = (L/2n), x*y = k = (px*L/2n)*(py*L/2n)
        n = self.number_of_markets
        price_series_ovlusd = self.ticker_to_time_series_of_prices_map[OVL_USD_TICKER]

        # initialize liquidity weights for each market at 1.0
        ticker_to_liquidity_weight_map = \
            {ticker: 1.0 for ticker in ticker_to_time_series_of_prices_map.keys()}

        if logging.root.level <= DEBUG_LEVEL:
            logger.debug(f"OVL-USD first sim price: {price_series_ovlusd[0]}")
            logger.debug(f"ticker_to_liquidity_weight_map = {ticker_to_liquidity_weight_map}")

        # initialize futures markers (Overlay)
        self.ticker_to_futures_market_map = {
            ticker: MonetaryFMarket(
                unique_id=ticker,
                nx=(self.liquidity/(2*n))*ticker_to_liquidity_weight_map[ticker],
                ny=(self.liquidity/(2*n))*ticker_to_liquidity_weight_map[ticker],
                px=price_series_ovlusd[0],  # px = n_usd/n_ovl
                py=price_series_ovlusd[0]/time_series_of_prices[0],  # py = px/p
                base_fee=base_market_fee,
                max_leverage=base_max_leverage,
                model=self
            )
            for ticker, time_series_of_prices
            in self.ticker_to_time_series_of_prices_map.items()
        }

        tickers = list(self.ticker_to_futures_market_map.keys())
        for i in range(self.num_agents):
            futures_market = self.ticker_to_futures_market_map[tickers[i % len(tickers)]]
            base_currency = futures_market.unique_id[:-len(f"-{USD_TICKER}")]
            base_quote_price = self.ticker_to_time_series_of_prices_map[futures_market.unique_id][0]
            if base_currency != OVL_TICKER:
                inventory: tp.Dict[str, float] = {
                    OVL_TICKER: self.base_wealth,
                    USD_TICKER: self.base_wealth*price_series_ovlusd[0],
                    base_currency: self.base_wealth*price_series_ovlusd[0]/base_quote_price,
                }  # 50/50 inventory of base and quote curr (3x base_wealth for total in OVL)
            else:
                inventory: tp.Dict[str, float] = {
                    OVL_TICKER: self.base_wealth*2,  # 2x since using for both spot and futures
                    USD_TICKER: self.base_wealth*price_series_ovlusd[0]
                }
            # For leverage max, pick an integer between 1.0 & 3.0 (vary by agent)
            leverage_max = (i % 3) + 1.0

            if i < self.num_arbitrageurs:
                agent = MonetaryArbitrageur(
                    unique_id=i,
                    model=self,
                    futures_market=futures_market,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitrageurs + self.num_keepers:
                agent = MonetaryKeeper(
                    unique_id=i,
                    model=self,
                    futures_market=futures_market,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitrageurs + self.num_keepers + self.num_holders:
                agent = MonetaryHolder(
                    unique_id=i,
                    model=self,
                    futures_market=futures_market,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitrageurs + self.num_keepers + self.num_holders + self.num_traders:
                agent = MonetaryTrader(
                    unique_id=i,
                    model=self,
                    futures_market=futures_market,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            else:
                from ovm.monetary.agents import MonetaryAgent
                agent = MonetaryAgent(
                    unique_id=i,
                    model=self,
                    futures_market=futures_market,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            if logging.root.level <= DEBUG_LEVEL:
                logger.debug("MonetaryModel.init: Adding agent to schedule ...")
                logger.debug(f"MonetaryModel.init: agent type={type(agent)}")
                logger.debug(f"MonetaryModel.init: unique_id={agent.unique_id}")
                logger.debug(f"MonetaryModel.init: futures market={agent.futures_market.unique_id}")
                logger.debug(f"MonetaryModel.init: leverage_max={agent.leverage_max}")
                logger.debug(f"MonetaryModel.init: inventory={agent.inventory}")

            self.schedule.add(agent)

        # simulation collector
        # TODO: Why are OVL-USD and ETH-USD futures markets not doing anything in terms of arb bots?
        # TODO: What happens if not enough OVL to sway the market time_series_of_prices on the platform? (i.e. all locked up)
        if self.data_collection_options.perform_data_collection:
            model_reporters = {
                price_deviation_label(ticker): partial(compute_price_difference, ticker=ticker)
                for ticker in tickers
            }
            model_reporters.update({
                spot_price_label(ticker): partial(compute_spot_price, ticker=ticker)
                for ticker in tickers
            })
            model_reporters.update({
                futures_price_label(ticker): partial(compute_futures_price, ticker=ticker)
                for ticker in tickers
            })
            model_reporters.update({
                skew_label(ticker): partial(compute_positional_imbalance_by_market, ticker=ticker)
                for ticker in tickers
            })

            if self.data_collection_options.compute_gini_coefficient:
                model_reporters.update({
                    GINI_LABEL: compute_gini,
                    GINI_ARBITRAGEURS_LABEL: partial(compute_gini, agent_type=MonetaryArbitrageur)
                })

            model_reporters.update({
                SUPPLY_LABEL: compute_supply,
                TREASURY_LABEL: compute_treasury,
                LIQUIDITY_LABEL: compute_liquidity
            })

            if self.data_collection_options.compute_wealth:
                model_reporters.update({
                    "Agent": partial(compute_wealth_for_agent_type, agent_type=None)
                })

            for agent_type_name, agent_type in [("Arbitrageurs", MonetaryArbitrageur),
                                                ("Keepers", MonetaryKeeper),
                                                ("Traders", MonetaryTrader),
                                                ("Holders", MonetaryHolder)]:
                if self.data_collection_options.compute_wealth:
                    model_reporters[f'{agent_type_name} Wealth (OVL)'] = partial(compute_wealth_for_agent_type, agent_type=agent_type)

                if self.data_collection_options.compute_inventory_wealth:
                    model_reporters.update({
                        inventory_wealth_ovl_label(agent_type_name): partial(compute_inventory_wealth_for_agent_type, agent_type=agent_type),
                        inventory_wealth_usd_label(agent_type_name): partial(compute_inventory_wealth_for_agent_type, agent_type=agent_type, in_usd=True)
                    })

            self.data_collector = DataCollector(
                model_reporters=model_reporters,
                agent_reporters={"Wealth": "wealth"},
            )

        self.running = True
        if self.data_collection_options.perform_data_collection:
            self.data_collector.collect(self)

    @property
    def number_of_markets(self) -> int:
        return len(self.ticker_to_time_series_of_prices_map)

    def step(self):
        """
        A model step. Used for collecting simulation and advancing the schedule
        """
        if self.data_collection_options.perform_data_collection:
            self.data_collector.collect(self)
        self.schedule.step()
