import logging
from functools import partial
import typing as tp

from logs import console_log

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from ovm.debug_level import DEBUG_LEVEL
from ovm.tickers import OVL_USD_TICKER

from ovm.monetary.options import DataCollectionOptions
from ovm.monetary.plot_labels import (
    price_deviation_label,
    spot_price_label,
    futures_price_label,
    skew_label,
    inventory_wealth_ovl_label,
    inventory_wealth_usd_label,
    agent_wealth_ovl_label,
    GINI_LABEL,
    GINI_ARBITRAGEURS_LABEL,
    SUPPLY_LABEL,
    TREASURY_LABEL,
    LIQUIDITY_LABEL
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
        num_snipers: int,
        num_liquidators: int,
        sims: tp.Dict[str, tp.List[float]],
        base_wealth: float,
        base_market_fee: float,
        base_max_leverage: float,
        base_maintenance: float,
        base_liquidate_reward: float,
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
            MonetaryTrader,
            MonetarySniper,
            MonetaryLiquidator,
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
            compute_positional_imbalance_by_market,
        )

        super().__init__()
        self.num_agents = num_arbitrageurs + num_keepers + \
            num_traders + num_holders + num_snipers + num_liquidators
        self.num_arbitraguers = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_traders = num_traders
        self.num_holders = num_holders
        self.num_snipers = num_snipers
        self.num_liquidators = num_liquidators
        self.base_wealth = base_wealth
        self.base_market_fee = base_market_fee
        self.base_max_leverage = base_max_leverage
        self.base_maintenance = base_maintenance
        self.liquidity = liquidity
        self.treasury = treasury
        self.data_collection_options = data_collection_options
        self.sampling_interval = sampling_interval
        self.supply = base_wealth * self.num_agents + liquidity
        self.schedule = RandomActivation(self)
        self.sims = sims  # { k: [ prices ] }

        console_log(logger, [
            "Model kwargs for initial conditions of sim:",
            f"num_arbitrageurs = {num_arbitrageurs}",
            f"num_snipers = {num_snipers}",
            f"num_keepers = {num_keepers}",
            f"num_traders = {num_traders}",
            f"num_holders = {num_holders}",
            f"num_liquidators = {num_liquidators}",
            f"base_wealth = {base_wealth}",
            f"total_supply = {self.supply}",
            f"num_agents * base_wealth + liquidity = {self.num_agents*self.base_wealth + self.liquidity}",
        ], level=logging.INFO)

        # Markets: Assume OVL-USD is in here and only have X-USD pairs for now ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        # if x + y = L/n and x/y = p; nx = (L/2n), ny = (L/2n), x*y = k = (px*L/2n)*(py*L/2n)
        n = len(sims.keys())
        prices_ovlusd = self.sims[OVL_USD_TICKER]
        liquidity_weight = {
            list(sims.keys())[i]: 1
            for i in range(n)
        }
        self.fmarkets = {
            ticker: MonetaryFMarket(
                unique_id=ticker,
                nx=(self.liquidity/(2*n))*liquidity_weight[ticker],
                ny=(self.liquidity/(2*n))*liquidity_weight[ticker],
                px=prices_ovlusd[0],  # px = n_usd/n_ovl
                py=prices_ovlusd[0]/prices[0],  # py = px/p
                base_fee=base_market_fee,
                max_leverage=base_max_leverage,
                liquidate_reward=base_liquidate_reward,
                maintenance=base_maintenance,
                model=self,
            )
            for ticker, prices in self.sims.items()
        }

        tickers = list(self.fmarkets.keys())
        for i in range(self.num_agents):
            agent = None
            fmarket = self.fmarkets[tickers[i % len(tickers)]]
            base_curr = fmarket.unique_id[:-len("-USD")]
            base_quote_price = self.sims[fmarket.unique_id][0]
            inventory: tp.Dict[str, float] = {}
            if base_curr != 'OVL':
                inventory = {
                    'OVL': self.base_wealth,
                    'USD': self.base_wealth*prices_ovlusd[0],
                    base_curr: self.base_wealth*prices_ovlusd[0]/base_quote_price,
                }  # 50/50 inventory of base and quote curr (3x base_wealth for total in OVL)
            else:
                inventory = {
                    'OVL': self.base_wealth*2,  # 2x since using for both spot and futures
                    'USD': self.base_wealth*prices_ovlusd[0]
                }
            # For leverage max, pick an integer between 1.0 & 5.0 (vary by agent)
            leverage_max = (i % 9.0) + 1.0

            if i < self.num_arbitraguers:
                agent = MonetaryArbitrageur(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers:
                agent = MonetaryKeeper(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders:
                agent = MonetaryHolder(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders:
                agent = MonetaryTrader(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )
            elif i < self.num_arbitraguers + self.num_keepers + self.num_holders + self.num_traders + self.num_snipers:
                sniper_leverage_max = (i % 3.0) + 1.0
                agent = MonetarySniper(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max,
                    trade_delay=4*10,  # 15 s blocks ... TODO: make this inverse with amount remaining to lock
                    size_increment=0.05,
                    min_edge=0.0,
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
                    GINI_ARBITRAGEURS_LABEL: partial(
                        compute_gini, agent_type=MonetaryArbitrageur)
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
                    model_reporters[agent_wealth_ovl_label(agent_type_name)] = partial(
                        compute_wealth_for_agent_type, agent_type=agent_type)

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
        return len(self.sims)

    def step(self):
        """
        A model step. Used for collecting simulation and advancing the schedule
        """
        from ovm.monetary.agents import (
            MonetaryArbitrageur,
            MonetarySniper,
            MonetaryLiquidator,
        )
        if self.data_collection_options.perform_data_collection and \
           self.schedule.steps % self.data_collection_options.data_collection_interval == 0:
            self.data_collector.collect(self)

        if logger.getEffectiveLevel() <= 10:
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
            console_log(logger, [
                "========================================",
                f"Model.step: Sniper wealths top 10 -> {top_10_snipers_wealth}",
                f"Model.step: Sniper wealths bottom 10 -> {bottom_10_snipers_wealth}",
            ], level=logging.INFO)

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
            console_log(logger, [
                "========================================",
                f"Model.step: Arb wealths top 10 -> {top_10_arbs_wealth}",
                f"Model.step: Arb wealths bottom 10 -> {bottom_10_arbs_wealth}",
            ], level=logging.INFO)

            # Liquidators
            top_10_liqs = sorted(
                [a for a in self.schedule.agents if type(a) == MonetaryLiquidator],
                key=lambda item: item.wealth,
                reverse=True
            )[:10]
            bottom_10_liqs = sorted(
                [a for a in self.schedule.agents if type(a) == MonetaryLiquidator],
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
            console_log(logger, [
                "========================================",
                f"Model.step: Liq wealths top 10 -> {top_10_liqs_wealth}",
                f"Model.step: Liq wealths bottom 10 -> {bottom_10_liqs_wealth}",
            ], level=logging.INFO)

        self.schedule.step()
