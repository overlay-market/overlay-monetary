import logging
from functools import partial
import typing as tp

from logs import console_log

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from ovm.debug_level import DEBUG_LEVEL
from ovm.tickers import (
    USD_TICKER,
    OVL_TICKER,
    OVL_USD_TICKER
)

from options import DataCollectionOptions
from plot_labels import (
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
        sampling_interval: int
    ):
        from agents import (
            MonetaryArbitrageur,
            MonetaryKeeper,
            MonetaryHolder,
            MonetaryTrader,
            MonetarySniper,
            MonetaryLiquidator,
        )

        from markets import MonetaryFMarket

        from reporters import (
            compute_gini,
            compute_price_diff,
            compute_fprice,
            compute_sprice,
            compute_supply,
            compute_liquidity,
            compute_treasury,
            compute_wealth,
            compute_inventory_wealth,
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
        prices_ovlusd = self.sims["OVL-USD"]
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
            sniper_leverage_max = (i % 3.0) + 1.0

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
                agent = MonetarySniper(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=sniper_leverage_max,
                    trade_delay=4*10,  # 15 s blocks ... TODO: make this inverse with amount remaining to lock
                    size_increment=0.01,
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
                from .agents import MonetaryAgent
                agent = MonetaryAgent(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )

            self.schedule.add(agent)

        # simulation collector
        # TODO: Why are OVL-USD and ETH-USD futures markets not doing anything in terms of arb bots?
        # TODO: What happens if not enough OVL to sway the market prices on the platform? (i.e. all locked up)
        model_reporters = {
            f"d-{ticker}": partial(compute_price_diff, ticker=ticker)
            for ticker in tickers
        }
        model_reporters.update({
            f"s-{ticker}": partial(compute_sprice, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            f"f-{ticker}": partial(compute_fprice, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            "Gini": compute_gini,
            #"Gini (Arbitrageurs)": partial(compute_gini, agent_type=MonetaryArbitrageur),
            "Supply": compute_supply,
            "Treasury": compute_treasury,
            #"Liquidity": compute_liquidity,
            #"Agent": partial(compute_wealth, agent_type=None),
            #"Arbitrageurs Wealth (OVL)": partial(compute_wealth, agent_type=MonetaryArbitrageur),
            #"Arbitrageurs Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetaryArbitrageur),
            #"Arbitrageurs OVL Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetaryArbitrageur, inventory_type="OVL"),
            #"Arbitrageurs Inventory (USD)": partial(compute_inventory_wealth, agent_type=MonetaryArbitrageur, in_usd=True),
            #"Snipers Wealth (OVL)": partial(compute_wealth, agent_type=MonetarySniper),
            #"Snipers Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetarySniper),
            #"Snipers OVL Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetarySniper, inventory_type="OVL"),
            #"Snipers Inventory (USD)": partial(compute_inventory_wealth, agent_type=MonetarySniper, in_usd=True),
            #"Keepers Wealth (OVL)": partial(compute_wealth, agent_type=MonetaryKeeper),
            #"Keepers Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetaryKeeper),
            #"Keepers Inventory (USD)": partial(compute_inventory_wealth, agent_type=MonetaryKeeper, in_usd=True),
            #"Traders Wealth (OVL)": partial(compute_wealth, agent_type=MonetaryTrader),
            #"Traders Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetaryKeeper),
            #"Traders Inventory (USD)": partial(compute_inventory_wealth, agent_type=MonetaryKeeper, in_usd=True),
            #"Holders Wealth (OVL)": partial(compute_wealth, agent_type=MonetaryHolder),
            #"Holders Inventory (OVL)": partial(compute_inventory_wealth, agent_type=MonetaryHolder),
            #"Holders Inventory (USD)": partial(compute_inventory_wealth, agent_type=MonetaryHolder, in_usd=True),
        })
        self.data_collector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters={"Wealth": "wealth"},
        )

        self.running = True
        self.data_collector.collect(self)

    def step(self):
        """
        A model step. Used for collecting simulation and advancing the schedule
        """
        from agents import (
            MonetaryArbitrageur,
            MonetarySniper,
            MonetaryLiquidator,
        )
        self.data_collector.collect(self)

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
