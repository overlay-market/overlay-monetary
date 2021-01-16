from functools import partial
import typing as tp

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


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
        ticker_to_time_series_of_prices_map: tp.Dict[str, tp.List[float]],
        base_wealth: float,
        base_market_fee: float,
        base_max_leverage: float,
        liquidity: float,
        liquidity_supply_emission: tp.List[float],
        treasury: float,
        sampling_interval: int
    ):
        from agents import (
            MonetaryArbitrageur,
            MonetaryKeeper,
            MonetaryHolder,
            MonetaryTrader
        )

        from markets import MonetaryFMarket

        from reporters import (
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
        self.num_arbitraguers = num_arbitrageurs
        self.num_keepers = num_keepers
        self.num_traders = num_traders
        self.num_holders = num_holders
        self.base_wealth = base_wealth
        self.base_market_fee = base_market_fee
        self.base_max_leverage = base_max_leverage
        self.liquidity = liquidity
        self.treasury = treasury
        self.sampling_interval = sampling_interval
        self.supply = base_wealth * self.num_agents + liquidity
        self.schedule = RandomActivation(self)
        self.ticker_to_time_series_of_prices_map = ticker_to_time_series_of_prices_map  # { k: [ time_series_of_prices ] }

        # Markets: Assume OVL-USD is in here and only have X-USD pairs for now ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        # if x + y = L/n and x/y = p; nx = (L/2n), ny = (L/2n), x*y = k = (px*L/2n)*(py*L/2n)
        n = len(ticker_to_time_series_of_prices_map.keys())
        prices_ovlusd = self.ticker_to_time_series_of_prices_map["OVL-USD"]
        print(f"OVL-USD first sim price: {prices_ovlusd[0]}")
        liquidity_weight = {
            list(ticker_to_time_series_of_prices_map.keys())[i]: 1
            for i in range(n)
        }
        print(f"liquidity_weight = {liquidity_weight}")

        # initialize futures markers (Overlay)
        self.ticker_to_futures_market_map = {
            ticker: MonetaryFMarket(
                unique_id=ticker,
                nx=(self.liquidity/(2*n))*liquidity_weight[ticker],
                ny=(self.liquidity/(2*n))*liquidity_weight[ticker],
                px=prices_ovlusd[0],  # px = n_usd/n_ovl
                py=prices_ovlusd[0]/time_series_of_prices[0],  # py = px/p
                base_fee=base_market_fee,
                max_leverage=base_max_leverage,
                model=self
            )
            for ticker, time_series_of_prices
            in self.ticker_to_time_series_of_prices_map.items()
        }

        tickers = list(self.ticker_to_futures_market_map.keys())
        for i in range(self.num_agents):
            fmarket = self.ticker_to_futures_market_map[tickers[i % len(tickers)]]
            base_curr = fmarket.unique_id[:-len("-USD")]
            base_quote_price = self.ticker_to_time_series_of_prices_map[fmarket.unique_id][0]
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
            leverage_max = (i % 3.0) + 1.0

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
            else:
                from .agents import MonetaryAgent
                agent = MonetaryAgent(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )

            print("MonetaryModel.init: Adding agent to schedule ...")
            print(f"MonetaryModel.init: agent type={type(agent)}")
            print(f"MonetaryModel.init: unique_id={agent.unique_id}")
            print(f"MonetaryModel.init: fmarket={agent.fmarket.unique_id}")
            print(f"MonetaryModel.init: leverage_max={agent.leverage_max}")
            print(f"MonetaryModel.init: inventory={agent.inventory}")

            self.schedule.add(agent)

        # simulation collector
        # TODO: Why are OVL-USD and ETH-USD futures markets not doing anything in terms of arb bots?
        # TODO: What happens if not enough OVL to sway the market time_series_of_prices on the platform? (i.e. all locked up)
        model_reporters = {
            f"d-{ticker}": partial(compute_price_difference, ticker=ticker)
            for ticker in tickers
        }
        model_reporters.update({
            f"s-{ticker}": partial(compute_spot_price, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            f"f-{ticker}": partial(compute_futures_price, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            f"Skew {ticker}": partial(compute_positional_imbalance_by_market, ticker=ticker)
            for ticker in tickers
        })

        model_reporters.update({
            "Gini": compute_gini,
            "Gini (Arbitrageurs)": partial(compute_gini, agent_type=MonetaryArbitrageur),
            "Supply": compute_supply,
            "Treasury": compute_treasury,
            "Liquidity": compute_liquidity,
            "Agent": partial(compute_wealth_for_agent_type, agent_type=None),
            "Arbitrageurs Wealth (OVL)": partial(compute_wealth_for_agent_type, agent_type=MonetaryArbitrageur),
            "Arbitrageurs Inventory (OVL)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryArbitrageur),
            "Arbitrageurs Inventory (USD)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryArbitrageur, in_usd=True),
            "Keepers Wealth (OVL)": partial(compute_wealth_for_agent_type, agent_type=MonetaryKeeper),
            "Keepers Inventory (OVL)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryKeeper),
            "Keepers Inventory (USD)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryKeeper, in_usd=True),
            "Traders Wealth (OVL)": partial(compute_wealth_for_agent_type, agent_type=MonetaryTrader),
            "Traders Inventory (OVL)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryKeeper),
            "Traders Inventory (USD)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryKeeper, in_usd=True),
            "Holders Wealth (OVL)": partial(compute_wealth_for_agent_type, agent_type=MonetaryHolder),
            "Holders Inventory (OVL)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryHolder),
            "Holders Inventory (USD)": partial(compute_inventory_wealth_for_agent_type, agent_type=MonetaryHolder, in_usd=True),
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
        self.data_collector.collect(self)
        self.schedule.step()
