from functools import partial
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from .agents import (
    MonetaryAgent, MonetaryArbitrageur, MonetaryKeeper, MonetaryHolder,
    MonetaryTrader,
)
from .markets import MonetaryFMarket
from .utils import (
    compute_gini, compute_price_diff, compute_fprice, compute_sprice,
    compute_supply, compute_liquidity, compute_treasury, compute_wealth
)


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
        num_arbitrageurs,
        num_keepers,
        num_traders,
        num_holders,
        sims,
        base_wealth,
        base_market_fee,
        base_max_leverage,
        liquidity,
        liquidity_supply_emission,
        treasury,
        sampling_interval
    ):
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
        self.sims = sims  # { k: [ prices ] }

        # Markets: Assume OVL-USD is in here and only have X-USD pairs for now ...
        # Spread liquidity from liquidity pool by 1/N for now ..
        # if x + y = L/n and x/y = p; nx = (L/2n), ny = (L/2n), x*y = k = (px*L/2n)*(py*L/2n)
        n = len(sims.keys())
        prices_ovlusd = self.sims["OVL-USD"]
        print(f"OVL-USD first sim price: {prices_ovlusd[0]}")
        liquidity_weight = {
            list(sims.keys())[i]: 1
            for i in range(n)
        }
        print("liquidity_weight", liquidity_weight)
        self.fmarkets = {
            ticker: MonetaryFMarket(
                unique_id=ticker,
                nx=(self.liquidity/(2*n))*liquidity_weight[ticker],
                ny=(self.liquidity/(2*n))*liquidity_weight[ticker],
                px=prices_ovlusd[0],  # px = n_usd/n_ovl
                py=prices_ovlusd[0]/prices[0],  # py = px/p
                base_fee=base_market_fee,
                max_leverage=base_max_leverage,
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
            inventory = {}
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
            # For leverage max, pick number between 1.0, 2.0, 3.0 (vary by agent)
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
                agent = MonetaryAgent(
                    unique_id=i,
                    model=self,
                    fmarket=fmarket,
                    inventory=inventory,
                    leverage_max=leverage_max
                )

            print("MonetaryModel.init: Adding agent to schedule ...")
            print("MonetaryModel.init: type", type(agent))
            print("MonetaryModel.init: unique_id", agent.unique_id)
            print("MonetaryModel.init: fmarket", agent.fmarket.unique_id)
            print("MonetaryModel.init: leverage_max", agent.leverage_max)
            print("MonetaryModel.init: inventory", agent.inventory)

            self.schedule.add(agent)

        # data collector
        # TODO: Why are OVL-USD and ETH-USD futures markets not doing anything in terms of arb bots?
        # TODO: What happens if not enough OVL to sway the market prices on the platform? (i.e. all locked up)
        model_reporters = {
            "{}-{}".format("d", ticker): partial(compute_price_diff, ticker=ticker)
            for ticker in tickers
        }
        model_reporters.update({
            "{}-{}".format("s", ticker): partial(compute_sprice, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            "{}-{}".format("f", ticker): partial(compute_fprice, ticker=ticker)
            for ticker in tickers
        })
        model_reporters.update({
            "Gini": compute_gini,
            "Supply": compute_supply,
            "Treasury": compute_treasury,
            "Liquidity": compute_liquidity,
            "Agent": partial(compute_wealth, agent_type=None),
            "Arbitrageurs": partial(compute_wealth, agent_type=MonetaryArbitrageur),
            "Keepers": partial(compute_wealth, agent_type=MonetaryKeeper),
            "Traders": partial(compute_wealth, agent_type=MonetaryTrader),
            "Holders": partial(compute_wealth, agent_type=MonetaryHolder),
        })
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters={"Wealth": "wealth"},
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        self.datacollector.collect(self)
        self.schedule.step()
