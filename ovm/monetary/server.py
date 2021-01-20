"""
Configure visualization elements and instantiate a server
"""
import logging
import random
import typing as tp

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from ovm.debug_level import DEBUG_LEVEL
from ovm.monetary.model import MonetaryModel
from ovm.monetary.data_io import construct_ticker_to_series_of_prices_map

from ovm.tickers import (
    ETH_USD_TICKER,
    COMP_USD_TICKER,
    LINK_USD_TICKER,
    YFI_USD_TICKER
)

from ovm.time_resolution import TimeResolution


# set up logging
logger = logging.getLogger(__name__)


def random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


TIME_RESOLUTION = TimeResolution.FIFTEEN_SECONDS
DATA_SIM_RNG = 42

# Constants
STEPS_MONTH = int((86400*30) / TIME_RESOLUTION.in_seconds)

# Load sims from csv files as arrays
TICKERS = [ETH_USD_TICKER,
           # not a long history of simulation (can we use a different token instead)
           COMP_USD_TICKER,
           # not a long history of simulation (can we use a different token instead)
           LINK_USD_TICKER,
           # less than half a year of simulation (can we use a different token instead)
           YFI_USD_TICKER
           ]

ticker_to_time_series_of_prices_map = \
    construct_ticker_to_series_of_prices_map(data_sim_rng=DATA_SIM_RNG,
                                             time_resolution=TIME_RESOLUTION,
                                             tickers=TICKERS)

total_supply = 100000  # OVL
base_wealth = 0.0003*100000  # OVL
base_market_fee = 0.0030
base_max_leverage = 10.0
base_liquidate_reward = 0.1
base_maintenance = 0.6
liquidity = 0.2*total_supply
time_liquidity_mine = STEPS_MONTH

# For the first 30 days, emit until reach 100% of total supply; ONLY USE IN LIQUDITIY FOR NOW JUST AS TEST!
liquidity_supply_emission = [(0.51*total_supply/time_liquidity_mine)*i + 0.285*total_supply
                             for i
                             in range(time_liquidity_mine)]

num_arbitrageurs = int(total_supply*0.14/base_wealth)
num_keepers = int(total_supply*0.005/base_wealth)
num_traders = int(total_supply*0.0/base_wealth)
num_holders = int(total_supply*0.5/base_wealth)
num_snipers = int(total_supply*0.15/base_wealth)
num_liquidators = int(total_supply*0.005/base_wealth)
num_agents = num_arbitrageurs + num_keepers + \
    num_traders + num_holders + num_snipers + num_liquidators

DATA_COLLECTOR_NAME = 'data_collector'


# TODO: Have separate lines for each bot along with the aggregate!
def construct_chart_elements(tickers) -> tp.List:
    chart_elements = [
        ChartModule([{"Label": "Supply", "Color": "Black"}],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": "Treasury", "Color": "Green"}],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": f"d-{ticker}", "Color": random_color()} for ticker in ticker_to_time_series_of_prices_map.keys()],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": f"Skew {ticker}", "Color": random_color()} for ticker in ticker_to_time_series_of_prices_map.keys()],
                    data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Arbitrageurs Wealth (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Arbitrageurs Inventory (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Arbitrageurs OVL Inventory (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Arbitrageurs Inventory (USD)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Snipers Wealth (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Snipers Inventory (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Snipers OVL Inventory (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Snipers Inventory (USD)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Traders Inventory (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Traders Inventory (USD)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Holders Inventory (OVL)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Holders Inventory (USD)", "Color": random_color()}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        #ChartModule([{"Label": "Liquidity", "Color": "Blue"}],
        #            data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": "Gini", "Color": "Black"}],
                    data_collector_name=DATA_COLLECTOR_NAME),
        #ChartModule([{"Label": "Gini (Arbitrageurs)", "Color": "Blue"}],
        #            data_collector_name=DATA_COLLECTOR_NAME),
    ]

    for ticker in tickers:
        chart_elements.append(
            ChartModule([
                {"Label": f"s-{ticker}", "Color": "Black"},
                {"Label": f"f-{ticker}", "Color": "Red"},
            ], data_collector_name='data_collector')
        )

    return chart_elements


# TODO: Vary these initial num_ ... numbers; for init, reference empirical #s already seeing for diff projects
MODEL_KWARGS = {
    "ticker_to_time_series_of_prices_map": ticker_to_time_series_of_prices_map,
    "num_arbitrageurs": num_arbitrageurs,
    "num_keepers": num_keepers,
    "num_traders": num_traders,
    "num_holders": num_holders,
    "num_snipers": num_snipers,
    "num_liquidators": num_liquidators,
    "base_wealth": base_wealth,
    "base_market_fee": base_market_fee,
    "base_max_leverage": base_max_leverage,
    "base_liquidate_reward": base_liquidate_reward,
    "base_maintenance": base_maintenance,
    # Setting liquidity = 100x agent-owned OVL for now; TODO: eventually have this be a function/array
    "liquidity": liquidity,
    "liquidity_supply_emission": liquidity_supply_emission,
    "treasury": 0.0,
    # TODO: 1920 ... 8h with 15s blocks (sim simulation is every 15s)
    "sampling_interval": 240,
}

if logging.root.level <= DEBUG_LEVEL:
    logger.debug("Model kwargs for initial conditions of sim:")
    logger.debug(f"num_arbitrageurs = {MODEL_KWARGS['num_arbitrageurs']}")
    logger.debug(f"num_keepers = {MODEL_KWARGS['num_keepers']}")
    logger.debug(f"num_traders = {MODEL_KWARGS['num_traders']}")
    logger.debug(f"num_holders = {MODEL_KWARGS['num_holders']}")
    logger.debug(f"base_wealth = {MODEL_KWARGS['base_wealth']}")

print("Model kwargs for initial conditions of sim:")
print(f"num_arbitrageurs = {MODEL_KWARGS['num_arbitrageurs']}")
print(f"num_snipers = {MODEL_KWARGS['num_snipers']}")
print(f"num_keepers = {MODEL_KWARGS['num_keepers']}")
print(f"num_traders = {MODEL_KWARGS['num_traders']}")
print(f"num_holders = {MODEL_KWARGS['num_holders']}")
print(f"num_liquidators = {MODEL_KWARGS['num_liquidators']}")
print(f"base_wealth = {MODEL_KWARGS['base_wealth']}")
print(f"total_supply = {total_supply}")
print(
    f"num_agents * base_wealth + liquidity = {num_agents*base_wealth + liquidity}")

chart_elements = construct_chart_elements(sims.keys())

server = ModularServer(
    MonetaryModel,
    chart_elements,
    "Monetary",
    MODEL_KWARGS,
)
