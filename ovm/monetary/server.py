"""
Configure visualization elements and instantiate a server
"""
import os
import logging
import random
import typing as tp
from pathlib import Path
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from ovm.time_resolution import TimeResolution
from ovm.tickers import (
    ETH_USD_TICKER,
    COMP_USD_TICKER,
    LINK_USD_TICKER,
    YFI_USD_TICKER
)

from ovm.monetary.model import MonetaryModel
from ovm.monetary.data_io import construct_sims_map
from ovm.monetary.options import DataCollectionOptions
from plot_labels import (
    agent_wealth_ovl_label,
    price_deviation_label,
    spot_price_label,
    futures_price_label,
    skew_label,
    position_count_label,
    inventory_wealth_ovl_label,
    inventory_wealth_usd_label,
    GINI_LABEL,
    GINI_ARBITRAGEURS_LABEL,
    SUPPLY_LABEL,
    TREASURY_LABEL,
    LIQUIDITY_LABEL
)


# set up logging
logger = logging.getLogger(__name__)


def random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


TIME_RESOLUTION = TimeResolution.FIFTEEN_SECONDS
DATA_SIM_RNG = 115

# Constants
STEPS_MONTH = int((86400*30)/TIME_RESOLUTION.in_seconds)

# Load sims from csv files as arrays
TICKERS = [ETH_USD_TICKER,
           # not a long history of simulation (can we use a different token instead)
           COMP_USD_TICKER,
           # not a long history of simulation (can we use a different token instead)
           LINK_USD_TICKER,
           # less than half a year of simulation (can we use a different token instead)
           YFI_USD_TICKER
           ]


BASE_DIR = str(Path(os.path.dirname(__file__)).parents[1])
SIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'simulation')
OVL_TICKER = "YFI-USD"  # for sim source, since OVL doesn't actually exist yet
sims = construct_sims_map(data_sim_rng=DATA_SIM_RNG,
                          time_resolution=TIME_RESOLUTION,
                          tickers=TICKERS,
                          sim_data_dir=SIM_DATA_DIR,
                          ovl_ticker=YFI_USD_TICKER)


total_supply = 100000  # OVL
base_wealth = 0.0002*100000  # OVL
base_market_fee = 0.0030
base_max_leverage = 10.0
base_liquidate_reward = 0.1
base_maintenance = 0.6
liquidity = 0.285*total_supply
time_liquidity_mine = STEPS_MONTH

# For the first 30 days, emit until reach 100% of total supply; ONLY USE IN LIQUDITIY FOR NOW JUST AS TEST!
liquidity_supply_emission = [(0.51*total_supply/time_liquidity_mine)*i + 0.285*total_supply
                             for i
                             in range(time_liquidity_mine)]

num_arbitrageurs = int(total_supply*0.1/base_wealth)
num_keepers = int(total_supply*0.005/base_wealth)
num_traders = int(total_supply*0.005/base_wealth)
num_holders = int(total_supply*0.5/base_wealth)
num_snipers = int(total_supply*0.1/base_wealth)
num_liquidators = int(total_supply*0.005/base_wealth)
num_agents = num_arbitrageurs + num_keepers + \
    num_traders + num_holders + num_snipers + num_liquidators

DATA_COLLECTOR_NAME = 'data_collector'
data_collection_options = \
    DataCollectionOptions(compute_gini_coefficient=True,
                          compute_wealth=True,
                          compute_inventory_wealth=True)


# TODO: Have separate lines for each bot along with the aggregate!
def construct_chart_elements(tickers, data_collection_options: DataCollectionOptions) -> tp.List:
    chart_elements = [
        ChartModule([{"Label": SUPPLY_LABEL, "Color": "Black"}],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": TREASURY_LABEL, "Color": "Green"}],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": price_deviation_label(ticker), "Color": random_color()}
                     for ticker
                     in sims.keys()],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": skew_label(ticker), "Color": random_color()}
                     for ticker
                     in sims.keys()],
                    data_collector_name=DATA_COLLECTOR_NAME),
        ChartModule([{"Label": position_count_label(ticker), "Color": random_color()}
                     for ticker
                     in sims.keys()],
                    data_collector_name=DATA_COLLECTOR_NAME),
    ]

    if data_collection_options.compute_inventory_wealth:
        for agent_type_name in ["Arbitrageurs", "Traders", "Holders", "Liquidators", "Snipers"]:
            chart_elements += [
                ChartModule([{"Label": agent_wealth_ovl_label(agent_type_name), "Color": random_color()}],
                            data_collector_name=DATA_COLLECTOR_NAME),
                #ChartModule([{"Label": inventory_wealth_ovl_label(agent_type_name), "Color": random_color()}],
                #            data_collector_name=DATA_COLLECTOR_NAME),
                #ChartModule([{"Label": inventory_wealth_usd_label(agent_type_name), "Color": random_color()}],
                #            data_collector_name=DATA_COLLECTOR_NAME),
            ]

    chart_elements += [
        ChartModule([{"Label": LIQUIDITY_LABEL, "Color": "Blue"}],
                    data_collector_name=DATA_COLLECTOR_NAME),
    ]

    if data_collection_options.compute_gini_coefficient:
        chart_elements += [
            ChartModule([{"Label": GINI_LABEL, "Color": "Black"}],
                        data_collector_name=DATA_COLLECTOR_NAME),
            ChartModule([{"Label": GINI_ARBITRAGEURS_LABEL, "Color": "Blue"}],
                        data_collector_name=DATA_COLLECTOR_NAME),
    ]

    for ticker in tickers:
        chart_elements.append(
            ChartModule([
                {"Label": spot_price_label(ticker), "Color": "Black"},
                {"Label": futures_price_label(ticker), "Color": "Red"},
            ], data_collector_name='data_collector')
        )

    return chart_elements


# TODO: Vary these initial num_ ... numbers; for init, reference empirical #s already seeing for diff projects
MODEL_KWARGS = {
    "sims": sims,
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

chart_elements = construct_chart_elements(sims.keys(),
                                          data_collection_options=data_collection_options)

server = ModularServer(
    MonetaryModel,
    chart_elements,
    "Monetary",
    MODEL_KWARGS,
)
