"""
Configure visualization elements and instantiate a server
"""
import pandas as pd
import numpy as np
import random
from .model import MonetaryModel  # noqa

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule


def circle_portrayal_example(agent):
    if agent is None:
        return

    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "r": 0.5,
        "Color": "Pink",
    }
    return portrayal


def random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# Data freq in seconds
data_freq = {
    '15s': 15,
    '1m': 60,
    '5m': 300,
    '15m': 900,
}

DATA_FREQ_KEY = '15s'
DATA_SIM_RNG = 115
DATA_FREQ = data_freq[DATA_FREQ_KEY]

# Constants
STEPS_MONTH = int((86400*30)/DATA_FREQ)

# Load sims from csv files as arrays
tickers = ["ETH-USD", "COMP-USD", "LINK-USD", "YFI-USD"]
ovl_ticker = "YFI-USD"  # for sim source, since OVL doesn't actually exist yet
sims = {}
for ticker in tickers:
    rpath = './sims/{}/sims-{}/sim-{}.csv'.format(
        DATA_FREQ_KEY, DATA_SIM_RNG, ticker
    )
    print("Reading in sim data from", rpath)
    f = pd.read_csv(rpath)
    if ticker == ovl_ticker:
        sims["OVL-USD"] = f.transpose().values.tolist()[0]
    else:
        sims[ticker] = f.transpose().values.tolist()[0]

total_supply = 100000  # OVL
base_wealth = 0.0001*100000  # OVL
base_market_fee = 0.0030
base_max_leverage = 10.0
time_liquidity_mine = STEPS_MONTH
liquidity_supply_emission = [
    (0.51*total_supply/time_liquidity_mine)*i + 0.285*total_supply
    for i in range(time_liquidity_mine)
]  # For the first 30 days, emit until reach 100% of total supply; ONLY USE IN LIQUDITIY FOR NOW JUST AS TEST!


chart_elements = [
    ChartModule([
        {"Label": "Supply", "Color": "Black"},
    ], data_collector_name='datacollector'),
    ChartModule([
        {"Label": "Arbitrageurs", "Color": "Red"},
        {"Label": "Keepers", "Color": "Indigo"},
        {"Label": "Traders", "Color": "Violet"},
        {"Label": "Holders", "Color": "Black"},
        {"Label": "Liquidity", "Color": "Blue"},
    ], data_collector_name='datacollector'),
    ChartModule([
        {"Label": "Treasury", "Color": "Green"},
    ], data_collector_name='datacollector'),
    ChartModule([
        {"Label": "{}-{}".format("d", ticker), "Color": random_color()} for ticker in sims.keys()
    ], data_collector_name='datacollector'),
    ChartModule([{"Label": "Gini", "Color": "Black"}],
                data_collector_name='datacollector'),
]
for ticker in sims.keys():
    chart_elements.append(
        ChartModule([
            {"Label": "{}-{}".format("s", ticker), "Color": "Black"},
            {"Label": "{}-{}".format("f", ticker), "Color": "Red"},
        ], data_collector_name='datacollector')
    )


# TODO: Vary these initial num_ ... numbers; for init, reference empirical #s already seeing for diff projects
model_kwargs = {
    "sims": sims,
    "num_arbitrageurs": max(len(sims.keys()) * 5, int(total_supply*0.01/base_wealth)),
    "num_keepers": max(len(sims.keys()), int(total_supply*0.005/base_wealth)),
    "num_traders": int(total_supply*0.2/base_wealth),
    "num_holders": int(total_supply*0.5/base_wealth),
    "base_wealth": base_wealth,
    "base_market_fee": base_market_fee,
    "base_max_leverage": base_max_leverage,
    # Setting liquidity = 100x agent-owned OVL for now; TODO: eventually have this be a function/array
    "liquidity": 0.285*total_supply,
    "liquidity_supply_emission": liquidity_supply_emission,
    "treasury": 0.0,
    # TODO: 1920 ... 8h with 15s blocks (sim data is every 15s)
    "sampling_interval": 240,
}

print("Model kwargs for initial conditions of sim:")
print("num_arbitrageurs", model_kwargs["num_arbitrageurs"])
print("num_keepers", model_kwargs["num_keepers"])
print("num_traders", model_kwargs["num_traders"])
print("num_holders", model_kwargs["num_holders"])
print("base_wealth", model_kwargs["base_wealth"])

server = ModularServer(
    MonetaryModel,
    chart_elements,
    "Monetary",
    model_kwargs,
)
