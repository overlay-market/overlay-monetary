"""
Configure visualization elements and instantiate a server
"""
import pandas as pd
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
    ChartModule([{"Label": "Gini", "Color": "Black"}],
                data_collector_name='datacollector'),
]

# Load sims from csv files as arrays
tickers = ["ETH-USD", "COMP-USD", "LINK-USD", "YFI-USD"]
ovl_ticker = "YFI-USD"  # for sim source, since OVL doesn't actually exist yet
sims = {}
for ticker in tickers:
    f = pd.read_csv('./sims/sim-{}.csv'.format(ticker))
    if ticker == ovl_ticker:
        sims["OVL-USD"] = f.transpose().values.tolist()[0]
    else:
        sims[ticker] = f.transpose().values.tolist()[0]

total_supply = 100000  # OVL
base_wealth = 0.0001*100000  # OVL
base_market_fee = 0.0015
base_max_leverage = 10.0

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
