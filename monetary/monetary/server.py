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


chart_element = ChartModule([{"Label": "Gini",
                      "Color": "Black"}],
                    data_collector_name='datacollector')

# num_arbitrageurs, num_keepers, num_holders, sims, base_wealth

# Load sims from csv files as arrays
tickers = ["ETH-USD", "COMP-USD", "LINK-USD", "YFI-USD"]
ovl_ticker = "YFI-USD" # for sim source, since OVL doesn't actually exist yet
sims = {}
for ticker in tickers:
    f = pd.read_csv('./sims/sim-{}.csv'.format(ticker))
    if ticker == ovl_ticker:
        sims["OVL-USD"] = f.transpose().values.tolist()[0]
    else:
        sims[ticker] = f.transpose().values.tolist()[0]

model_kwargs = {
    "sims": sims,
    "num_arbitrageurs": len(sims.keys()) * 5,
    "num_keepers": len(sims.keys()),
    "num_holders": 0,
    "base_wealth": 100,
    "liquidity": 100*100 * (len(sims.keys()) * 5 + len(sims.keys()) + 0), # Setting liquidity = 100x agent-owned OVL for now; TODO: eventually have this be a function/array
    "sampling_interval": 240, # TODO: 1920 ... 8h with 15s blocks (sim data is every 15s)
}

server = ModularServer(
    MonetaryModel,
    [chart_element],
    "Monetary",
    model_kwargs,
)
