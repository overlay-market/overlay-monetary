"""
Configure visualization elements and instantiate a server
"""

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
sims = {
    "OVLETH": [1.0],
    "ETHUSD": [1.0],
    "AAVEETH": [1.0],
}

model_kwargs = {
    "sims": sims,
    "num_arbitrageurs": len(sims.keys()) * 5,
    "num_keepers": len(sims.keys()),
    "num_holders": 0,
    "base_wealth": 100,
    "sampling_interval": 1920, # 8h with 15s blocks
}

server = ModularServer(
    MonetaryModel,
    [chart_element],
    "Monetary",
    model_kwargs,
)
