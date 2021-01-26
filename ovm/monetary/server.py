"""
Configure visualization elements and instantiate a server
"""
import os
import logging
from pathlib import Path
from mesa.visualization.ModularVisualization import ModularServer

from ovm.monetary.chart_elements import construct_chart_elements
from ovm.time_resolution import TimeResolution
from ovm.tickers import (
    EOS_ETH_TICKER,
    MKR_ETH_TICKER,
    SNX_ETH_TICKER,
    XRP_ETH_TICKER,
    ETH_TICKER,
    ovl_quote_ticker,
)

from ovm.monetary.model import MonetaryModel
from ovm.monetary.data_io import construct_sims_map
from ovm.monetary.data_collection import DataCollectionOptions
from ovm.paths import (
    HistoricalDataSource,
    construct_simulated_data_directory
)

# set up logging
logger = logging.getLogger(__name__)

################################################################################
# Simulation Parameters
################################################################################
historical_data_source = HistoricalDataSource.KUCOIN
time_resolution = TimeResolution.ONE_MINUTE
DATA_SIM_RNG = 42

# Load sims from csv files as arrays
tickers = [EOS_ETH_TICKER,
           MKR_ETH_TICKER,
           SNX_ETH_TICKER,
           XRP_ETH_TICKER]

# sim_data_dir = \
#     os.path.join(str(Path(os.path.dirname(__file__)).parents[1]), 'data', 'simulation')
#
# sim_data_dir_2 = \
#     construct_simulated_data_directory(historical_data_source=HistoricalDataSource.KUCOIN,
#                                        time_resolution=time_resolution)

ovl_ticker = SNX_ETH_TICKER  # for sim source, since OVL doesn't actually exist yet
quote_ticker = ETH_TICKER
ovl_quote_ticker = ovl_quote_ticker(quote_ticker)

total_supply = 100000  # OVL
base_wealth = 0.001*100000  # OVL
base_market_fee = 0.0030
base_max_leverage = 10.0
base_liquidate_reward = 0.1
base_maintenance = 0.6
liquidity = 0.285*total_supply
time_liquidity_mine = time_resolution.steps_per_month_clamped
treasury = 0.0
sampling_interval = int(3600 / time_resolution.in_seconds)

num_arbitrageurs = int(total_supply*0.1/base_wealth)
num_keepers = int(total_supply*0.005/base_wealth)
num_traders = int(total_supply*0.005/base_wealth)
num_holders = int(total_supply*0.5/base_wealth)
num_snipers = int(total_supply*0.1/base_wealth)
num_liquidators = int(total_supply*0.005/base_wealth)
num_agents = num_arbitrageurs + num_keepers + \
    num_traders + num_holders + num_snipers + num_liquidators

data_collection_options = \
    DataCollectionOptions(compute_gini_coefficient=True,
                          compute_wealth=True,
                          compute_inventory_wealth=True)


################################################################################
# Construct ticker to price series map
################################################################################
# sims = construct_sims_map(data_sim_rng=DATA_SIM_RNG,
#                           time_resolution=time_resolution,
#                           tickers=tickers,
#                           sim_data_dir=sim_data_dir,
#                           ovl_ticker=SNX_ETH_TICKER,
#                           ovl_quote_ticker=ovl_quote_ticker)

sims = construct_sims_map(data_sim_rng=DATA_SIM_RNG,
                          time_resolution=time_resolution,
                          tickers=tickers,
                          historical_data_source=historical_data_source,
                          ovl_ticker=SNX_ETH_TICKER,
                          ovl_quote_ticker=ovl_quote_ticker)

################################################################################
# Set up liquidity supply emission
################################################################################
# For the first 30 days, emit until reach 100% of total supply; ONLY USE IN LIQUDITIY FOR NOW JUST AS TEST!
liquidity_supply_emission = [(0.51*total_supply/time_liquidity_mine)*i + 0.285*total_supply
                             for i
                             in range(time_liquidity_mine)]


################################################################################
# Construct Chart Elements
################################################################################
chart_elements = \
    construct_chart_elements(tickers=sims.keys(),
                             data_collection_options=data_collection_options)

################################################################################
# Start Server
################################################################################
# TODO: Vary these initial num_ ... numbers; for init, reference empirical #s already seeing for diff projects
model_kwargs = {
    "sims": sims,
    "quote_ticker": quote_ticker,
    "ovl_quote_ticker": ovl_quote_ticker,
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
    "treasury": treasury,
    "sampling_interval": sampling_interval,
}


server = ModularServer(
    MonetaryModel,
    chart_elements,
    "Monetary",
    model_kwargs,
)
