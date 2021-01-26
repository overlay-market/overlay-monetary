from contexttimer import Timer
import numpy as np
import os

from ovm.simulation.bootstrap import plot_multivariate_simulation

from ovm.paths import (
    SIMULATED_DATA_DIRECTORY,
    HistoricalDataSource,
    construct_historical_data_directory,
    construct_simulated_data_directory
)

from ovm.simulation.resampling import (
    load_log_returns,
    simulate_new_price_series_via_bootstrap,
    store_simulated_price_series_in_output_directory
)

from ovm.tickers import (
    BTC_USD_TICKER,
    ETH_USD_TICKER,
    YFI_USD_TICKER,
    BAL_USD_TICKER,
    COMP_USD_TICKER,
    LINK_USD_TICKER
)

from ovm.time_resolution import TimeResolution

# use simulation sampled at 15 second intervals from FTX
time_resolution = TimeResolution.FIFTEEN_SECONDS
historical_data_source = HistoricalDataSource.FTX

directory_path = \
    construct_historical_data_directory(
        historical_data_source=historical_data_source,
        time_resolution=time_resolution)

# Make the block size approximately 6 hours
block_length = np.ceil(6 * 60 * 60 / time_resolution.in_seconds)

# Number of paths to simulate
number_of_paths = 1

# The exchange rate series we want to simulate returns for (in that order)
series_names = \
    [BTC_USD_TICKER,
     ETH_USD_TICKER,
     YFI_USD_TICKER,
     BAL_USD_TICKER,
     COMP_USD_TICKER,
     LINK_USD_TICKER]

# specify numpy seed for simulations
NUMPY_SEED = 42


def main():
    np.random.seed(NUMPY_SEED)

    with Timer() as timer:
        log_return_df, closing_price_df, initial_prices = \
            load_log_returns(series_names=series_names,
                             period_length_in_seconds=time_resolution.in_seconds,
                             directory_path=directory_path)

    print(f'Time to load all price series: {timer.elapsed} seconds')

    simulated_sample_length_in_steps = len(log_return_df)
    simulated_sample_length_in_seconds = simulated_sample_length_in_steps * time_resolution.in_seconds

    with Timer() as timer:
        simulated_prices = \
            simulate_new_price_series_via_bootstrap(
                initial_prices=initial_prices,
                input_log_return_df=log_return_df,
                time_resolution=time_resolution,
                block_length_in_seconds=4 * 24 * 60 * 60,  # 4 day block length
                simulated_sample_length_in_steps_in_seconds=simulated_sample_length_in_seconds,
                number_of_paths=1)

    print(f'Time to simulate {number_of_paths} paths of prices and returns: {timer.elapsed} seconds')

    plot_multivariate_simulation(simulated_data=simulated_prices,
                                 series_names=series_names,
                                 time_resolution=time_resolution,
                                 title='Exchange Rates')

    store_simulated_price_series_in_output_directory(
        series_names=series_names,
        simulated_prices=simulated_prices,
        time_resolution=time_resolution,
        historical_data_source=historical_data_source,
        numpy_seed=NUMPY_SEED)


if __name__ == '__main__':
    main()
