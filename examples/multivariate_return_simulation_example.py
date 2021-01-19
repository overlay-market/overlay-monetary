from contexttimer import Timer
import numpy as np
import os

from ovm.simulation.bootstrap import plot_multivariate_simulation

from ovm.paths import (
    HISTORICAL_DATA_DIRECTORY,
    SIMULATED_DATA_DIRECTORY
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
directory_path = os.path.join(HISTORICAL_DATA_DIRECTORY, str(time_resolution.value))

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

        # block_length = \
        #     convert_and_ceil_time_period_from_seconds_to_number_of_periods(
        #         # block_length_in_seconds=15 * 60 * 60,  # 15 hour block length
        #         time_periods_in_seconds=4 * 24 * 60 * 60,  # 4 day block length
        #         period_length_in_seconds=time_resolution.in_seconds)

    print(f'Time to load all price series: {timer.elapsed} seconds')

    simulated_sample_length_in_steps = len(log_return_df)
    simulated_sample_length_in_seconds = simulated_sample_length_in_steps * time_resolution.in_seconds

    with Timer() as timer:
    #     # resample returns
    #     simulated_log_returns = \
    #         stationary_bootstrap(
    #             log_return_df.values,
    #             block_length=block_length,
    #             replications=number_of_paths,
    #             sub_sample_length=simulated_sample_length_in_steps)
    #
    #     # convert to prices
    #     simulated_prices = \
    #         convert_log_simulated_returns_to_prices(simulated_log_returns=simulated_log_returns,
    #                                                 initial_prices=initial_prices.values)

        simulated_prices = \
            simulate_new_price_series_via_bootstrap(
                initial_prices=initial_prices,
                input_log_return_df=log_return_df,
                time_resolution=time_resolution,
                block_length_in_seconds=4 * 24 * 60 * 60,  # 4 day block length
                simulated_sample_length_in_steps_in_seconds=simulated_sample_length_in_seconds,
                number_of_paths=1)

    print(f'Time to simulate {number_of_paths} paths of prices and returns: {timer.elapsed} seconds')

    # # plot the first monte carlo replication of log returns and prices
    # plot_multivariate_simulation(simulated_data=simulated_log_returns,
    #                              series_names=series_names,
    #                              title='Log Returns')

    plot_multivariate_simulation(simulated_data=simulated_prices,
                                 series_names=series_names,
                                 title='Exchange Rates')

    # # create output directory
    # simulation_output_directory = \
    #     construct_simulation_output_directory(time_resolution=time_resolution,
    #                                           numpy_seed=NUMPY_SEED,
    #                                           simulated_data_directory=SIMULATED_DATA_DIRECTORY)
    #
    # if not os.path.exists(simulation_output_directory):
    #     os.makedirs(simulation_output_directory)
    #
    # # output simulated paths to csv files ...
    # for series in series_names:
    #     simulation_output_filepath = os.path.join(simulation_output_directory,
    #                                               f'sim-{series}.csv')
    #
    #     pd.DataFrame(simulated_prices[0, 1:, series_names.index(series)]).to_csv(
    #         simulation_output_filepath,
    #         index=False
    #     )

    store_simulated_price_series_in_output_directory(
        series_names=series_names,
        simulated_prices=simulated_prices,
        time_resolution=time_resolution,
        numpy_seed=NUMPY_SEED,
        simulated_data_directory=SIMULATED_DATA_DIRECTORY)


if __name__ == '__main__':
    main()
