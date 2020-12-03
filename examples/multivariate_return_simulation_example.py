import typing as tp

import numpy as np
import os
import pandas as pd

from ovm.bootstrap import (
    convert_block_length_from_seconds_to_blocks,
    plot_multivariate_simulation
)

from ovm.historical_data_io import (
    load_price_histories,
    construct_series_name_to_closing_price_map,
    construct_closing_price_df,
    compute_log_return_df
)

from ovm.utils import TimeResolution

from recombinator import (
    stationary_bootstrap
)

# specify base directory for data files
base_directory = os.path.join('..', 'notebooks')

# use data sampled at 15 second intervals from FTX
time_resolution = TimeResolution.FIFTEEN_MINUTES
directory_path = os.path.join(base_directory, time_resolution.value)

# Make the block size approximately 6 hours
block_length = np.ceil(6 * 60 * 60 / time_resolution.in_seconds)

# Use ETH/USD exchange rate
price_history_file_name = 'ETH-USD'

# The exchange rate series we want to simulate returns for (in that order)
series_names = \
    ['BTC-USD',
     'ETH-USD',
     'YFI-USD',
     'BAL-USD',
     'COMP-USD',
     'LINK-USD']


def load_log_returns(series_names: tp.Sequence[str],
                     period_length_in_seconds: float,
                     directory_path: tp.Optional[str] = None) \
        -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # load price data
    series_name_to_price_history_map = \
        load_price_histories(series_names=series_names,
                             period_length_in_seconds=period_length_in_seconds,
                             directory_path=directory_path)

    # construct log returns
    closing_price_df = \
        (construct_closing_price_df(
            construct_series_name_to_closing_price_map(series_name_to_price_history_map))
         .loc[:, series_names].dropna())

    initial_prices = closing_price_df.iloc[0, :]

    log_return_df = compute_log_return_df(closing_price_df).dropna()

    return log_return_df, closing_price_df, initial_prices


def convert_log_simulated_returns_to_prices(simulated_log_returns: np.ndarray,
                                            initial_prices: np.ndarray):
    # simulated_log_returns is an array with shape
    # (number of monte carlo replications,
    # length of simulated time series,
    # number of cryptocurrencies simulated)
    return np.exp(np.log(initial_prices.reshape((1, 1, -1))) + simulated_log_returns.cumsum(axis=1))


def extract_single_cryptocurrency_path_from_simulated_data(
        simulated_data: np.ndarray,
        series_names: tp.Sequence[str],
        series_name: str,
        path: int = 0):
    return simulated_data[path, :, series_names.index(series_name)]


def main():
    log_return_df, closing_price_df, initial_prices = \
        load_log_returns(series_names=series_names,
                         period_length_in_seconds=time_resolution.in_seconds,
                         directory_path=directory_path)

    block_length = \
        convert_block_length_from_seconds_to_blocks(
            # block_length_in_seconds=15 * 60 * 60,  # 15 hour block length
            block_length_in_seconds=4 * 24 * 60 * 60,  # 4 day block length
            period_length_in_seconds=time_resolution.in_seconds)

    # resample returns
    simulated_log_returns = \
        stationary_bootstrap(
            log_return_df.values,
            block_length=block_length,
            replications=1)

    # convert to prices
    simulated_prices = \
        convert_log_simulated_returns_to_prices(simulated_log_returns=simulated_log_returns,
                                                initial_prices=initial_prices.values)

    # # plot the first monte carlo replication of log returns and prices
    plot_multivariate_simulation(simulated_data=simulated_log_returns,
                                 series_names=series_names,
                                 title='Log Returns')

    plot_multivariate_simulation(simulated_data=simulated_prices,
                                 series_names=series_names,
                                 title='Exchange Rates')

    # output the first simulated path of the simulated ETH-USD series:
    print(extract_single_cryptocurrency_path_from_simulated_data(simulated_prices,
                                                                 series_names=series_names,
                                                                 series_name='ETH-USD'))


if __name__ == '__main__':
    main()
