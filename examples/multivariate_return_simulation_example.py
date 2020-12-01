from datetime import datetime
import typing as tp

import arch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyarrow
import scipy as sp
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from ovm.bootstrap import (
    convert_block_lenghts_to_seconds,
    convert_block_length_from_seconds_to_blocks,
    estimate_optimal_block_lengths_for_multiple_price_series,
    estimate_optimal_block_lengths_in_seconds_for_multiple_price_series
)

from ovm.historical_data_io import (
    PriceHistoryColumnNames as PHCN,
    compute_number_of_days_in_price_history,
    compute_log_returns_from_price_history,
    save_price_histories,
    load_price_history,
    load_price_histories,
    construct_series_name_to_closing_price_map,
    construct_closing_price_df,
    compute_log_return_df
)

from ovm.utils import TimeResolution

from recombinator import (
    stationary_bootstrap,
    tapered_block_bootstrap
)

from recombinator.optimal_block_length import optimal_block_length

# specify base directory for data files
base_directory = os.path.join('..', 'notebooks')

# use data sampled at 15 second intervals from FTX
time_resolution = TimeResolution.FIFTEEN_SECONDS
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
                     directory_path: tp.Optional[str] = None):
    # load price data
    series_name_to_price_history_map = \
        load_price_histories(series_names=series_names,
                             period_length_in_seconds=period_length_in_seconds,
                             directory_path=directory_path)

    # construct log returns
    log_return_df = \
        compute_log_return_df(
            construct_closing_price_df(
                construct_series_name_to_closing_price_map(series_name_to_price_history_map)))
    selected_log_return_df = log_return_df.loc[:, series_names].dropna()

    return selected_log_return_df


def main():
    selected_log_return_df = load_log_returns(series_names=series_names,
                                              period_length_in_seconds=time_resolution.in_seconds,
                                              directory_path=directory_path)

    block_length = \
        convert_block_length_from_seconds_to_blocks(
            15 * 60 * 60,  # 15 hour block length
            period_length_in_seconds=time_resolution.in_seconds)

    # resample returns
    bootstrap_simulation_result = \
        stationary_bootstrap(
            selected_log_return_df.values,
            block_length=block_length,
            replications=1)

    # plot the first monte carlo replication
    fig, axs = plt.subplots(bootstrap_simulation_result.shape[-1], figsize=(16, 9))
    for i, series_name in enumerate(series_names):
        axs[i].plot(bootstrap_simulation_result[0, :, i])
        axs[i].set_title(series_name)
    plt.show()


if __name__ == '__main__':
    main()
