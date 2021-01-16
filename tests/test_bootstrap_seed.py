import os
import typing as tp

import numpy as np
import pandas as pd
import pytest

from ovm.bootstrap import convert_block_length_from_seconds_to_blocks

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
time_resolution = TimeResolution.FIFTEEN_SECONDS
directory_path = os.path.join(base_directory, time_resolution.value)

# Number of paths to simulate
number_of_paths = 1

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

# specify numpy seed for simulations
NUMPY_SEED = 42


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


block_length = \
    convert_block_length_from_seconds_to_blocks(
        block_length_in_seconds=4 * 24 * 60 * 60,  # 4 day block length
        period_length_in_seconds=time_resolution.in_seconds)


@pytest.fixture(scope='session')
def log_return_df() -> pd.DataFrame:
    log_return_df, closing_price_df, initial_prices = \
        load_log_returns(series_names=series_names,
                         period_length_in_seconds=time_resolution.in_seconds,
                         directory_path=directory_path)

    return log_return_df


def set_seed_and_simulate(log_return_df: pd.DataFrame):
    np.random.seed(NUMPY_SEED)

    return stationary_bootstrap(log_return_df.values,
                                block_length=block_length,
                                replications=number_of_paths)


def test_bootstrap_equality(log_return_df: pd.DataFrame):
    first_simulation = set_seed_and_simulate(log_return_df)
    second_simulation = set_seed_and_simulate(log_return_df)

    assert np.allclose(first_simulation, second_simulation)
