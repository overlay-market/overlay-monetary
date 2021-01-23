import os
import typing as tp

import numpy as np
import pandas as pd
import pytest

from ovm.simulation.bootstrap import convert_and_ceil_time_period_from_seconds_to_number_of_periods

from ovm.historical.data_io import (
    load_price_histories,
    construct_series_name_to_closing_price_map,
    construct_closing_price_df,
    compute_log_return_df
)

from ovm.paths import (
    SIMULATED_DATA_DIRECTORY,
    HistoricalDataSource,
    construct_historical_data_directory
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

from recombinator import (
    stationary_bootstrap
)

# use simulation sampled at 15 second intervals from FTX
time_resolution = TimeResolution.FIFTEEN_SECONDS
directory_path = \
    construct_historical_data_directory(
        historical_data_source=HistoricalDataSource.FTX,
        time_resolution=time_resolution)

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


def load_log_returns(series_names: tp.Sequence[str],
                     period_length_in_seconds: float,
                     directory_path: tp.Optional[str] = None) \
        -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # load price simulation
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
    convert_and_ceil_time_period_from_seconds_to_number_of_periods(
        time_periods_in_seconds=4 * 24 * 60 * 60,  # 4 day block length
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
