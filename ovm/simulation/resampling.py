import os
import typing as tp

import numpy as np
import pandas as pd
from recombinator import stationary_bootstrap

from ovm.historical.data_io import (
    load_price_histories,
    construct_closing_price_df,
    construct_series_name_to_closing_price_map,
    compute_log_return_df
)

from ovm.paths import (
    HistoricalDataSource,
    construct_simulated_data_directory

)

from ovm.simulation.bootstrap import convert_and_ceil_time_period_from_seconds_to_number_of_periods
from ovm.time_resolution import TimeResolution


def load_log_returns(series_names: tp.Sequence[str],
                     period_length_in_seconds: float,
                     directory_path: tp.Optional[str] = None) \
        -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # load price history
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


def simulate_new_return_series_via_bootstrap(
        input_log_return_df: pd.DataFrame,
        time_resolution: TimeResolution,
        block_length_in_seconds: float,
        simulated_sample_length_in_steps_in_seconds: float,
        number_of_paths: int = 1) -> np.ndarray:
    block_length_in_periods = \
        convert_and_ceil_time_period_from_seconds_to_number_of_periods(
            time_periods_in_seconds=block_length_in_seconds,
            period_length_in_seconds=time_resolution.in_seconds)

    simulated_sample_length_in_periods = \
        convert_and_ceil_time_period_from_seconds_to_number_of_periods(
            time_periods_in_seconds=simulated_sample_length_in_steps_in_seconds,
            period_length_in_seconds=time_resolution.in_seconds)

    # resample returns
    simulated_log_returns = \
        stationary_bootstrap(
            input_log_return_df.values,
            block_length=block_length_in_periods,
            replications=number_of_paths,
            sub_sample_length=simulated_sample_length_in_periods)

    return simulated_log_returns


def simulate_new_price_series_via_bootstrap(
        initial_prices: pd.Series,
        input_log_return_df: pd.DataFrame,
        time_resolution: TimeResolution,
        block_length_in_seconds: float,
        simulated_sample_length_in_steps_in_seconds: float,
        number_of_paths: int = 1) -> np.ndarray:
    simulated_log_returns = \
        simulate_new_return_series_via_bootstrap(
            input_log_return_df=input_log_return_df,
            time_resolution=time_resolution,
            block_length_in_seconds=block_length_in_seconds,
            simulated_sample_length_in_steps_in_seconds=simulated_sample_length_in_steps_in_seconds,
            number_of_paths=number_of_paths)

    simulated_prices = \
        convert_log_simulated_returns_to_prices(simulated_log_returns=simulated_log_returns,
                                                initial_prices=initial_prices.values)

    return simulated_prices


def store_simulated_price_series_in_output_directory(
        series_names: tp.Sequence[str],
        simulated_prices: np.ndarray,  # a numpy array with shape (1, length, number of price series)
        time_resolution: TimeResolution,
        historical_data_source: HistoricalDataSource,
        numpy_seed: int):
    assert simulated_prices.ndim == 3
    assert simulated_prices.shape[0] == 1

    simulation_output_directory = \
        os.path.join(construct_simulated_data_directory(
                        historical_data_source=historical_data_source,
                        time_resolution=time_resolution),
                     f'sims-{numpy_seed}')

    print(f'simulation_output_directory={simulation_output_directory}')

    if not os.path.exists(simulation_output_directory):
        os.makedirs(simulation_output_directory)

    for series in series_names:
        simulation_output_filepath = os.path.join(simulation_output_directory,
                                                  f'sim-{series}.csv')

        pd.DataFrame(simulated_prices[0, 1:, series_names.index(series)]).to_csv(
            simulation_output_filepath,
            index=False
        )
