import typing as tp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from recombinator.optimal_block_length import (
    optimal_block_length,
    OptimalBlockLength
)

from tqdm import tqdm

from ovm.historical.data_io import PriceHistory
from ovm.time_resolution import (
    TimeResolution,
    TimeScale
)


def estimate_optimal_block_lengths_for_multiple_price_series(
        series_name_to_price_history_map: tp.Dict[str, PriceHistory],
        moment: float = 1.0) \
        -> tp.Dict[str, OptimalBlockLength]:
    series_name_to_optimal_block_length_map = \
        {series_name: optimal_block_length(price_history.unscaled_log_returns.values**moment)
         for series_name, price_history
         in tqdm(series_name_to_price_history_map.items())}

    return series_name_to_optimal_block_length_map


def convert_block_lenghts_to_seconds(
        series_name_to_optimal_block_length_map: tp.Dict[str, OptimalBlockLength],
        period_length_in_seconds: float) -> tp.Dict[str, float]:
    series_name_to_block_lengths_in_seconds_map = \
        {series_name: obl[0].b_star_sb * period_length_in_seconds
         for series_name, obl
         in series_name_to_optimal_block_length_map.items()}

    return series_name_to_block_lengths_in_seconds_map


def convert_and_ceil_time_period_from_seconds_to_number_of_periods(
        time_periods_in_seconds: float,
        period_length_in_seconds: float) -> int:
    # Example:
    # period length = 15s
    # block length = 60s
    # That means 4 steps
    return int(np.ceil(time_periods_in_seconds / period_length_in_seconds))


def estimate_optimal_block_lengths_in_seconds_for_multiple_price_series(
        series_name_to_price_history_map: tp.Dict[str, PriceHistory],
        period_length_in_seconds: float) -> tp.Dict[str, float]:
    series_name_to_optimal_block_length_map = \
        estimate_optimal_block_lengths_for_multiple_price_series(
            series_name_to_price_history_map)

    return convert_block_lenghts_to_seconds(series_name_to_optimal_block_length_map,
                                            period_length_in_seconds=period_length_in_seconds)


def max_optimal_block_length_in_seconds_for_selected_series(
        series_name_to_block_lengths_in_seconds_map: tp.Dict[str, float],
        selected_series_names: tp.Iterable[str]) -> float:
    max_optimal_block_length_in_seconds_for_selected_series = \
        max(block_lengths_in_seconds
            for series_name, block_lengths_in_seconds
            in series_name_to_block_lengths_in_seconds_map.items()
            if series_name in selected_series_names)

    return max_optimal_block_length_in_seconds_for_selected_series


def plot_multivariate_simulation(simulated_data: np.ndarray,
                                 series_names: tp.Sequence[str],
                                 time_resolution: TimeResolution,
                                 plot_time_scale: TimeScale = TimeScale.YEARS,
                                 title: tp.Optional[str] = None):
    # simulated_data is an array with shape
    # (number of monte carlo replications,
    # length of simulated time series,
    # number of cryptocurrencies simulated)

    # time_axis = \
    #     np.linspace(0, simulated_data.shape[1], simulated_data.shape[1]) * \
    #     time_resolution.in_seconds / (60 * 60 * 24 * 365.25)

    time_axis = \
        np.linspace(0, simulated_data.shape[1], simulated_data.shape[1]) * \
        time_resolution.in_seconds / plot_time_scale.in_seconds()

    fig, axs = plt.subplots(simulated_data.shape[-1], figsize=(16, 9))
    for i, series_name in enumerate(series_names):
        axs[i].plot(time_axis, simulated_data[0, :, i])
        axs[i].set_title(series_name)
        # axs[i].xaxis.set_label(f'Time in {plot_time_scale}')
        axs[i].set_xlabel(f'Time in {plot_time_scale}')

    if title is not None:
        fig.suptitle(title)

    # plt.xlabel(f'Time in {plot_time_scale}')

    plt.show()
