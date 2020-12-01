import typing as tp

import numpy as np
import pandas as pd
from recombinator.optimal_block_length import (
    optimal_block_length,
    OptimalBlockLength
)

from tqdm import tqdm

from .historical_data_io import PriceHistory


def estimate_optimal_block_lengths_for_multiple_price_series(
        series_name_to_price_history_map: tp.Dict[str, PriceHistory]) \
        -> tp.Dict[str, OptimalBlockLength]:
    series_name_to_optimal_block_length_map = \
        {series_name: optimal_block_length(price_history.unscaled_log_returns.values)
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


def convert_block_length_from_seconds_to_blocks(block_length_in_seconds: float,
                                                period_length_in_seconds: float) -> int:
    return np.ceil(block_length_in_seconds / period_length_in_seconds)


def estimate_optimal_block_lengths_in_seconds_for_multiple_price_series(
        series_name_to_price_history_map: tp.Dict[str, PriceHistory],
        period_length_in_seconds: float) -> tp.Dict[str, float]:
    series_name_to_optimal_block_length_map = \
        estimate_optimal_block_lengths_for_multiple_price_series(series_name_to_price_history_map)

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
