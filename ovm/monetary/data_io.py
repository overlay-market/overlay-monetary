import os
import typing as tp

import numpy as np
import pandas as pd

from ovm.paths import SIMULATED_DATA_DIRECTORY
from ovm.tickers import (
    OVL_USD_TICKER,
    YFI_USD_TICKER
)

from ovm.time_resolution import TimeResolution


def construct_ticker_to_series_of_prices_map(
        data_sim_rng: int,
        time_resolution: TimeResolution,
        tickers: tp.Sequence[str],
        ovl_ticker: str = YFI_USD_TICKER,   # for sim source, since OVL doesn't actually exist yet
        verbose: bool = False) -> tp.Dict[str, np.ndarray]:

    ticker_to_time_series_of_prices_map = {}
    for ticker in tickers:
        rpath = os.path.join(SIMULATED_DATA_DIRECTORY,
                             str(time_resolution.value),
                             f'sims-{data_sim_rng}',
                             f'sim-{ticker}.csv')

        if verbose:
            print(f"Reading in sim simulation from {rpath}")
        f = pd.read_csv(rpath)
        if ticker == ovl_ticker:
            ticker_to_time_series_of_prices_map[OVL_USD_TICKER] = f.transpose().values.reshape((-1, ))
        else:
            ticker_to_time_series_of_prices_map[ticker] = f.transpose().values.reshape((-1, ))

    return ticker_to_time_series_of_prices_map


def construct_ticker_to_series_of_prices_map_from_simulated_prices(
        simulated_prices: np.ndarray,
        tickers: tp.Sequence[str]) \
        -> tp.Dict[str, np.ndarray]:
    ticker_to_time_series_of_prices_map = \
        {ticker: simulated_prices[0, :, i] for i, ticker in enumerate(tickers)}

    return ticker_to_time_series_of_prices_map
