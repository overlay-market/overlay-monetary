import os
import typing as tp

import pandas as pd

from ovm.paths import SIMULATED_DATA_DIRECTORY
from ovm.tickers import (
    OVL_USD_TICKER,
    YFI_USD_TICKER
)

from ovm.time_resolution import TimeResolution


def construct_sims_map(
        data_sim_rng: int,
        time_resolution: TimeResolution,
        tickers: tp.Sequence[str],
        # for sim source, since OVL doesn't actually exist yet
        ovl_ticker: str = YFI_USD_TICKER,
        sim_data_dir: str = SIMULATED_DATA_DIRECTORY,
        verbose: bool = False) -> tp.Dict[str, tp.List[float]]:

    ticker_to_time_series_of_prices_map = {}
    for ticker in tickers:
        rpath = os.path.join(sim_data_dir,
                             str(time_resolution.value),
                             f'sims-{data_sim_rng}',
                             f'sim-{ticker}.csv')

        if verbose:
            print(f"Reading in sim simulation from {rpath}")
        f = pd.read_csv(rpath)
        if ticker == ovl_ticker:
            ticker_to_time_series_of_prices_map[OVL_USD_TICKER] = f.transpose().values.tolist()[
                                                                              0]
        else:
            ticker_to_time_series_of_prices_map[ticker] = f.transpose().values.tolist()[
                                                                      0]

    return ticker_to_time_series_of_prices_map
