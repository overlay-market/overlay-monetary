import os
import typing as tp
import numpy as np
import pandas as pd

from ovm.paths import SIMULATED_DATA_DIRECTORY
from ovm.tickers import (
    OVL_USD_TICKER,
    YFI_USD_TICKER,
    ETH_TICKER,
    USD_TICKER,
)

from ovm.time_resolution import TimeResolution


def construct_sims_map(
        data_sim_rng: int,
        time_resolution: TimeResolution,
        tickers: tp.Sequence[str],
        # for sim source, since OVL doesn't actually exist yet
        ovl_ticker: str = YFI_USD_TICKER,
        ovl_quote_ticker: str = OVL_USD_TICKER,
        sim_data_dir: str = SIMULATED_DATA_DIRECTORY,
        verbose: bool = False) -> tp.Dict[str, np.ndarray]:

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
            ticker_to_time_series_of_prices_map[ovl_quote_ticker] = f.transpose(
            ).values.reshape((-1, ))
        else:
            ticker_to_time_series_of_prices_map[ticker] = f.transpose(
            ).values.reshape((-1, ))

    return ticker_to_time_series_of_prices_map


def construct_ticker_to_series_of_prices_map_from_simulated_prices(
        simulated_prices: np.ndarray,
        tickers: tp.Sequence[str]) \
        -> tp.Dict[str, np.ndarray]:
    ticker_to_time_series_of_prices_map = \
        {ticker: simulated_prices[0, :, i] for i, ticker in enumerate(tickers)}

    return ticker_to_time_series_of_prices_map


def construct_ticker_to_series_of_prices_map_from_historical_prices(
        historical_price_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        ovl_ticker: str,
        ovl_quote_ticker: str) -> tp.Dict[str, np.ndarray]:
    """
    This function takes a dataframe with closing prices of different exchange rates.
    The name of the exchange rate w.r.t. to the quote currency to use as for the OVL exchange rate
    is given by ovl_ticker. This name is replaced with ovl_quote_ticker.
    """
    result = {ovl_quote_ticker
              if ticker == ovl_ticker
              else ticker: historical_price_df.loc[:, ticker].values
              for ticker
              in tickers}

    return result
