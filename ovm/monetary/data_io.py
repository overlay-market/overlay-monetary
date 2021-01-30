from dataclasses import dataclass
import os
import typing as tp
import numpy as np
import pandas as pd

from ovm.historical.data_io import PriceHistoryColumnNames as PHCN
from ovm.paths import (
    HistoricalDataSource,
    construct_simulated_data_directory,
    construct_historical_data_directory
)

from ovm.simulation.resampling import load_log_returns

from ovm.tickers import (
    OVL_USD_TICKER,
    YFI_USD_TICKER,
    ETH_TICKER,
    USD_TICKER,
)

from ovm.time_resolution import TimeResolution

# FILE_EXTENSION = 'csv'
FILE_EXTENSION = 'parq'


@dataclass(frozen=True)
class AgentBasedSimulationInputData:
    ticker_to_series_of_prices_map: tp.Dict[str, np.ndarray]
    time_resolution: TimeResolution
    historical_data_source: HistoricalDataSource
    ovl_ticker: str
    ovl_quote_ticker: str
    numpy_seed: tp.Optional[int] = None  # this is the same as data_sim_rng

    @property
    def is_resampled(self) -> bool:
        return self.numpy_seed is not None

    @property
    def tickers(self) -> tp.Tuple[str]:
        return tuple(self.ticker_to_series_of_prices_map.keys())


def convert_to_parq(csv_path: str):
    """Converts a csv file to parquet"""
    root, csv_ext = os.path.splitext(csv_path)
    if csv_ext.lower() == ".csv":
        parq_path = root + '.parq'
        data = pd.read_csv(csv_path)
        data.to_parquet(parq_path)


# ToDo: Merge this function into construct_abs_data_input_from_resampled_data when all call sites have switched
def construct_sims_map(
        data_sim_rng: int,
        time_resolution: TimeResolution,
        historical_data_source: HistoricalDataSource,
        tickers: tp.Sequence[str],
        # for sim source, since OVL doesn't actually exist yet
        ovl_ticker: str = YFI_USD_TICKER,
        ovl_quote_ticker: str = OVL_USD_TICKER,
        verbose: bool = False) -> tp.Dict[str, np.ndarray]:
    sim_data_dir = \
        construct_simulated_data_directory(
            historical_data_source=historical_data_source,
            time_resolution=time_resolution)

    ticker_to_time_series_of_prices_map = {}
    for ticker in tickers:
        rpath = os.path.join(sim_data_dir,
                             f'sims-{data_sim_rng}',
                             f'sim-{ticker}.{FILE_EXTENSION}')

        if verbose:
            print(f"Reading in sim simulation from {rpath}")
        # f = pd.read_csv(rpath)
        f = pd.read_parquet(rpath)
        if ticker == ovl_ticker:
            ticker_to_time_series_of_prices_map[ovl_quote_ticker] = f.transpose(
            ).values.reshape((-1, ))
        else:
            ticker_to_time_series_of_prices_map[ticker] = f.transpose(
            ).values.reshape((-1, ))

    return ticker_to_time_series_of_prices_map


def construct_abs_data_input_from_resampled_data(
        data_sim_rng: int,
        time_resolution: TimeResolution,
        historical_data_source: HistoricalDataSource,
        tickers: tp.Sequence[str],
        # for sim source, since OVL doesn't actually exist yet
        ovl_ticker: str = YFI_USD_TICKER,
        ovl_quote_ticker: str = OVL_USD_TICKER,
        verbose: bool = False) -> AgentBasedSimulationInputData:
    ticker_to_time_series_of_prices_map = \
        construct_sims_map(data_sim_rng=data_sim_rng,
                           time_resolution=time_resolution,
                           historical_data_source=historical_data_source,
                           tickers=tickers,
                           ovl_ticker=ovl_ticker,
                           ovl_quote_ticker=ovl_quote_ticker,
                           verbose=verbose)

    return AgentBasedSimulationInputData(
            ticker_to_series_of_prices_map=ticker_to_time_series_of_prices_map,
            time_resolution=time_resolution,
            historical_data_source=historical_data_source,
            ovl_ticker=ovl_ticker,
            ovl_quote_ticker=ovl_quote_ticker,
            numpy_seed=data_sim_rng)


# ToDo: Merge this function into construct_abs_data_input_from_historical_data when all call sites have switched
def construct_hist_map(
        time_resolution: TimeResolution,
        historical_data_source: HistoricalDataSource,
        tickers: tp.Sequence[str],
        ovl_ticker: str = YFI_USD_TICKER,
        ovl_quote_ticker: str = OVL_USD_TICKER,
        verbose: bool = False):
    hist_data_dir = \
        construct_historical_data_directory(
            historical_data_source=historical_data_source,
            time_resolution=time_resolution)

    close_prices = {}
    for ticker in tickers:
        rpath = os.path.join(hist_data_dir, f'{ticker}.{FILE_EXTENSION}')

        if verbose:
            print(f"Reading in sim history from {rpath}")

        df = pd.read_parquet(rpath)
        f = df[PHCN.CLOSE]
        if ticker == ovl_ticker:
            close_prices[ovl_quote_ticker] = f.transpose(
            ).values.reshape((-1, ))
        else:
            close_prices[ticker] = f.transpose().values.reshape((-1, ))

    return close_prices


def construct_abs_data_input_from_historical_data(
        time_resolution: TimeResolution,
        historical_data_source: HistoricalDataSource,
        tickers: tp.Sequence[str],
        ovl_ticker: str = YFI_USD_TICKER,
        ovl_quote_ticker: str = OVL_USD_TICKER,
        verbose: bool = False) -> AgentBasedSimulationInputData:
    ticker_to_series_of_prices_map = \
        construct_hist_map(time_resolution=time_resolution,
                           historical_data_source=historical_data_source,
                           tickers=tickers,
                           ovl_ticker=ovl_ticker,
                           ovl_quote_ticker=ovl_quote_ticker,
                           verbose=verbose)
    return AgentBasedSimulationInputData(
            ticker_to_series_of_prices_map=ticker_to_series_of_prices_map,
            time_resolution=time_resolution,
            historical_data_source=historical_data_source,
            ovl_ticker=ovl_ticker,
            ovl_quote_ticker=ovl_quote_ticker,
            numpy_seed=None)


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


def load_and_construct_ticker_to_series_of_prices_map_from_historical_prices(
        time_resolution: TimeResolution,
        historical_data_source: HistoricalDataSource,
        tickers: tp.Sequence[str],
        ovl_ticker: str,
        ovl_quote_ticker: str):
    directory_path = \
        construct_historical_data_directory(
            historical_data_source=historical_data_source,
            time_resolution=time_resolution)

    log_return_df, closing_price_df, initial_prices = \
        load_log_returns(series_names=tickers,
                         period_length_in_seconds=time_resolution.in_seconds,
                         directory_path=directory_path)

    result = construct_ticker_to_series_of_prices_map_from_historical_prices(
        historical_price_df=closing_price_df,
        tickers=tickers,
        ovl_ticker=ovl_ticker,
        ovl_quote_ticker=ovl_quote_ticker)

    return result
