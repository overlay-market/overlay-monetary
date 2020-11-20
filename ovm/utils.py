import ccxt
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import typing as tp


def fetch_data(symbols: tp.Sequence[str],
               exchange_id: str = 'ftx',
               timeframe: str = '15s',
               since: int = None,
               until: int = None,
               limit: int = 1500,
               params: tp.Optional[tp.Dict] = None):
    if params is None:
        params = {}

    exchange = getattr(ccxt, exchange_id)()
    ms = exchange.parse_timeframe(timeframe) * 1000

    if not until:
        until = datetime.now().timestamp() * 1000

    if not since:
        since = until - limit*ms

    n = int((until-since) / (limit * ms))
    sinces = [ since + i*limit*ms for i in range(n) ]

    data = {
        symbol: list(itertools.chain.from_iterable([
            exchange.fetch_ohlcv(symbol, timeframe, since, limit, params)
            for since in sinces
        ]))
        for symbol in symbols
    }

    return data


PriceHistory = tp.Sequence[tp.Tuple[int, float, float, float, float, float]]
FTX_COLUMN_NAMES = ['start_time', 'open', 'high', 'low', 'close', 'volume']


def convert_price_history_from_nested_list_to_dataframe(
        price_history: PriceHistory,
        set_time_index: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(data=price_history, columns=FTX_COLUMN_NAMES)
    if set_time_index:
        df.set_index('start_time', inplace=True)

    return df


def convert_multiple_price_histories_from_nested_lists_to_dict_of_dataframes(
        name_to_price_history_map: tp.Dict[str, PriceHistory],
        set_time_index: bool = True) -> tp.Dict[str, pd.DataFrame]:
    return {name: convert_price_history_from_nested_list_to_dataframe(price_history=price_history,
                                                                      set_time_index=set_time_index)
            for name, price_history
            in name_to_price_history_map.items()}


def compute_number_of_days_in_price_history(price_history_df: pd.DataFrame,
                                            period_length_in_seconds: float) -> float:
    return len(price_history_df) / 60 / 60 / 24 * period_length_in_seconds


def save_price_history_df(name: str, price_history_df: pd.DataFrame):
    price_history_df.to_parquet(name.replace('/', '-'))


def save_price_histories(name_to_price_history_df_map: tp.Dict[str, pd.DataFrame]):
    for name, price_history_df in name_to_price_history_df_map.items():
        save_price_history_df(name=name, price_history_df=price_history_df)


def compute_log_returns_from_price_history(price_history_df: pd.DataFrame,
                                           period_length_in_seconds: float,
                                           name: tp.Optional[str] = None) -> pd.Series:
    log_returns = np.log(price_history_df['close']).diff().dropna() * np.sqrt(365 * 24 * 60 * 60 / period_length_in_seconds)
    if name is not None:
        log_returns.name = name
    return log_returns
