from dataclasses import dataclass
import os
import typing as tp

import numpy as np
import pandas as pd


class PriceHistoryColumnNames:
    START_TIME = 'start_time'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'


PHCN = PriceHistoryColumnNames


FTX_COLUMN_NAMES = [PHCN.START_TIME,
                    PHCN.OPEN,
                    PHCN.HIGH,
                    PHCN.LOW,
                    PHCN.CLOSE,
                    PHCN.VOLUME]

RawPriceHistory = tp.Sequence[tp.Tuple[int, float, float, float, float, float]]


def convert_raw_price_history_from_nested_list_to_dataframe(
        price_history: RawPriceHistory,
        set_time_index: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(data=price_history, columns=FTX_COLUMN_NAMES)
    if set_time_index:
        df.set_index(PHCN.START_TIME, inplace=True)

    return df


def convert_multiple_raw_price_histories_from_nested_lists_to_dict_of_dataframes(
        name_to_price_history_map: tp.Dict[str, RawPriceHistory],
        set_time_index: bool = True) -> tp.Dict[str, pd.DataFrame]:
    return {name: convert_raw_price_history_from_nested_list_to_dataframe(
                    price_history=price_history,
                    set_time_index=set_time_index)
            for name, price_history
            in name_to_price_history_map.items()}


def compute_number_of_days_in_price_history(price_history_df: pd.DataFrame,
                                            period_length_in_seconds: float) -> float:
    return len(price_history_df) / 60 / 60 / 24 * period_length_in_seconds


def _construct_file_path(filename: str,
                         directory_path: tp.Optional[str] = None) -> str:
    if directory_path is None:
        return filename
    else:
        return os.path.join(directory_path, filename)


def save_price_history_df(price_history_df: pd.DataFrame,
                          filename: str,
                          directory_path: tp.Optional[str] = None):
    filename = filename.replace('/', '-')
    file_path = _construct_file_path(filename=filename, directory_path=directory_path)
    price_history_df.to_parquet(file_path)


def save_price_histories(name_to_price_history_df_map: tp.Dict[str, pd.DataFrame],
                         directory_path: tp.Optional[str] = None):
    for name, price_history_df in name_to_price_history_df_map.items():
        save_price_history_df(price_history_df=price_history_df,
                              filename=name,
                              directory_path=directory_path)


def compute_log_returns_from_price_history(price_history_df: pd.DataFrame,
                                           period_length_in_seconds: float,
                                           name: tp.Optional[str] = None) -> pd.Series:
    log_returns = np.log(price_history_df[PHCN.CLOSE]).diff().dropna() * \
                  np.sqrt(365 * 24 * 60 * 60 / period_length_in_seconds)

    if name is not None:
        log_returns.name = name

    return log_returns


# def scale_log_returns_for_garch


@dataclass(frozen=False)
class PriceHistory:
    name: str
    price_history_df: pd.DataFrame
    period_length_in_seconds: float

    @property
    def unscaled_log_returns(self) -> pd.Series:
        log_returns = np.log(self.price_history_df[PHCN.CLOSE]).diff().dropna()
        log_returns.name = self.name

        return log_returns

        # return compute_log_returns_from_price_history(
        #             price_history_df=self.price_history_df,
        #             period_length_in_seconds=self.period_length_in_seconds,
        #             name=self.name)

    @property
    def garch_scaling_factor(self) -> float:
        return np.sqrt(365 * 24 * 60 * 60 / self.period_length_in_seconds)

    @property
    def garch_scaled_log_returns(self) -> pd.Series:
        return self.garch_scaling_factor * self.unscaled_log_returns


# class PriceHistory:
#     def __init__(self,
#                  name: str,
#                  price_history_df: pd.DataFrame,
#                  period_length_in_seconds: tp.Optional[float] = None):
#         self._name = name
#         self._price_history_df = price_history_df
#         self._period_length_in_seconds = period_length_in_seconds
#
#     @property
#     def name(self) -> str:
#         return self._name
#
#     @property
#     def price_history_df(self) -> pd.DataFrame:
#         return self._price_history_df


def load_price_history(filename: str,
                       series_name: str,
                       directory_path: str,
                       period_length_in_seconds: float) -> PriceHistory:
    file_path = _construct_file_path(filename=filename, directory_path=directory_path)
    price_history_df = pd.read_parquet(file_path)

    return PriceHistory(name=series_name,
                        price_history_df=price_history_df,
                        period_length_in_seconds=period_length_in_seconds)