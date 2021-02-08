import os
import random
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ovm.monetary.plot_labels import (
    price_deviation_label,
    spot_price_label,
    futures_price_label,
    skew_label,
    open_positions_label,
    SUPPLY_LABEL,
    TREASURY_LABEL,
    LIQUIDITY_LABEL,
)

from ovm.time_resolution import (
    TimeResolution,
    TimeScale
)

DEFAULT_FIGURE_SIZE = (16, 9)

SUPPLY_FILE_NAME = 'supply_plot'
TREASURY_FILE_NAME = 'treasury_plot'
LIQUIDITY_FILE_NAME = 'liquidity_plot'
PRICE_DEVIATIONS_FILE_NAME = 'price_deviations_plot'
OPEN_POSITIONS_FILE_NAME = 'open_positions_plot'
SKEWS_FILE_NAME = 'skews_plot'
SPOT_VS_FUTURES_PRICE_FILE_NAME_START = "spot_vs_futures_price"


def convert_time_in_seconds_to_index(
    time_resolution: TimeResolution,
    time_in_seconds: float) -> int:
    return int(np.floor(time_in_seconds / time_resolution.in_seconds))


def convert_time_interval_in_seconds_to_indices(
        data_length: int,
        time_resolution: TimeResolution,
        time_interval_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        data_interval: int = 1) -> tp.Tuple[int, int]:
    data_length_in_seconds = data_length * time_resolution.in_seconds * data_interval

    begin_time_in_seconds = 0
    end_time_in_seconds = data_length_in_seconds - 1

    if time_interval_in_seconds is not None:
        if time_interval_in_seconds[0] is not None:
            begin_time_in_seconds = time_interval_in_seconds[0]

        if time_interval_in_seconds[1] is not None:
            end_time_in_seconds = time_interval_in_seconds[1]

    begin_index = int(
        np.floor(begin_time_in_seconds / (data_interval * time_resolution.in_seconds)))
    end_index = int(
        np.floor(end_time_in_seconds / (data_interval * time_resolution.in_seconds))) + 1

    return begin_index, end_index


def construct_full_time_axis(
    data_length: int,
    time_resolution: TimeResolution,
    plot_time_scale: TimeScale = TimeScale.YEARS,
    data_interval: int = 1):
    return np.linspace(0, data_length * data_interval * time_resolution.in_seconds / plot_time_scale.in_seconds(), data_length)


def construct_time_axis(data_length: int,
                        time_resolution: TimeResolution,
                        time_interval_in_seconds: tp.Optional[
                            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
                        plot_time_scale: TimeScale = TimeScale.YEARS,
                        data_interval: int = 1) -> np.ndarray:
    begin_index, end_index = \
        convert_time_interval_in_seconds_to_indices(
            data_length=data_length,
            time_resolution=time_resolution,
            time_interval_in_seconds=time_interval_in_seconds,
            data_interval=data_interval)

    time_axis = np.linspace(0, data_length, data_length)[
                begin_index:end_index] * time_resolution.in_seconds / plot_time_scale.in_seconds() * data_interval

    return time_axis


def get_indices_and_time_axis_to_plot(
        data_length: int,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        data_interval: int = 1) -> tp.Tuple[int, int, np.ndarray]:
    begin_index, end_index = \
        convert_time_interval_in_seconds_to_indices(
            data_length=data_length,
            time_resolution=time_resolution,
            time_interval_in_seconds=time_interval_to_plot_in_seconds,
            data_interval=data_interval)

    full_time_axis = \
        construct_full_time_axis(
            data_length=data_length,
            time_resolution=time_resolution,
            plot_time_scale=plot_time_scale,
            data_interval=data_interval)

    time_axis_to_plot = full_time_axis[begin_index:end_index]

    return begin_index, end_index, time_axis_to_plot


def plot_multiple_variables_over_time(
        model_vars_df: pd.DataFrame,
        column_name_to_label_map: tp.Dict[str, str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        title: tp.Optional[str] = None,
        legend: bool = True,
        figure_save_path: tp.Optional[str] = None):
    begin_index, end_index, time_axis_to_plot = \
        get_indices_and_time_axis_to_plot(
            data_length=model_vars_df.shape[0],
            plot_time_scale=plot_time_scale,
            time_resolution=time_resolution,
            time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
            data_interval=data_interval)

    plt.figure(figsize=figure_size);
    for column_name, label in column_name_to_label_map.items():
        data_to_plot = model_vars_df.loc[:, column_name].values[begin_index:end_index]
        plt.plot(time_axis_to_plot, data_to_plot, label=label);

    plt.xlabel(f'time in {plot_time_scale.value}');

    if title is not None:
        plt.title(title);

    if legend:
        plt.legend();

    if figure_save_path is not None:
        plt.savefig(figure_save_path)


def plot_price_deviations(
        model_vars_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    column_name_to_label_map = \
        {price_deviation_label(ticker): price_deviation_label(ticker)
         for ticker in tickers}

    plot_multiple_variables_over_time(
        model_vars_df=model_vars_df,
        column_name_to_label_map=column_name_to_label_map,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title='Deviation between Spot and Futures Prices',
        legend=True,
        figure_save_path=figure_save_path)


def plot_skews(
        model_vars_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    column_name_to_label_map = \
        {skew_label(ticker): skew_label(ticker)
         for ticker in tickers}

    plot_multiple_variables_over_time(
        model_vars_df=model_vars_df,
        column_name_to_label_map=column_name_to_label_map,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title='Positional Imbalance in Terms of OVL',
        legend=True,
        figure_save_path=figure_save_path)


def plot_open_positions(
        model_vars_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    column_name_to_label_map = \
        {open_positions_label(ticker): open_positions_label(ticker)
         for ticker in tickers}

    plot_multiple_variables_over_time(
        model_vars_df=model_vars_df,
        column_name_to_label_map=column_name_to_label_map,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title='Number of Open Positions',
        legend=True,
        figure_save_path=figure_save_path)


def plot_single_variable_over_time_from_numpy_array(
        array: np.ndarray,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        title: tp.Optional[str] = None,
        figure_save_path: tp.Optional[str] = None):
    if array.ndim != 1:
        raise ValueError(f'array must be a NumPy array with one axis but has shape {array.shape}')

    begin_index, end_index, time_axis_to_plot = \
        get_indices_and_time_axis_to_plot(
            data_length=array.shape[0],
            plot_time_scale=plot_time_scale,
            time_resolution=time_resolution,
            time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
            data_interval=data_interval)
    data_to_plot = array[begin_index:end_index]

    plt.figure(figsize=figure_size);
    plt.plot(time_axis_to_plot, data_to_plot);
    plt.xlabel(f'time in {plot_time_scale.value}');
    if title is not None:
        plt.title(title)

    if figure_save_path is not None:
        plt.savefig(figure_save_path)


def plot_single_variable_over_time(
        model_vars_df: pd.DataFrame,
        column_name: str,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        title: tp.Optional[str] = None,
        figure_save_path: tp.Optional[str] = None):
    plot_single_variable_over_time_from_numpy_array(
        array=model_vars_df.loc[:, column_name].values,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title=title,
        figure_save_path=figure_save_path)


def plot_supply(
        model_vars_df: pd.DataFrame,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        supply_label: str = SUPPLY_LABEL,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    plot_single_variable_over_time(
        model_vars_df=model_vars_df,
        column_name=supply_label,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title='OVL Supply',
        figure_save_path=figure_save_path)


def plot_treasury(
        model_vars_df: pd.DataFrame,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        treasury_label: str = TREASURY_LABEL,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    plot_single_variable_over_time(
        model_vars_df=model_vars_df,
        column_name=treasury_label,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title='Treasury',
        figure_save_path=figure_save_path)


def plot_liquidity(
        model_vars_df: pd.DataFrame,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        liquidity_label: str = LIQUIDITY_LABEL,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    plot_single_variable_over_time(
        model_vars_df=model_vars_df,
        column_name=liquidity_label,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title='Liquidity',
        figure_save_path=figure_save_path)


def plot_spot_vs_futures_price(
        model_vars_df: pd.DataFrame,
        ticker: str,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds:
        tp.Optional[tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_path: tp.Optional[str] = None):
    column_name_to_label_map = \
        {futures_price_label(ticker): futures_price_label(ticker),
         spot_price_label(ticker): spot_price_label(ticker)}

    plot_multiple_variables_over_time(
        model_vars_df=model_vars_df,
        column_name_to_label_map=column_name_to_label_map,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        title=f'Spot Price vs Futures Price for {ticker}',
        legend=True,
        figure_save_path=figure_save_path)


def plot_all_model_level_variables(
        model_vars_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds:
        tp.Optional[tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1,
        figure_save_directory: tp.Optional[str] = None):

    if figure_save_directory is not None:
        if not os.path.exists(figure_save_directory):
            os.makedirs(figure_save_directory)

        supply_file_path = os.path.join(figure_save_directory, SUPPLY_FILE_NAME)
        treasury_file_path = os.path.join(figure_save_directory, TREASURY_FILE_NAME)
        liquidity_file_path = os.path.join(figure_save_directory, LIQUIDITY_FILE_NAME)
        price_deviations_file_path = \
            os.path.join(figure_save_directory, PRICE_DEVIATIONS_FILE_NAME)
        open_positions_file_path = \
            os.path.join(figure_save_directory, OPEN_POSITIONS_FILE_NAME)
        skews_file_path = \
            os.path.join(figure_save_directory, SKEWS_FILE_NAME)
    else:
        supply_file_path = None
        treasury_file_path = None
        liquidity_file_path = None
        price_deviations_file_path = None
        open_positions_file_path = None
        skews_file_path = None

    # plot_supply
    plot_supply(
        model_vars_df=model_vars_df,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        figure_save_path=supply_file_path)

    # plot_treasury
    plot_treasury(
        model_vars_df=model_vars_df,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        figure_save_path=treasury_file_path)

    # plot_liquidity
    plot_liquidity(
        model_vars_df=model_vars_df,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        figure_save_path=liquidity_file_path)

    # plot_price_deviations
    plot_price_deviations(
        model_vars_df=model_vars_df,
        tickers=tickers,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        figure_save_path=price_deviations_file_path)

    # plot_open_positions
    plot_open_positions(
        model_vars_df=model_vars_df,
        tickers=tickers,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        figure_save_path=open_positions_file_path)

    # plot_skews
    plot_skews(
        model_vars_df=model_vars_df,
        tickers=tickers,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval,
        figure_save_path=skews_file_path)

    # plot_spot_vs_futures_price
    for ticker in tickers:
        if figure_save_directory is not None:
            spot_vs_future_price_file_path = \
                os.path.join(figure_save_directory,
                             f'{SPOT_VS_FUTURES_PRICE_FILE_NAME_START}_{ticker}')
        else:
            spot_vs_future_price_file_path = None

        plot_spot_vs_futures_price(
            model_vars_df=model_vars_df,
            ticker=ticker,
            plot_time_scale=plot_time_scale,
            time_resolution=time_resolution,
            time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
            figure_size=figure_size,
            data_interval=data_interval,
            figure_save_path=spot_vs_future_price_file_path)


def random_color():
    return '#%02X%02X%02X' % (random.randint(0, 255),
                              random.randint(0, 255),
                              random.randint(0, 255))
