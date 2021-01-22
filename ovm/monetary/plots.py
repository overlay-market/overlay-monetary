import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ovm.monetary.plot_labels import (
    price_deviation_label,
    spot_price_label,
    futures_price_label,
    skew_label,
    inventory_wealth_ovl_label,
    inventory_wealth_usd_label,
    GINI_LABEL,
    GINI_ARBITRAGEURS_LABEL,
    SUPPLY_LABEL,
    TREASURY_LABEL,
    LIQUIDITY_LABEL,
)

from ovm.time_resolution import (
    TimeResolution,
    TimeScale
)

DEFAULT_FIGURE_SIZE = (16, 9)


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
        data_interval: int = 1):
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


def plot_price_deviations(
        model_vars_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
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
        data_interval=data_interval)

    plt.legend();
    plt.title('Deviation between Spot and Futures Prices');


def plot_skews(
        model_vars_df: pd.DataFrame,
        tickers: tp.Sequence[str],
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
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
        data_interval=data_interval)

    plt.legend();
    plt.title('Positional Imbalance in Terms of OVL');


def plot_single_variable_over_time(
        model_vars_df: pd.DataFrame,
        column_name: str,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
    begin_index, end_index, time_axis_to_plot = \
        get_indices_and_time_axis_to_plot(
            data_length=model_vars_df.shape[0],
            plot_time_scale=plot_time_scale,
            time_resolution=time_resolution,
            time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
            data_interval=data_interval)
    data_to_plot = model_vars_df.loc[:, column_name].values[begin_index:end_index]

    plt.figure(figsize=figure_size);
    plt.plot(time_axis_to_plot, data_to_plot);
    plt.xlabel(f'time in {plot_time_scale.value}');


def plot_supply(
        model_vars_df: pd.DataFrame,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        supply_label: str = SUPPLY_LABEL,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
    plot_single_variable_over_time(
        model_vars_df=model_vars_df,
        column_name=supply_label,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval)

    plt.title('OVL Supply');


def plot_treasury(
        model_vars_df: pd.DataFrame,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        treasury_label: str = TREASURY_LABEL,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
    plot_single_variable_over_time(
        model_vars_df=model_vars_df,
        column_name=treasury_label,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval)

    plt.title('Treasury');


def plot_liquidity(
        model_vars_df: pd.DataFrame,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        liquidity_label: str = LIQUIDITY_LABEL,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
    plot_single_variable_over_time(
        model_vars_df=model_vars_df,
        column_name=liquidity_label,
        plot_time_scale=plot_time_scale,
        time_resolution=time_resolution,
        time_interval_to_plot_in_seconds=time_interval_to_plot_in_seconds,
        figure_size=figure_size,
        data_interval=data_interval)

    plt.title('Liquidity');


def plot_spot_vs_futures_price(
        model_vars_df: pd.DataFrame,
        ticker: str,
        plot_time_scale: TimeScale,
        time_resolution: TimeResolution,
        time_interval_to_plot_in_seconds: tp.Optional[
            tp.Tuple[tp.Optional[float], tp.Optional[float]]] = None,
        figure_size: tp.Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        data_interval: int = 1):
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
        data_interval=data_interval)

    plt.legend();
    plt.title(f'Spot Price vs Futures Price for {ticker}')
