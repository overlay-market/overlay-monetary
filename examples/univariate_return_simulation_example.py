from datetime import datetime
import math
import typing as tp

import arch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyarrow
from recombinator import stationary_bootstrap
import scipy as sp
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
from statsmodels.tsa.stattools import adfuller

from ovm.garch_estimation import estimate_garch_parameters

from ovm.historical_data_io import (
    PriceHistoryColumnNames as PHCN,
    compute_scaling_factor,
    compute_scaled_log_returns,
    PriceHistory,
    compute_number_of_days_in_price_history,
    compute_log_returns_from_price_history,
    save_price_histories,
    load_price_history
)

from ovm.utils import TimeResolution

# specify base directory for data files
base_directory = os.path.join('..', 'notebooks')

# use data sampled at 15 second intervals from FTX
time_resolution = TimeResolution.FIFTEEN_SECONDS
directory_path = os.path.join(base_directory, time_resolution.value)

# Make the block size approximately 6 hours
block_length = math.ceil(6 * 60 * 60 / time_resolution.in_seconds)

# Use ETH/USD exchange rate
price_history_file_name = 'ETH-USD'


def main():
    # load price data
    price_history = \
        load_price_history(filename=price_history_file_name,
                           series_name=price_history_file_name,
                           directory_path=directory_path,
                           period_length_in_seconds=time_resolution.in_seconds)

    # resample returns
    bootstrap_simulation_result = \
        stationary_bootstrap(x=price_history.unscaled_log_returns,
                             block_length=block_length,
                             replications=1).squeeze()

    plt.plot(bootstrap_simulation_result)
    plt.show()


if __name__ == '__main__':
    main()
