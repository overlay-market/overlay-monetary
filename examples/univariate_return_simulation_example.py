import math

import matplotlib.pyplot as plt
import os
from recombinator import stationary_bootstrap

from ovm.historical.data_io import load_price_history

from ovm.paths import HISTORICAL_DATA_DIRECTORY

from ovm.utils import TimeResolution

# use simulation sampled at 15 second intervals from FTX
time_resolution = TimeResolution.FIFTEEN_SECONDS
directory_path = os.path.join(HISTORICAL_DATA_DIRECTORY, time_resolution.value)

# Make the block size approximately 6 hours
block_length = math.ceil(6 * 60 * 60 / time_resolution.in_seconds)

# Use ETH/USD exchange rate
price_history_file_name = 'ETH-USD'


def main():
    # load price simulation
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
