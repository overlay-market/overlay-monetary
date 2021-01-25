from enum import Enum
import os
from pathlib import Path

import ovm
from ovm.time_resolution import TimeResolution

DATA_BASE_PATH_ENV_NAME = 'OVERLAY_MONETARY_BASE_PATH'

if not os.environ.get(DATA_BASE_PATH_ENV_NAME):
    BASE_DIRECTORY = str(Path(os.path.dirname(ovm.__file__)).parents[0])
    print(f'environment variable {DATA_BASE_PATH_ENV_NAME} not set defaulting to {BASE_DIRECTORY}')
else:
    BASE_DIRECTORY = os.environ.get(DATA_BASE_PATH_ENV_NAME)
    print(f'environment variable {DATA_BASE_PATH_ENV_NAME} set, using {BASE_DIRECTORY}')

HISTORICAL_DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'data', 'historical')
SIMULATED_DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'data', 'simulation')


class HistoricalDataSource(Enum):
    BINANCE = 'binance'
    FTX = 'ftx'
    KUCOIN = 'kucoin'


def construct_historical_data_directory(
        historical_data_source: HistoricalDataSource,
        time_resolution: TimeResolution) -> str:
    return os.path.join(HISTORICAL_DATA_DIRECTORY,
                        historical_data_source.value,
                        time_resolution.value)
