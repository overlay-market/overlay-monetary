from enum import Enum
import os
from pathlib import Path

import ovm
from ovm.time_resolution import TimeResolution

DATA_INPUT_BASE_PATH_ENV_NAME = 'OVERLAY_MONETARY_INPUT_BASE_PATH'
DATA_OUTPUT_BASE_PATH_ENV_NAME = 'OVERLAY_MONETARY_OUTPUT_BASE_PATH'

if not os.environ.get(DATA_INPUT_BASE_PATH_ENV_NAME):
    DATA_INPUT_BASE_PATH = str(Path(os.path.dirname(ovm.__file__)).parents[0])
    print(f'environment variable {DATA_INPUT_BASE_PATH_ENV_NAME} not set defaulting to {DATA_INPUT_BASE_PATH}')
else:
    DATA_INPUT_BASE_PATH = os.environ.get(DATA_INPUT_BASE_PATH_ENV_NAME)
    print(f'environment variable {DATA_INPUT_BASE_PATH_ENV_NAME} set, using {DATA_INPUT_BASE_PATH}')

if not os.environ.get(DATA_OUTPUT_BASE_PATH_ENV_NAME):
    DATA_OUTPUT_BASE_PATH = str(Path(os.path.dirname(ovm.__file__)).parents[0])
    print(f'environment variable {DATA_INPUT_BASE_PATH_ENV_NAME} not set defaulting to {DATA_OUTPUT_BASE_PATH}')
else:
    DATA_OUTPUT_BASE_PATH = os.environ.get(DATA_INPUT_BASE_PATH_ENV_NAME)
    print(f'environment variable {DATA_INPUT_BASE_PATH_ENV_NAME} set, using {DATA_OUTPUT_BASE_PATH}')


DATA_DIRECTORY_NAME = 'data'
HISTORICAL_DIRECTORY_NAME = 'historical'
SIMULATED_DIRECTORY_NAME = 'simulation'

HISTORICAL_DATA_DIRECTORY = \
    os.path.join(DATA_INPUT_BASE_PATH, DATA_DIRECTORY_NAME, HISTORICAL_DIRECTORY_NAME)

SIMULATED_DATA_DIRECTORY = \
    os.path.join(DATA_INPUT_BASE_PATH, DATA_DIRECTORY_NAME, SIMULATED_DIRECTORY_NAME)

OUTPUT_DATA_DIRECTORY = \
    os.path.join(DATA_OUTPUT_BASE_PATH, 'agent_based_simulation_output_data')


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


def construct_simulated_data_directory(
        historical_data_source: HistoricalDataSource,
        time_resolution: TimeResolution) -> str:
    return os.path.join(SIMULATED_DATA_DIRECTORY,
                        historical_data_source.value,
                        time_resolution.value)
