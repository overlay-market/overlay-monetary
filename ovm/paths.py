from pathlib import Path
import os

import ovm


def base_directory() -> str:
    return str(Path(os.path.dirname(ovm.__file__)).parents[0])


def historical_data_directory():
    return os.path.join(base_directory(), 'notebooks')


def simulated_data_directory():
    return os.path.join(base_directory(), 'monetary', 'sims')
