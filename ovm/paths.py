from pathlib import Path
import os

import ovm


BASE_DIRECTORY = str(Path(os.path.dirname(ovm.__file__)).parents[0])
HISTORICAL_DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'historical_data')
SIMULATED_DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'monetary', 'sims')
