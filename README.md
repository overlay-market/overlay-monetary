# Overlay Monetary

## Fetching Data for Estimation and Simulation

```
from ovm.historical.utils import fetch_data
from datetime import datetime

symbols = [ 'BTC/USD', 'ETH/USD', 'AAVE/USD', 'YFI/USD', 'UNI/USD', 'BAL/USD', 'COMP/USD', 'LINK/USD', 'CREAM/USD', 'SUSHI/USD' ]

dur = 1000*86400
until = datetime.now().timestamp() * 1000
since = until - dur
data = fetch_data(symbols, since=since, until=until)
```

## Simulating Resampled Prices Output as CSV Files

```
python -m examples/multivariate_return_simulation_example.py
```

The seed can be controlled by setting the constant `NUMPY_SEED` in the script. It will save the csv files for that seed in folder named `f'sims-{NUMPY_SEED}'`.


## Running sims

Make sure you've generated needed simulated price paths above and stored them in a directory 'monetary/simulation/data/{DATA_FREQUENCY}' (e.g. 'monetary/simulation/data/15s/sims-42/sim-ETH-USD.csv').

Then run
```
$ cd ovm/monetary
$ mesa runserver
```

in the project's base directory.
