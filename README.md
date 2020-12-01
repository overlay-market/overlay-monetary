# Overlay Monetary

## Fetching data for fits

```
from ovm.utils import fetch_data
from datetime import datetime

symbols = [ 'BTC/USD', 'ETH/USD', 'AAVE/USD', 'YFI/USD', 'UNI/USD', 'BAL/USD', 'COMP/USD', 'LINK/USD', 'CREAM/USD', 'SUSHI/USD' ]

dur = 1000*86400
until = datetime.now().timestamp() * 1000
since = until - dur
data = fetch_data(symbols, since=since, until=until)
```
