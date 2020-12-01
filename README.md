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

Optimal Block-Length in Seconds for Stationary Block Bootstrap Estimated from Data Samples at 15s Intervals:
{'BTC-USD': 8701.888227282909,
 'ETH-USD': 1839.007927474633,
 'AAVE-USD': 1339.8647558975172,
 'YFI-USD': 276.1829467479429,
 'UNI-USD': 580.8111835260969,
 'BAL-USD': 2732.46175494859,
 'COMP-USD': 28102.412799694423,
 'LINK-USD': 762.1127989781374,
 'CREAM-USD': 1703.575495633909,
 'SUSHI-USD': 2257.2420827965734}
 
Optimal Block-Length in Seconds for Stationary Block Bootstrap Estimated from Data Samples at 1m Intervals:
{'BTC-USD': 22515.06000855199,
 'ETH-USD': 8817.566245597449,
 'AAVE-USD': 1647.2639693363026,
 'YFI-USD': 11.13829033337917,
 'UNI-USD': 810.6021061829844,
 'BAL-USD': 5538.179172568409,
 'COMP-USD': 4613.967912181678,
 'LINK-USD': 454.601046245657,
 'CREAM-USD': 5314.288145310793,
 'SUSHI-USD': 2753.7217332161003}

Optimal Block-Length in Seconds for Stationary Block Bootstrap Estimated from Data Samples at 5m Intervals:
{'BTC-USD': 19129.421484121667,
 'ETH-USD': 24198.34767287368,
 'AAVE-USD': 1968.9077880069717,
 'YFI-USD': 9988.884089687495,
 'UNI-USD': 778.491406678681,
 'BAL-USD': 7228.886280304408,
 'COMP-USD': 9664.636713581926,
 'LINK-USD': 7042.39279758125,
 'CREAM-USD': 6221.812669040542,
 'SUSHI-USD': 12732.50545265789}
 
 Optimal Block-Length in Seconds for Stationary Block Bootstrap Estimated from Data Samples at 5m Intervals:
 {'BTC-USD': 53805.20232678347,
 'ETH-USD': 25813.264459000082,
 'AAVE-USD': 2390.547686238729,
 'YFI-USD': 1742.7813342914674,
 'UNI-USD': 3372.5955728885147,
 'BAL-USD': 5821.908768821018,
 'COMP-USD': 26408.615697511766,
 'LINK-USD': 10629.986144578435,
 'CREAM-USD': 3024.6533849087496,
 'SUSHI-USD': 1512.9629972802327}
 