import ccxt
import itertools
from datetime import datetime, timedelta


def fetch_data(symbols, exchange_id='ftx', timeframe='15s', since=None, until=None, limit=1500, params={}):
    exchange = getattr(ccxt, exchange_id)()
    ms = exchange.parse_timeframe(timeframe) * 1000
    data = {}

    if not until:
        until = datetime.now().timestamp() * 1000

    if not since:
        since = until - limit*ms

    n = int((until-since) / (limit * ms))
    sinces = [ since + i*limit*ms for i in range(n) ]

    data = {
        symbol: list(itertools.chain.from_iterable([
            exchange.fetch_ohlcv(symbol, timeframe, since, limit, params)
            for since in sinces
        ]))
        for symbol in symbols
    }
    return data
