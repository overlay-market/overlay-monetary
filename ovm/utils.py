import ccxt
import itertools
from datetime import datetime
import typing as tp


def fetch_data(symbols: tp.Sequence[str],
               exchange_id: str = 'ftx',
               timeframe: str = '15s',
               since: int = None,
               until: int = None,
               limit: int = 1500,
               params: tp.Optional[tp.Dict] = None):
    if params is None:
        params = {}

    exchange = getattr(ccxt, exchange_id)()
    ms = exchange.parse_timeframe(timeframe) * 1000
    # data = {}

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
