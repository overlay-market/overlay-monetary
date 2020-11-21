from enum import Enum
from datetime import datetime
import itertools
import typing as tp

import ccxt


class TimeResolution(Enum):
    FIFTEEN_SECONDS = '15s'
    ONE_MINUTE = '60s'
    FIVE_MINUTES = '300s'
    FIFTEEN_MINUTES = '900s'
    ONE_HOUR = '3600s'
    FOUR_HOURS = '14400s'
    ONE_DAY = '86400s'

    @property
    def in_seconds(self) -> int:
        return int(self.value[0:-1])


def fetch_data(symbols: tp.Sequence[str],
               exchange_id: str = 'ftx',
               timeframe: str = TimeResolution.FIFTEEN_SECONDS.value,
               since: int = None,
               until: int = None,
               limit: int = 1500,
               params: tp.Optional[tp.Dict] = None):
    if params is None:
        params = {}

    exchange = getattr(ccxt, exchange_id)()
    ms = exchange.parse_timeframe(timeframe) * 1000

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


if __name__ == '__main__':
    tr = TimeResolution.FIFTEEN_SECONDS
    print(tr.in_seconds)
