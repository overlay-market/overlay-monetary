from datetime import datetime
import itertools
import typing as tp

import ccxt
from tqdm import tqdm

from ovm.time_resolution import TimeResolution


def fetch_data_for_single_symbol(symbol: str,
                                 exchange_id: str = 'ftx',
                                 timeframe: str = TimeResolution.FIFTEEN_SECONDS.value,
                                 since: int = None,
                                 until: int = None,
                                 limit: int = 1500,
                                 params: tp.Optional[tp.Dict] = None) -> tp.List[tp.List]:
    if params is None:
        params = {}

    exchange = getattr(ccxt, exchange_id)()
    ms = exchange.parse_timeframe(timeframe) * 1000

    if not until:
        until = datetime.now().timestamp() * 1000

    if not since:
        since = until - limit*ms

    n = int((until-since) / (limit * ms))
    sinces = [since + i*limit*ms for i in range(n)]

    result = \
        list(itertools.chain
             .from_iterable(
                [exchange.fetch_ohlcv(symbol, timeframe, since, limit, params)
                 for since
                 in tqdm(sinces)]))

    return result


def fetch_data(symbols: tp.Sequence[str],
               exchange_id: str = 'ftx',
               timeframe: str = TimeResolution.FIFTEEN_SECONDS.value,
               since: int = None,
               until: int = None,
               limit: int = 1500,
               params: tp.Optional[tp.Dict] = None,
               symbol_to_data_map: tp.Optional[tp.Dict[str, tp.List[tp.List]]] = None,
               verbose: bool = True) \
        -> tp.Dict[str, tp.List[tp.List]]:
    if symbol_to_data_map is None:
        symbol_to_data_map = {}

    for symbol in symbols:
        if verbose:
            print(f'fetching data for symbol {symbol}')

        symbol_to_data_map[symbol] = \
            fetch_data_for_single_symbol(symbol=symbol,
                                         exchange_id=exchange_id,
                                         timeframe=timeframe,
                                         since=since,
                                         until=until,
                                         limit=limit,
                                         params=params)

    return symbol_to_data_map
