from enum import Enum

TIME_RESOLUTION_TO_SECONDS_MAP = {
        '15s': 15,
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
    }


class TimeResolution(Enum):
    FIFTEEN_SECONDS = '15s'
    ONE_MINUTE = '1m'
    FIVE_MINUTES = '5m'
    FIFTEEN_MINUTES = '15m'
    ONE_HOUR = '1h'
    FOUR_HOURS = '4h'
    ONE_DAY = '1d'

    @property
    def in_seconds(self) -> int:
        return TIME_RESOLUTION_TO_SECONDS_MAP[self.value]