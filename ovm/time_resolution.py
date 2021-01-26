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

    @property
    def steps_per_month(self) -> float:
        return (86400 * 30) / self.in_seconds

    @property
    def steps_per_month_clamped(self) -> int:
        return int(self.steps_per_month)

    @property
    def steps_per_year(self) -> float:
        return (365.25 * 24 * 60 * 60) / self.in_seconds

    @property
    def steps_per_year_clamped(self) -> int:
        return int(self.steps_per_year)


class TimeScale(Enum):
    SECONDS = 'seconds'
    MINUTES = 'minutes'
    HOURS = 'hours'
    DAYS = 'days'
    MONTHS = 'months'
    YEARS = 'years'

    def in_seconds(self) -> int:
        time_scale_to_seconds_map = \
            {TimeScale.SECONDS: 1,
             TimeScale.MINUTES: 60,
             TimeScale.HOURS: 60 * 60,
             TimeScale.DAYS: 24 * 60 * 60,
             TimeScale.MONTHS: 365.25 * 24 * 60 * 60 / 12,
             TimeScale.YEARS: 365.25 * 24 * 60 * 60}

        return time_scale_to_seconds_map[self]
