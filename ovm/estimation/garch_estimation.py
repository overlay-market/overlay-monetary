from dataclasses import dataclass
import typing as tp

import arch
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm


from ovm.historical.data_io import PriceHistory

VOLATILITY_MODEL = 'garch'
DISTRIBUTION = 'StudentsT'


@dataclass(frozen=True)
class GARCHEstimationResult:
    log_returns: pd.Series
    model: arch.univariate.mean.HARX
    result: arch.univariate.base.ARCHModelResult
    price_history: PriceHistory

    def qq_plot(self):
        sm.qqplot(self.result.resid, dist=sp.stats.t(self.result.params['nu']))

    def simulate(self,
                 path_length: int,
                 initial_discard_length: int = 500,
                 initial_value: tp.Optional[tp.Union[float, np.ndarray]] = None) -> pd.Series:
        sim_mod = arch.arch_model(y=None,
                                  vol=VOLATILITY_MODEL,
                                  p=1,
                                  o=0,
                                  q=1,
                                  dist=DISTRIBUTION)

        scaled_simulation_result = \
            sim_mod.simulate(self.result.params,
                             nobs=path_length,
                             burn=initial_discard_length,
                             initial_value=initial_value)['data']

        return scaled_simulation_result / self.price_history.garch_scaling_factor


def estimate_garch_parameters(price_history: PriceHistory) \
        -> GARCHEstimationResult:
    log_returns = price_history.garch_scaled_log_returns

    garch_11_t = \
        arch.arch_model(log_returns,
                        vol='garch',
                        p=1,
                        o=0,
                        q=1,
                        dist='StudentsT'
                        )
    garch_11_t_fitted = garch_11_t.fit()

    return GARCHEstimationResult(log_returns=log_returns,
                                 model=garch_11_t,
                                 result=garch_11_t_fitted,
                                 price_history=price_history)
