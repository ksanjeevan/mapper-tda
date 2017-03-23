import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
import explore_mapper as em

try:
    import params
except ImportError:
    import params_default as params

class EccentricityP(em.FilterFuctionTDA):

    def __init__(self, data):
        self.P = params.eccentricity_P
        self.var_vec = [v if v > 0 else 1. for v in np.var(data, axis=0)]


    def _fin_ecc(self, x, data):

        num = np.sum(cdist(x, data, metric='seuclidean',V=self.var_vec)[0])
        result = np.power(num, self.P) / len(data)
        return np.power(result, 1./self.P)

    def _inf_ecc(self, x, data):
        return np.max(cdist(x, data, metric='seuclidean',V=self.var_vec)[0])

    def filter_func(self, x, data):
        if self.P == 'inf':
            return self._inf_ecc(x, data)
        return self._fin_ecc(x, data)

