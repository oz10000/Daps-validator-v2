import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SimilaritySearch:
    """
    Búsqueda de vecinos históricos SIN usar futuro.
    Se debe indexar incrementalmente: al evaluar t, index contiene solo < t.
    """
    def __init__(self, feature_cols, k=100, min_neighbors=30):
        self.cols = feature_cols
        self.k = k
        self.min_neighbors = min_neighbors
        self.X = None
        self.meta = None
        self._mu = None
        self._sd = None

    def fit(self, features: pd.DataFrame, meta: pd.DataFrame):
        X = features[self.cols].dropna()
        meta = meta.loc[X.index]
        self._mu = X.mean()
        self._sd = X.std().replace(0, 1)
        self.X = ((X - self._mu) / self._sd).values
        self.meta = meta.reset_index(drop=True)
        return self

    def query(self, row_features: pd.Series, exclude_last=True):
        if self.X is None or len(self.X) < self.min_neighbors:
            return None
        v = ((row_features[self.cols] - self._mu) / self._sd).values.astype(float)
        if np.isnan(v).any():
            return None
        nn = NearestNeighbors(n_neighbors=min(self.k, len(self.X))).fit(self.X)
        dist, idx = nn.kneighbors(v.reshape(1, -1))
        idx = idx[0]
        neigh = self.meta.iloc[idx]
        n = len(neigh)
        if n < self.min_neighbors:
            return None
        wins = (neigh['pnl_pct'] > 0).sum()
        return {
            'n_neighbors': n,
            'win_rate': wins / n,
            'mean_pnl': neigh['pnl_pct'].mean(),
            'median_pnl': neigh['pnl_pct'].median(),
            'profit_factor': self._pf(neigh['pnl_pct']),
            'p_tp_first': (neigh['exit_reason'] == 'TP').mean(),
            'p_sl_first': (neigh['exit_reason'] == 'SL').mean(),
            'mean_dist': float(dist.mean()),
        }

    @staticmethod
    def _pf(pnl):
        gp = pnl[pnl > 0].sum()
        gl = abs(pnl[pnl <= 0].sum())
        return gp / gl if gl > 0 else float('inf')