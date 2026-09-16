import numpy as np


def block_bootstrap(trades_pnl, n_iter=10000, block=5, capital=10000.0, seed=42):
    """
    Bootstrap por bloques: preserva autocorrelación serial de los trades.
    """
    pnl = np.asarray(trades_pnl, dtype=float) / 100.0
    n = len(pnl)
    if n < block * 2:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = n // block
    results = np.zeros((n_iter, 4))
    for i in range(n_iter):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([pnl[s:s + block] for s in starts])[:n]
        eq = capital * np.cumprod(1 + sample)
        ret = eq[-1] / capital - 1
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak).min()
        sh = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
        wr = (sample > 0).mean()
        results[i] = [ret, dd, sh, wr]
    return {
        'final_return_ci95': (np.percentile(results[:, 0], 2.5),
                              np.percentile(results[:, 0], 97.5)),
        'max_dd_ci95': (np.percentile(results[:, 1], 2.5),
                        np.percentile(results[:, 1], 97.5)),
        'sharpe_ci95': (np.percentile(results[:, 2], 2.5),
                        np.percentile(results[:, 2], 97.5)),
        'win_rate_ci95': (np.percentile(results[:, 3], 2.5),
                          np.percentile(results[:, 3], 97.5)),
        'prob_positive': float((results[:, 0] > 0).mean()),
    }