# compute_pidelta_score — quitar la constante +0.20
quality = (
    0.30 * min(abs(trend_raw) * 10, 1.0) +
    0.25 * min(adx_v / 40.0, 1.0) +
    0.20 * min(max(ker_v, 0), 1.0) +
    0.10 * min(max(atr_rel, 0.5), 2.0) / 2.0 +
    0.15 * min(abs(momentum_raw) * 20, 1.0)
)   # suma exacta = 1.00