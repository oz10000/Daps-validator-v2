import pandas as pd


def certification_report(systems: dict, thresholds=None):
    """
    systems: {nombre: {'metrics': {...}, 'wfo': [...], 'mc': {...}, 'stress': {...}}}
    No certifica sin datos reales: si falta una métrica, lo marca como NO DISPONIBLE.
    """
    t = thresholds or {
        'demo':     {'pf': 1.2, 'dd': 20.0, 'sharpe': 0.5},
        'production': {'pf': 1.5, 'dd': 15.0, 'sharpe': 1.0},
    }
    rows = []
    for name, data in systems.items():
        m = data.get('metrics', {})
        wfo = data.get('wfo', [])
        wfo_pos = sum(1 for w in wfo
                      if w.get('test_metrics', {}).get('total_return_pct', 0) > 0)
        wfo_total = len(wfo)
        rows.append({
            'system': name,
            'trades': m.get('total_trades', 'N/D'),
            'win_rate': m.get('win_rate', 'N/D'),
            'pf': m.get('profit_factor', 'N/D'),
            'sharpe': m.get('sharpe', 'N/D'),
            'sortino': m.get('sortino', 'N/D'),
            'calmar': m.get('calmar', 'N/D'),
            'max_dd': m.get('max_drawdown_pct', 'N/D'),
            'expectancy': m.get('expectancy_pct', 'N/D'),
            'wfo_positive_pct': (100 * wfo_pos / wfo_total) if wfo_total else 'N/D',
            'demo_ready': _check(m, t['demo']),
            'prod_ready': _check(m, t['production']),
        })
    return pd.DataFrame(rows)


def _check(m, th):
    if not isinstance(m.get('profit_factor'), (int, float)):
        return 'NO DISPONIBLE'
    return ('SÍ' if (m.get('profit_factor', 0) >= th['pf'] and
                     abs(m.get('max_drawdown_pct', 999)) <= th['dd'] and
                     m.get('sharpe', 0) >= th['sharpe'])
            else 'NO')