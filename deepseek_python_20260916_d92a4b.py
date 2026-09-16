from datetime import timedelta
import numpy as np
import pandas as pd


def walk_forward_optimize(signals_full, data_dict, param_grid,
                          train_days=30, test_days=7,
                          backtest_factory=None, select_metric='expectancy_pct'):
    """
    WFO real: en cada ventana optimiza sobre train, congela, evalúa en test.
    backtest_factory: callable(params) -> (engine, signal_builder)
    """
    signals_full = signals_full.sort_values('timestamp')
    start = signals_full['timestamp'].min()
    end = signals_full['timestamp'].max()

    results = []
    cursor = start
    best_params_history = []

    while cursor + timedelta(days=train_days + test_days) <= end:
        t_end = cursor + timedelta(days=train_days)
        v_end = t_end + timedelta(days=test_days)

        train_sigs = signals_full[(signals_full['timestamp'] >= cursor) &
                                  (signals_full['timestamp'] < t_end)]
        test_sigs = signals_full[(signals_full['timestamp'] >= t_end) &
                                 (signals_full['timestamp'] < v_end)]

        # Optimización en train
        best_score, best_params, best_engine = -np.inf, None, None
        for params in param_grid:
            engine, sig_builder = backtest_factory(params)
            s_train = sig_builder(train_sigs)
            tr = engine.simulate(s_train, data_dict)
            m = engine.metrics(tr)
            score = m.get(select_metric, -np.inf)
            if score > best_score:
                best_score, best_params, best_engine = score, params, engine

        # Evaluación en test con parámetros congelados
        _, sig_builder = backtest_factory(best_params)
        s_test = sig_builder(test_sigs)
        tr_test = best_engine.simulate(s_test, data_dict)
        m_test = best_engine.metrics(tr_test)

        results.append({
            'train_start': cursor, 'train_end': t_end,
            'test_start': t_end, 'test_end': v_end,
            'best_params': best_params,
            'train_metric': best_score,
            'test_metrics': m_test,
        })
        best_params_history.append(best_params)
        cursor += timedelta(days=test_days)

    return results