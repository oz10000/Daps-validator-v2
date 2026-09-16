"""
Pipeline mínimo end-to-end.
Ajustar SYMBOLS, TIMEFRAMES y PERIOD según tu infraestructura.
"""
from omega.validator import DataValidator
from omega.indicators import build_features
from omega.labels import triple_barrier
from omega.backtest import EventBacktest
from omega.monte_carlo import block_bootstrap
from omega.similarity import SimilaritySearch
from omega.signal_generator import SignalGenerator
from omega.certification import certification_report


def pipeline(data_provider, symbols, timeframe='5m', limit=5000):
    validator = DataValidator()
    engine = EventBacktest()
    gen = SignalGenerator()
    valid_data, reports = {}, {}

    for sym in symbols:
        df = data_provider(sym, timeframe, limit)
        df, rep = validator.validate(df, sym, timeframe)
        reports[sym] = rep
        if rep['ok']:
            valid_data[sym] = df

    # Features y labels por activo
    features = {s: build_features(d) for s, d in valid_data.items()}
    # ... construir señales, entrenar modelo, etc.

    return {
        'validator_reports': reports,
        'valid_data': valid_data,
        'features': features,
    }