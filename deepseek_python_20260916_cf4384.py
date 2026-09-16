import pandas as pd


TF_DELTA = {
    '1m': '1min', '3m': '3min', '5m': '5min',
    '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D',
}


class DataValidator:
    MIN_WARMUP = 250
    MIN_POST_WARMUP = 500

    def validate(self, df: pd.DataFrame, symbol: str, timeframe: str):
        report = {
            'symbol': symbol, 'timeframe': timeframe,
            'rows_in': len(df), 'rows_out': 0,
            'duplicates': 0, 'gaps': [], 'nan_rows': 0,
            'invalid_rows': 0, 'ok': False, 'reason': '',
        }
        if df is None or df.empty:
            report['reason'] = 'empty'
            return df, report

        # 1. Duplicados
        dup = df.index.duplicated().sum()
        report['duplicates'] = int(dup)
        df = df[~df.index.duplicated(keep='last')]

        # 2. Orden
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # 3. Gaps
        expected = pd.Timedelta(TF_DELTA.get(timeframe, '5min'))
        gaps = df.index.to_series().diff()
        bad = gaps[gaps > expected * 1.5]
        report['gaps'] = [str(t) for t in bad.index[:50]]

        # 4. NaN / inf
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.replace([float('inf'), -float('inf')], float('nan'))
        nan_mask = df[cols].isna().any(axis=1)
        report['nan_rows'] = int(nan_mask.sum())
        df = df[~nan_mask]

        # 5. Coherencia
        bad = (df['close'] <= 0) | (df['high'] < df['low']) | \
              (df['high'] < df['close']) | (df['low'] > df['close'])
        report['invalid_rows'] = int(bad.sum())
        df = df[~bad]

        report['rows_out'] = len(df)

        # 6. Suficiencia
        need = self.MIN_WARMUP + self.MIN_POST_WARMUP
        if len(df) < need:
            report['reason'] = f'insufficient: {len(df)} < {need}'
            return df, report

        report['ok'] = True
        return df, report