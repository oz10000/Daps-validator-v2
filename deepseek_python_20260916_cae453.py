import numpy as np
import pandas as pd


class EventBacktest:
    """
    Simulación realista:
      - señal en barra i (cierre de i)
      - entrada en open de i+1 con slippage
      - SL/TP/BE/trailing evaluados barra a barra
      - si SL y TP se tocan en la misma barra, asume SL (conservador)
    """
    def __init__(self, fee=0.001, slippage=0.0005, spread_bps=2):
        self.fee = fee
        self.slippage = slippage
        self.spread = spread_bps / 10000.0

    def _entry(self, price, direction):
        c = self.slippage + self.spread / 2
        return price * (1 + c) if direction == 'LONG' else price * (1 - c)

    def _exit(self, price, direction):
        c = self.slippage + self.spread / 2
        return price * (1 - c) if direction == 'LONG' else price * (1 + c)

    def simulate(self, signals_df, data_dict, max_hold=60):
        trades = []
        for _, s in signals_df.iterrows():
            sym = s['symbol']
            df = data_dict.get(sym)
            if df is None or df.empty:
                continue
            try:
                i = df.index.get_indexer([s['timestamp']], method='nearest')[0]
            except Exception:
                continue
            if i < 0 or i + 1 >= len(df):
                continue

            direction = s['direction']
            entry_raw = df['open'].iloc[i + 1]
            entry = self._entry(entry_raw, direction)

            atr_pct = s.get('atr_pct', 0.01)
            sl_mult = s.get('sl_mult', 1.0)
            tp_mult = s.get('tp_mult', 2.0)
            be_trigger_pct = s.get('be_trigger_pct', 0.002)

            if direction == 'LONG':
                sl = entry * (1 - sl_mult * atr_pct)
                tp = entry * (1 + tp_mult * atr_pct)
            else:
                sl = entry * (1 + sl_mult * atr_pct)
                tp = entry * (1 - tp_mult * atr_pct)

            end = min(i + 1 + max_hold, len(df))
            exit_price = None
            exit_reason = 'TIME'
            be_active = False

            for j in range(i + 1, end):
                h, l = df['high'].iloc[j], df['low'].iloc[j]
                if direction == 'LONG':
                    if l <= sl:
                        exit_price = self._exit(sl, direction); exit_reason = 'SL'; break
                    if h >= tp:
                        exit_price = self._exit(tp, direction); exit_reason = 'TP'; break
                    if not be_active and (h / entry - 1) >= be_trigger_pct:
                        be_active = True
                        sl = entry
                    if be_active:
                        sl = max(sl, df['close'].iloc[j] * (1 - be_trigger_pct))
                else:
                    if h >= sl:
                        exit_price = self._exit(sl, direction); exit_reason = 'SL'; break
                    if l <= tp:
                        exit_price = self._exit(tp, direction); exit_reason = 'TP'; break
                    if not be_active and (1 - l / entry) >= be_trigger_pct:
                        be_active = True
                        sl = entry
                    if be_active:
                        sl = min(sl, df['close'].iloc[j] * (1 + be_trigger_pct))

            if exit_price is None:
                last = df['close'].iloc[end - 1]
                exit_price = self._exit(last, direction)

            gross = (exit_price - entry) / entry if direction == 'LONG' \
                    else (entry - exit_price) / entry
            net = gross - 2 * self.fee

            trades.append({
                'symbol': sym, 'direction': direction,
                'entry_time': df.index[i + 1],
                'exit_time': df.index[end - 1],
                'entry_price': entry, 'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': net * 100,
                'level': s.get('level', 'NA'),
                'regime': s.get('regime', 'NA'),
                'score': s.get('score', 0.0),
            })
        return pd.DataFrame(trades)

    def metrics(self, trades, capital=10000.0):
        if trades is None or trades.empty:
            return {'total_trades': 0}
        w = trades[trades['pnl_pct'] > 0]
        l = trades[trades['pnl_pct'] <= 0]
        n = len(trades)
        wr = len(w) / n
        gp = w['pnl_pct'].sum() if len(w) else 0
        gl = abs(l['pnl_pct'].sum()) if len(l) else 1e-9
        pf = gp / gl if gl > 0 else float('inf')
        exp = trades['pnl_pct'].mean()
        eq = capital * (1 + trades['pnl_pct'] / 100).cumprod()
        peak = eq.cummax()
        dd = ((eq - peak) / peak).min() * 100
        rets = trades['pnl_pct'] / 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        dn = rets[rets < 0].std()
        sortino = rets.mean() / dn * np.sqrt(252) if dn and dn > 0 else 0
        total_ret = trades['pnl_pct'].sum()
        calmar = total_ret / abs(dd) if dd != 0 else 0
        return {
            'total_trades': n, 'wins': len(w), 'losses': len(l),
            'win_rate': wr * 100, 'profit_factor': pf,
            'expectancy_pct': exp, 'total_return_pct': total_ret,
            'max_drawdown_pct': dd, 'sharpe': sharpe,
            'sortino': sortino, 'calmar': calmar,
        }