import numpy as np
import pandas as pd


def triple_barrier(df, atr_pct, tp_mult=2.0, sl_mult=1.0, max_hold=30):
    """
    Etiqueta cada barra i según qué barrera se toca primero.
       +1 = TP-first
       -1 = SL-first
        0 = timeout
    Sin look-ahead: para la barra i usa solo info de i en adelante para el label.
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    for i in range(n - 1):
        if i + 1 >= n:
            break
        a = atr_pct.iloc[i]
        if a <= 0:
            continue
        tp = close[i] * (1 + tp_mult * a)
        sl = close[i] * (1 - sl_mult * a)
        end = min(i + max_hold, n - 1)
        hit_tp = hit_sl = False
        for j in range(i + 1, end + 1):
            if low[j] <= sl and not hit_tp:
                hit_sl = True; break
            if high[j] >= tp and not hit_sl:
                hit_tp = True; break
        if hit_tp and not hit_sl:
            labels[i] = 1
        elif hit_sl and not hit_tp:
            labels[i] = -1
        else:
            labels[i] = 0
    return pd.Series(labels, index=df.index, name='label')