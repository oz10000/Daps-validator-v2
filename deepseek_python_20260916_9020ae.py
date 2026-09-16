import numpy as np
import pandas as pd


def _tr(df):
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    return pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)


def atr(df, period=14):
    tr = _tr(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def rsi(df, period=14):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    ru = up.ewm(alpha=1 / period, adjust=False).mean()
    rd = dn.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - 100 / (1 + rs)


def macd(df, fast=12, slow=26, signal=9):
    ema_f = df['close'].ewm(span=fast, adjust=False).mean()
    ema_s = df['close'].ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def adx(df, period=14):
    high, low = df['high'], df['low']
    plus = high.diff()
    minus = -low.diff()
    plus = plus.where((plus > minus) & (plus > 0), 0.0)
    minus = minus.where((minus > plus) & (minus > 0), 0.0)
    tr = _tr(df)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / period, adjust=False).mean() / atr_s
    mdi = 100 * minus.ewm(alpha=1 / period, adjust=False).mean() / atr_s
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    return dx.fillna(0).ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def ker(df, period=10):
    c = df['close']
    return (c.diff(period).abs() / (c.diff().abs().rolling(period).sum() + 1e-9)).fillna(0)


def vwap(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    pv = (tp * df['volume']).rolling(period).sum()
    vv = df['volume'].rolling(period).sum().replace(0, np.nan)
    return (pv / vv).fillna(method='ffill')


def bollinger(df, period=20, mult=2):
    m = df['close'].rolling(period).mean()
    s = df['close'].rolling(period).std()
    return m + mult * s, m, m - mult * s


def keltner(df, period=20, mult=2):
    m = df['close'].ewm(span=period, adjust=False).mean()
    a = atr(df, period)
    return m + mult * a, m, m - mult * a


def donchian(df, period=20):
    return df['high'].rolling(period).max(), df['low'].rolling(period).min()


def obv(df):
    sign = np.sign(df['close'].diff()).fillna(0)
    return (sign * df['volume']).cumsum()


def mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - 100 / (1 + pos / (neg + 1e-12))


def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md + 1e-12)


def stochastic(df, k=14, d=3):
    hh = df['high'].rolling(k).max()
    ll = df['low'].rolling(k).min()
    K = 100 * (df['close'] - ll) / (hh - ll + 1e-12)
    return K, K.rolling(d).mean()


def williams_r(df, period=14):
    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    return -100 * (hh - df['close']) / (hh - ll + 1e-12)


def roc(df, period=10):
    return df['close'].pct_change(period) * 100


def volume_ratio(df, period=20):
    return df['volume'] / df['volume'].rolling(period).mean().replace(0, np.nan)


def volume_delta(df):
    return np.sign(df['close'].diff()) * df['volume']


def slope(s: pd.Series, n: int = 5) -> pd.Series:
    return (s - s.shift(n)) / n


def curvature(s: pd.Series, n: int = 5) -> pd.Series:
    return slope(s, n) - slope(s, n).shift(n)


def acceleration(s: pd.Series, n: int = 5) -> pd.Series:
    return slope(slope(s, n), n)


# ---- FeatureStore -----------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el FeatureStore. Devuelve un DataFrame indexado igual que df.
    Todas las columnas son causales: solo usan información hasta t inclusive.
    """
    f = pd.DataFrame(index=df.index)
    close = df['close']

    for p in (7, 14, 21):
        f[f'rsi_{p}'] = rsi(df, p)
        f[f'rsi_{p}_slope'] = slope(f[f'rsi_{p}'], 5)

    line, sig, hist = macd(df)
    f['macd_line'] = line
    f['macd_hist'] = hist
    f['macd_hist_slope'] = slope(hist, 5)

    f['ema_9'] = close.ewm(span=9, adjust=False).mean()
    f['ema_21'] = close.ewm(span=21, adjust=False).mean()
    f['ema_50'] = close.ewm(span=50, adjust=False).mean()
    f['ema_ratio'] = f['ema_9'] / (f['ema_50'] + 1e-12) - 1

    f['adx_14'] = adx(df, 14)
    f['adx_14_slope'] = slope(f['adx_14'], 5)

    a = atr(df, 14)
    f['atr_pct'] = a / (close + 1e-12)
    f['atr_slope'] = slope(f['atr_pct'], 5)

    f['ker_10'] = ker(df, 10)
    f['ker_slope'] = slope(f['ker_10'], 5)

    f['vwap_20_dev'] = close / (vwap(df, 20) + 1e-12) - 1

    bb_u, bb_m, bb_l = bollinger(df)
    f['bb_width'] = (bb_u - bb_l) / (bb_m + 1e-12)
    f['bb_pos'] = (close - bb_l) / (bb_u - bb_l + 1e-12)

    kc_u, kc_m, kc_l = keltner(df)
    f['kc_width'] = (kc_u - kc_l) / (kc_m + 1e-12)

    dc_u, dc_l = donchian(df)
    f['dc_pos'] = (close - dc_l) / (dc_u - dc_l + 1e-12)

    f['obv'] = obv(df)
    f['obv_slope'] = slope(f['obv'], 10)
    f['mfi_14'] = mfi(df, 14)
    f['cci_20'] = cci(df, 20)
    st_k, st_d = stochastic(df)
    f['stoch_k'] = st_k
    f['stoch_d'] = st_d
    f['williams_r'] = williams_r(df)
    f['roc_10'] = roc(df, 10)

    f['vol_ratio'] = volume_ratio(df, 20)
    f['vol_delta'] = volume_delta(df)
    f['vol_slope'] = slope(df['volume'], 5)
    f['vol_curv'] = curvature(df['volume'], 5)

    f['ret_1'] = close.pct_change(1)
    f['ret_5'] = close.pct_change(5)
    f['ret_20'] = close.pct_change(20)
    f['hour'] = df.index.hour
    f['dow'] = df.index.dayofweek

    return f.replace([np.inf, -np.inf], np.nan)