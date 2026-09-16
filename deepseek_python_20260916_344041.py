# Cambio en Signal.__init__: bandera explícita de drop
class Signal:
    def __init__(self, symbol, df, params=None, drop_last_bar=True):
        self.symbol = symbol
        self.params = params or DEFAULT_PARAMS
        if df is None or df.empty:
            self.df = df
        elif drop_last_bar:
            self.df = df.iloc[:-1]      # solo en vivo
        else:
            self.df = df                # backtest: usar todo lo conocido
        self._init_defaults()
        if self.df is not None and not self.df.empty and len(self.df) > 30:
            self._compute()