def _fix_symbol(self, symbol: str) -> str:
    if not symbol:
        raise ValueError("Símbolo vacío")
    if '/' in symbol:
        return symbol
    for quote in ('USDT', 'USDC', 'BUSD', 'USD'):
        if symbol.endswith(quote) and symbol != quote:
            return f"{symbol[:-len(quote)]}/{quote}"
    return f"{symbol}/USDT"