import json
import uuid
from datetime import datetime, timezone
import pandas as pd


class SignalGenerator:
    """
    Combina:
      - probabilidad del modelo calibrado
      - probabilidad empírica de vecinos históricos
      - régimen
      - liquidez
    y emite una señal operativa con clasificación APROBADA / OBSERVACIÓN / DESAPROBADA.
    """

    def __init__(self,
                 w_model=0.5, w_sim=0.4, w_regime=0.1,
                 th_approved=0.55, th_observe=0.45,
                 min_neighbors=30, min_pf=1.2):
        self.w_model, self.w_sim, self.w_regime = w_model, w_sim, w_regime
        self.th_a, self.th_o = th_approved, th_observe
        self.min_neighbors = min_neighbors
        self.min_pf = min_pf

    def _regime_quality(self, regime: str) -> float:
        return {
            'Expansión': 1.0, 'Tendencia Fuerte': 0.85,
            'Tendencia Débil': 0.6, 'Chop': 0.25,
        }.get(regime, 0.4)

    def generate(self, symbol, exchange, direction, features,
                 p_model, sim_stats, regime, entry_price, atr_pct,
                 sl_mult=1.0, tp_mult=2.0,
                 be_trigger_pct=0.0015, trailing=None):
        rq = self._regime_quality(regime)
        if sim_stats is None:
            p_sim = 0.5
            pf_sim = 0.0
            n_neigh = 0
        else:
            p_sim = sim_stats['p_tp_first']
            pf_sim = sim_stats['profit_factor']
            n_neigh = sim_stats['n_neighbors']

        score = (self.w_model * p_model +
                 self.w_sim * p_sim +
                 self.w_regime * rq)

        # Clasificación
        approved = (score >= self.th_a and
                    n_neigh >= self.min_neighbors and
                    pf_sim >= self.min_pf)
        observe = (not approved and score >= self.th_o) or \
                  (n_neigh < self.min_neighbors)
        classification = 'APROBADA' if approved else \
                         'OBSERVACIÓN' if observe else 'DESAPROBADA'

        if direction == 'LONG':
            tp = entry_price * (1 + tp_mult * atr_pct)
            sl = entry_price * (1 - sl_mult * atr_pct)
        else:
            tp = entry_price * (1 - tp_mult * atr_pct)
            sl = entry_price * (1 + sl_mult * atr_pct)

        return {
            'signal_id': str(uuid.uuid4()),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol, 'exchange': exchange, 'direction': direction,
            'entry': {
                'price': float(entry_price),
                'order_type': 'limit',
                'probability': float(p_model),
                'score': float(score),
                'neighbors_used': int(n_neigh),
                'pf_similar': float(pf_sim),
                'reason': self._reason(p_model, p_sim, rq, n_neigh),
            },
            'targets': {
                'tp_price': float(tp),
                'tp_probability': float(p_sim) if sim_stats else None,
                'tp_mult_atr': tp_mult,
            },
            'risk': {
                'sl_price': float(sl),
                'sl_probability': float(1 - p_sim) if sim_stats else None,
                'sl_mult_atr': sl_mult,
            },
            'management': {
                'break_even_trigger_pct': be_trigger_pct,
                'trailing': trailing or {'enabled': False},
            },
            'regime': regime,
            'classification': classification,
        }

    @staticmethod
    def _reason(p_model, p_sim, rq, n):
        return (f"P(modelo)={p_model:.2f} · "
                f"P(TP-first|vecinos)={p_sim:.2f} · "
                f"regime_q={rq:.2f} · n_vecinos={n}")

    def export(self, signals, prefix='signals'):
        df = pd.json_normalize(signals)
        df.to_csv(f'{prefix}.csv', index=False)
        with open(f'{prefix}.json', 'w') as f:
            json.dump(signals, f, indent=2, default=str)
        with open(f'{prefix}_report.md', 'w') as f:
            f.write(self._md(signals))
        return f'{prefix}.json', f'{prefix}.csv', f'{prefix}_report.md'

    def _md(self, signals):
        lines = ['# OMEGA — Reporte de señales\n']
        approved = [s for s in signals if s['classification'] == 'APROBADA']
        obs = [s for s in signals if s['classification'] == 'OBSERVACIÓN']
        rej = [s for s in signals if s['classification'] == 'DESAPROBADA']
        lines.append(f'- Aprobadas: {len(approved)}')
        lines.append(f'- Observación: {len(obs)}')
        lines.append(f'- Desaprobadas: {len(rej)}\n')
        for s in approved:
            lines.append(f"## {s['symbol']} {s['direction']}")
            lines.append(f"- Score: {s['entry']['score']:.3f}")
            lines.append(f"- Razón: {s['entry']['reason']}")
            lines.append(f"- TP: {s['targets']['tp_price']:.4f}")
            lines.append(f"- SL: {s['risk']['sl_price']:.4f}\n")
        return '\n'.join(lines)