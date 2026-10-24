# TA Indicators

`src/ta/ta_context.py` provides two kinds of indicators:

- **Series functions:** compute a whole column from a pandas Series. Ported from
  `py-systemC-bot` `runtime/engine/indicator.py` and `numba_kernels.py`, with
  identical output (same seeding, same warm-up NaNs).
- **`TAContext`:** an incremental, closed-bar EMA/ATR state machine used by the
  structure lane. It steps one bar at a time and can be saved and restored.

## Setup

Numba is required (`numba>=0.64.0` in `requirements.txt`). Importing `ta` fails
without it; there is no pure-Python fallback. The first call to each kernel
compiles it, so expect a short delay once per process.

```python
from ta import ema, atr, rsi, supertrend, st_flip, st_stable, st_step_count
```

Inputs must be bar-ordered Series. Indicators use only the current and earlier
bars, so each value is known at that bar's close.

## Series functions

| Function | Returns | Formula | Warm-up (NaN rows) |
|---|---|---|---|
| `sma(close, period=50)` | `sma_{period}` | Rolling mean of `period` closes | first `period − 1` |
| `ema(close, period=20)` | `ema_{period}` | Seed = SMA of first `period` closes; then `EMA = (close − prev) · 2/(N+1) + prev` | first `period − 1` |
| `kama(close, er_period=10, fast=2, slow=30)` | `kama_{er}_{fast}_{slow}` | Efficiency ratio `|ΔN| / Σ|Δ1|`; `sc = (er·(fast_sc − slow_sc) + slow_sc)²`; `KAMA += sc · (close − KAMA)` | first `er_period − 1` |
| `atr(high, low, close, period=14)` | `atr_{period}` | `TR = max(H−L, |H−C₋₁|, |L−C₋₁|)`, `TR[0] = H−L`; seed = mean of first `period` TRs; then Wilder `ATR += (TR − ATR)/N` | first `period − 1` |
| `rsi(close, period=14)` | `rsi_{period}` | Wilder average gain/loss (seed = simple mean of first `period` changes); `RSI = 100 − 100/(1 + gain/loss)`; 100 when loss = 0 | first `period` |
| `supertrend(high, low, close, period=12, multiplier=3.0)` | `(st_line, st_dir)` | Bands `HL2 ± mult · ATR(period)`. Final upper band only falls and lower band only rises while price stays inside. Direction flips when close crosses the active band. `st_dir` is +1 bull / −1 bear; `st_line` is the lower band in bull, upper in bear. | first `period − 1` |
| `st_flip(st_dir)` | bool Series | `st_dir` differs from the previous bar (first valid bar counts as a flip) | — |
| `st_stable(st_dir, bars=3)` | `st_stable_{bars}` | Last `bars` directions all present and equal | False until `bars` valid rows |
| `st_step_count(st_line, st_dir)` | `st_step_count` (int) | Bars since the last flip where the line moved in trend direction (up in bull, down in bear); 0 on the flip bar | 0 |

## TAContext

```python
from ta import TAContext

ctx = TAContext(spans=(50, 100, 200), atr_period=14, band={"ema": 50, "mult": 2.0})
snapshot = ctx.step(row)   # row: pd.Series with open/high/low/close, index = bar close time
```

| Output | Formula |
|---|---|
| `ema[span]` | Same as `ema()`: SMA-seeded, `α = 2/(span+1)` |
| `atr[period]` | Same as `atr()`: SMA-seeded Wilder RMA |
| `band` | `upper/lower = ema[band.ema] ± mult · atr` once both are ready |

Rules:

- Bars must strictly increase in time; tz-aware timestamps are converted to naive UTC.
- `band` requires `atr_period`, and `band.ema` must be one of `spans`.
- `export_state()` / `load_state()` round-trip the full state (version 2; version 1
  states only carry EMA). Loading fails if spans, ATR period or band differ.

On the same bars, `TAContext` ends on the same EMA and ATR values as the
series functions.

## Not included

Regime labels, session labels, pivots/ChoCh, hypothesis triggers and policies
are still in `py-systemC-bot` (`runtime/engine/features.py`, `engine.py`,
`strategy.py`, `policy.py`).
