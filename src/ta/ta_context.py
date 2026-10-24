from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numba import njit


DEFAULT_EMA_SPANS = (50, 100, 200)


# ---------------------------------------------------------------------------
# Series indicators — ported from py-systemC-bot runtime/engine/indicator.py
# and numba_kernels.py. Same seeding and warm-up NaNs.
# ---------------------------------------------------------------------------

def sma(close: pd.Series, period: int = 50) -> pd.Series:
    """SMA = sum(close, N) / N. NaN for the first `period - 1` rows."""
    result = close.rolling(window=period, min_periods=period).mean()
    result.name = f"sma_{period}"
    return result


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """EMA seeded with the SMA of the first `period` bars, alpha = 2 / (N + 1)."""
    result = ema_kernel(close.to_numpy(dtype=float), period)
    return pd.Series(result, index=close.index, name=f"ema_{period}")


def kama(close: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman's Adaptive Moving Average, seeded at index `er_period - 1`."""
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    prices = close.to_numpy(dtype=float)
    n = len(prices)
    result = np.full(n, np.nan)

    seed = er_period - 1
    if seed < n:
        result[seed] = prices[seed]
    for i in range(seed + 1, n):
        direction = abs(prices[i] - prices[i - er_period])
        volatility = np.sum(np.abs(np.diff(prices[i - er_period: i + 1])))
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (prices[i] - result[i - 1])

    return pd.Series(result, index=close.index, name=f"kama_{er_period}_{fast}_{slow}")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR: TR = max(H-L, |H-C1|, |L-C1|), TR[0] = H-L, SMA-seeded RMA."""
    result = atr_kernel(
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        close.to_numpy(dtype=float),
        period,
    )
    return pd.Series(result, index=close.index, name=f"atr_{period}")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI = 100 - 100 / (1 + avg_gain / avg_loss). NaN for the first `period` rows."""
    result = rsi_kernel(close.to_numpy(dtype=float), period)
    return pd.Series(result, index=close.index, name=f"rsi_{period}")


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 12,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    SuperTrend(period, multiplier) on HL2 ± multiplier * ATR.

    Final bands only tighten while price stays on the same side; direction
    flips when close crosses the active band. Returns (st_line, st_dir) with
    st_dir +1 bullish / -1 bearish, NaN during warm-up.
    """
    st_line, st_dir = supertrend_kernel(
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        close.to_numpy(dtype=float),
        period,
        float(multiplier),
    )
    return (
        pd.Series(st_line, index=close.index, name=f"st_line_{period}_{multiplier}"),
        pd.Series(st_dir, index=close.index, name=f"st_dir_{period}_{multiplier}"),
    )


def st_flip(st_dir: pd.Series) -> pd.Series:
    """True when SuperTrend direction differs from the previous bar."""
    return (st_dir != st_dir.shift(1)) & st_dir.notna()


def st_stable(st_dir: pd.Series, bars: int = 3) -> pd.Series:
    """True when the last `bars` SuperTrend directions are all present and equal."""
    stable = st_stable_kernel(st_dir.to_numpy(dtype=float), bars)
    return pd.Series(stable, index=st_dir.index, name=f"st_stable_{bars}")


def st_step_count(st_line: pd.Series, st_dir: pd.Series) -> pd.Series:
    """
    Count SuperTrend line steps in the trend direction since the last flip.

    Bullish episode counts bars where st_line rose; bearish counts bars where
    it fell. Resets to 0 on the flip bar.
    """
    counts = st_step_count_kernel(st_line.to_numpy(dtype=float), st_dir.to_numpy(dtype=float))
    return pd.Series(counts, index=st_dir.index, name="st_step_count")


@njit
def ema_kernel(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    if n < period:
        return result

    seed = period - 1
    total = 0.0
    for i in range(period):
        total += prices[i]
    result[seed] = total / period

    multiplier = 2.0 / (period + 1.0)
    for i in range(seed + 1, n):
        result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


@njit
def atr_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[:] = np.nan
    if n == 0:
        return tr

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        v1 = high[i] - low[i]
        v2 = abs(high[i] - close[i - 1])
        v3 = abs(low[i] - close[i - 1])
        tr[i] = max(v1, v2, v3)

    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    if n < period:
        return result

    total = 0.0
    for i in range(period):
        total += tr[i]
    result[period - 1] = total / period

    alpha = 1.0 / period
    for i in range(period, n):
        result[i] = result[i - 1] + alpha * (tr[i] - result[i - 1])
    return result


@njit
def rsi_kernel(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    if n < period + 1:
        return result

    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        if delta > 0.0:
            avg_gain += delta
        else:
            avg_loss -= delta
    avg_gain /= period
    avg_loss /= period

    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    alpha = 1.0 / period
    for i in range(period + 1, n):
        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        avg_gain = avg_gain + alpha * (gain - avg_gain)
        avg_loss = avg_loss + alpha * (loss - avg_loss)
        if avg_loss == 0.0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


@njit
def supertrend_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    st_line = np.empty(n, dtype=np.float64)
    st_dir = np.empty(n, dtype=np.float64)
    final_ub = np.empty(n, dtype=np.float64)
    final_lb = np.empty(n, dtype=np.float64)
    st_line[:] = np.nan
    st_dir[:] = np.nan
    final_ub[:] = np.nan
    final_lb[:] = np.nan

    atr_vals = atr_kernel(high, low, close, period)
    start = period - 1
    if start >= n:
        return st_line, st_dir

    hl2 = (high[start] + low[start]) / 2.0
    final_ub[start] = hl2 + multiplier * atr_vals[start]
    final_lb[start] = hl2 - multiplier * atr_vals[start]

    if close[start] <= final_ub[start]:
        st_dir[start] = 1.0
        st_line[start] = final_lb[start]
    else:
        st_dir[start] = -1.0
        st_line[start] = final_ub[start]

    for i in range(start + 1, n):
        hl2 = (high[i] + low[i]) / 2.0
        basic_ub = hl2 + multiplier * atr_vals[i]
        basic_lb = hl2 - multiplier * atr_vals[i]

        if basic_ub < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb
        else:
            final_lb[i] = final_lb[i - 1]

        if st_dir[i - 1] == 1.0:
            if close[i] < final_lb[i]:
                st_dir[i] = -1.0
                st_line[i] = final_ub[i]
            else:
                st_dir[i] = 1.0
                st_line[i] = final_lb[i]
        else:
            if close[i] > final_ub[i]:
                st_dir[i] = 1.0
                st_line[i] = final_lb[i]
            else:
                st_dir[i] = -1.0
                st_line[i] = final_ub[i]
    return st_line, st_dir


@njit
def st_stable_kernel(st_dir: np.ndarray, bars: int) -> np.ndarray:
    n = len(st_dir)
    stable = np.zeros(n, dtype=np.bool_)
    for i in range(bars - 1, n):
        last = st_dir[i]
        if np.isnan(last):
            continue
        ok = True
        for j in range(i - bars + 1, i):
            if np.isnan(st_dir[j]) or st_dir[j] != last:
                ok = False
                break
        stable[i] = ok
    return stable


@njit
def st_step_count_kernel(st_line: np.ndarray, st_dir: np.ndarray) -> np.ndarray:
    n = len(st_dir)
    counts = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        d = st_dir[i]
        if np.isnan(d) or np.isnan(st_line[i]):
            continue
        if st_dir[i - 1] != d:
            counts[i] = 0
            continue
        prev = st_line[i - 1]
        curr = st_line[i]
        step = 0
        if not (np.isnan(prev) or np.isnan(curr)):
            if d == 1.0 and curr > prev:
                step = 1
            elif d == -1.0 and curr < prev:
                step = 1
        counts[i] = counts[i - 1] + step
    return counts


class TAContext:
    """Closed-bar EMA and optional ATR-band context."""

    STATE_VERSION = 2

    def __init__(
        self,
        spans: Iterable[int] = DEFAULT_EMA_SPANS,
        atr_period: int | None = None,
        band: dict[str, Any] | None = None,
    ) -> None:
        raw_spans = tuple(spans)
        if any(isinstance(span, bool) or not isinstance(span, int) for span in raw_spans):
            raise ValueError("EMA spans must be unique positive integers")
        normalized = tuple(raw_spans)
        if not normalized or any(span <= 0 for span in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("EMA spans must be unique positive integers")
        self.spans = normalized
        if atr_period is not None and (
            isinstance(atr_period, bool) or not isinstance(atr_period, int) or atr_period <= 0
        ):
            raise ValueError("ATR period must be a positive integer")
        self.atr_period = atr_period
        self.band = self._normalize_band(band)
        if self.band is not None and self.atr_period is None:
            raise ValueError("TAContext atr_period is required when a band is configured")

        self.sample_count = 0
        self.ema: dict[int, float | None] = {span: None for span in normalized}
        self.seed_sum: dict[int, float] = {span: 0.0 for span in normalized}
        self.prev_close: float | None = None
        self.tr_seed_sum = 0.0
        self.tr_count = 0
        self.atr: float | None = None
        self.cursor: str | None = None

    def step(self, row: pd.Series) -> dict[str, Any]:
        known_at = getattr(row, "name", None)
        if known_at is None:
            raise ValueError("TA rows must carry their closed-bar timestamp as the index")
        ts = self._naive_utc(known_at)
        if self.cursor is not None and ts <= pd.Timestamp(self.cursor):
            raise ValueError(f"TA bars must increase: {ts} <= {self.cursor}")

        close = float(row["close"])
        self.sample_count += 1
        for span in self.spans:
            previous = self.ema[span]
            if previous is None:
                self.seed_sum[span] += close
                if self.sample_count >= span:
                    self.ema[span] = self.seed_sum[span] / span
            else:
                alpha = 2.0 / (span + 1.0)
                self.ema[span] = alpha * close + (1.0 - alpha) * previous

        if self.atr_period is not None:
            self._step_atr(row, close)

        self.cursor = ts.isoformat()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "ema": {str(span): self.ema[span] for span in self.spans},
            "sample_count": self.sample_count,
        }
        if self.atr_period is not None:
            snapshot["atr"] = {str(self.atr_period): self.atr}
            if self.band is not None:
                ema = self.ema[self.band["ema"]]
                ready = ema is not None and self.atr is not None
                snapshot["band"] = {
                    "ema": ema,
                    "atr": self.atr,
                    "mult": self.band["mult"],
                    "upper": ema + self.band["mult"] * self.atr if ready else None,
                    "lower": ema - self.band["mult"] * self.atr if ready else None,
                }
        return snapshot

    def export_state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "spans": list(self.spans),
            "sample_count": self.sample_count,
            "ema": {str(span): self.ema[span] for span in self.spans},
            "seed_sum": {str(span): self.seed_sum[span] for span in self.spans},
            "cursor": self.cursor,
            "prev_close": self.prev_close,
            "tr_seed_sum": self.tr_seed_sum,
            "tr_count": self.tr_count,
            "atr": self.atr,
            "atr_period": self.atr_period,
            "band": dict(self.band) if self.band is not None else None,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        state_version = int(state.get("state_version", 1))
        if state_version not in {1, self.STATE_VERSION}:
            raise ValueError("Unsupported TAContext state version")
        saved_spans = tuple(int(span) for span in state.get("spans", self.spans))
        if saved_spans != self.spans:
            raise ValueError(f"TAContext EMA spans mismatch: saved {saved_spans}, requested {self.spans}")
        if state_version == 1:
            if self.atr_period is not None:
                raise ValueError("TAContext legacy state has no ATR state")
        else:
            saved_period = state.get("atr_period")
            if saved_period != self.atr_period:
                raise ValueError(
                    f"TAContext ATR period mismatch: saved {saved_period}, requested {self.atr_period}"
                )
            saved_band = self._normalize_band(state.get("band"))
            if saved_band != self.band:
                raise ValueError(f"TAContext band mismatch: saved {saved_band}, requested {self.band}")

        self.sample_count = int(state.get("sample_count", 0))
        ema = state.get("ema") or {}
        seed_sum = state.get("seed_sum") or {}
        self.ema = {
            span: float(ema[str(span)]) if ema.get(str(span)) is not None else None
            for span in self.spans
        }
        self.seed_sum = {span: float(seed_sum.get(str(span), 0.0)) for span in self.spans}
        self.cursor = state.get("cursor")
        if state_version == 1:
            self.prev_close = None
            self.tr_seed_sum = 0.0
            self.tr_count = 0
            self.atr = None
        else:
            self.prev_close = (
                float(state["prev_close"]) if state.get("prev_close") is not None else None
            )
            self.tr_seed_sum = float(state.get("tr_seed_sum", 0.0))
            self.tr_count = int(state.get("tr_count", 0))
            self.atr = float(state["atr"]) if state.get("atr") is not None else None

    def _step_atr(self, row: pd.Series, close: float) -> None:
        high = float(row["high"])
        low = float(row["low"])
        if not all(math.isfinite(value) for value in (high, low, close)):
            raise ValueError("TA ATR inputs must be finite")
        true_range = high - low
        if self.prev_close is not None:
            true_range = max(
                true_range,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )
        self.prev_close = close
        self.tr_count += 1
        if self.atr is None:
            self.tr_seed_sum += true_range
            if self.tr_count >= self.atr_period:
                self.atr = self.tr_seed_sum / self.atr_period
            return
        self.atr = (true_range + (self.atr_period - 1) * self.atr) / self.atr_period

    def _normalize_band(self, band: dict[str, Any] | None) -> dict[str, Any] | None:
        if band is None:
            return None
        if not isinstance(band, dict):
            raise ValueError("TAContext band must be a mapping")
        unknown = sorted(set(band) - {"ema", "mult"})
        if unknown:
            raise ValueError(f"Unknown TAContext band key '{unknown[0]}'")
        ema = band.get("ema")
        mult = band.get("mult")
        if isinstance(ema, bool) or not isinstance(ema, int) or ema not in self.spans:
            raise ValueError("TAContext band EMA must be one of the configured spans")
        if isinstance(mult, bool) or not isinstance(mult, (int, float)):
            raise ValueError("TAContext band multiplier must be a positive finite number")
        mult = float(mult)
        if not math.isfinite(mult) or mult <= 0:
            raise ValueError("TAContext band multiplier must be a positive finite number")
        return {"ema": ema, "mult": mult}

    @staticmethod
    def _naive_utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp
