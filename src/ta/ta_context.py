from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd


DEFAULT_EMA_SPANS = (50, 100, 200)


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
