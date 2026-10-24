from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from .pivot_event import PivotEvent, PivotEventEngine
from .structure_breaks import TIERS, StructureBreaks


DEFAULT_ENGINE_CONFIG: dict[str, Any] = {
    "mode": "conservative",
    "str_bars": 2,
    "compute": {"str": True, "itr": True, "ltr": True, "xltr": False},
    "timeframe": "1D",
}
GENESIS_SEED = "low"
WARMUP_START = pd.Timestamp("2020-01-01")


class StructureReducer:
    """Single-pass pivot and internal-structure reducer for one bar series."""

    STATE_VERSION = 2

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        genesis_seed: str = GENESIS_SEED,
        bar_duration: str | pd.Timedelta = "1D",
        warmup_start: Any | None = WARMUP_START,
    ) -> None:
        if genesis_seed != "low":
            raise ValueError("The structure genesis seed is fixed to low-first")
        duration = pd.Timedelta(bar_duration)
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError("bar_duration must be a positive duration")
        self.engine_config = deepcopy(config or DEFAULT_ENGINE_CONFIG)
        self.genesis_seed = genesis_seed
        self.bar_duration = duration
        self.warmup_start = self._naive_utc(warmup_start) if warmup_start is not None else None
        self.pivot = PivotEventEngine(self.engine_config)
        self.breaks = StructureBreaks()
        self.cursor: dict[str, Any] = {"bar_index": -1, "last_known_at": None}
        self.first_ready_at: dict[str, str | None] = {tier: None for tier in TIERS}

        # The pivot engine defaults to a high first. Seed it low-first before bar 1.
        initial_pivot_state = self.pivot.export_state()
        initial_pivot_state["last_confirmed_side"] = 1
        self.pivot.load_state(initial_pivot_state)

    def step(self, row: pd.Series) -> dict[str, Any]:
        ts = self._naive_utc(row.name)
        previous_ts = self.cursor["last_known_at"]
        if previous_ts is not None and ts <= self._naive_utc(previous_ts):
            raise ValueError(f"Bars must be strictly increasing: {ts} <= {previous_ts}")

        bar = row.copy(deep=False)
        bar.name = ts
        self.cursor["bar_index"] += 1
        events: list[PivotEvent] = self.pivot.on_bar(bar)
        for event in events:
            self.breaks.register(event)
            self._mark_first_ready(event.tier, ts)

        break_events, wick_events = self.breaks.check(
            float(bar["high"]), float(bar["low"]), float(bar["close"]), ts
        )
        closed_levels = self.breaks.drain_closed()
        self.cursor["last_known_at"] = ts.isoformat()

        source_open = ts - self.bar_duration
        snapshot = self.breaks.snapshot()
        state = {
            "known_at": ts.isoformat(),
            "warmup": self.warmup_start is not None and source_open < self.warmup_start,
            "bar_index": self.cursor["bar_index"],
            "tiers": snapshot["tiers"],
            "lanes": snapshot["lanes"],
        }
        return {
            "events": events,
            "breaks": break_events,
            "wicks": wick_events,
            "closed_levels": closed_levels,
            "state": state,
        }

    @staticmethod
    def _naive_utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp

    def active_levels(self) -> list[dict[str, Any]]:
        return self.breaks.active_levels()

    def _mark_first_ready(self, tier: str, ts: pd.Timestamp) -> None:
        if self.first_ready_at[tier] is not None:
            return
        levels = self.breaks.snapshot()["tiers"][tier]
        if levels["high"] is not None and levels["low"] is not None:
            self.first_ready_at[tier] = ts.isoformat()

    def export_state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "pivot": self.pivot.export_state(),
            "breaks": self.breaks.export_state(),
            "cursor": dict(self.cursor),
            "first_ready_at": dict(self.first_ready_at),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported StructureReducer state version")
        self.pivot.load_state(state["pivot"])
        self.breaks.load_state(state["breaks"])
        cursor = state.get("cursor") or {}
        self.cursor = {
            "bar_index": int(cursor.get("bar_index", -1)),
            "last_known_at": cursor.get("last_known_at"),
        }
        ready = state.get("first_ready_at") or {}
        self.first_ready_at = {tier: ready.get(tier) for tier in TIERS}
