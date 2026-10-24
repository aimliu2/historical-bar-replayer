from __future__ import annotations

from typing import Any, Literal

import pandas as pd


Side = Literal["high", "low"]


class STRDefense:
    """Track an orderflow-selected STR pivot or one explicitly fixed side."""

    STATE_VERSION = 2

    def __init__(self, *, fixed_side: Side | None = None) -> None:
        if fixed_side not in {None, "high", "low"}:
            raise ValueError("STR defense fixed_side must be 'high', 'low' or None")
        self.fixed_side = fixed_side
        self.side: Side | None = None
        self.pivot_ts: str | None = None
        self.level_price: float | None = None
        self.alive = False
        self.cursor: str | None = None
        self.last_event: str | None = None
        self.last_event_level: float | None = None
        self.last_event_at: str | None = None
        self.line_event: dict[str, Any] | None = None

    def step(
        self,
        result: dict[str, Any],
        orderflow: dict[str, Any],
        *,
        close: float,
        known_at: Any,
    ) -> dict[str, Any]:
        ts = self._naive_utc(known_at)
        if self.cursor is not None and ts <= pd.Timestamp(self.cursor):
            raise ValueError(f"STR defense bars must increase: {ts} <= {self.cursor}")

        previous_pivot = self.pivot_ts
        previous_alive = self.alive
        previous_price = self.level_price if previous_alive else None
        had_previous = self.cursor is not None
        self.line_event = None
        direction = orderflow.get("direction")
        side: Side | None = self.fixed_side or (
            "low" if direction == "bull" else "high" if direction == "bear" else None
        )
        tiers = (result.get("state") or {}).get("tiers") or {}
        slot = (tiers.get("str") or {}).get(side) if side is not None else None
        if side is None or slot is None:
            self.side = side
            self.pivot_ts = None
            self.level_price = None
            self.alive = False
        else:
            pivot_ts = pd.Timestamp(slot.get("pivot_ts", slot.get("ts")))
            if pivot_ts.tzinfo is not None:
                pivot_ts = pivot_ts.tz_convert("UTC").tz_localize(None)
            identity_changed = side != self.side or pivot_ts.isoformat() != self.pivot_ts
            if identity_changed:
                self.side = side
                self.pivot_ts = pivot_ts.isoformat()
                self.level_price = float(slot["price"])
                self.alive = True
            if self.alive and self.level_price is not None:
                closed_through = close < self.level_price if side == "low" else close > self.level_price
                if closed_through:
                    self.alive = False

        # Only fixed-side daily lines have unambiguous event identities.
        if self.fixed_side is not None and had_previous and slot is not None:
            moved = self.pivot_ts != previous_pivot
            event = None
            level = None
            if previous_alive and not self.alive and not moved:
                event, level = "broken", previous_price
            elif moved and self.alive:
                event = (
                    "update up" if self.level_price > previous_price else "update down"
                ) if previous_alive else "new"
                level = self.level_price
            elif moved and not self.alive:
                event, level = "new + broken", float(close)
            if event is not None:
                self.last_event = event
                self.last_event_level = level
                self.last_event_at = ts.isoformat()
                self.line_event = {
                    "side": self.fixed_side, "event": event, "level": level,
                    "old_level": previous_price, "pivot_ts": self.pivot_ts,
                    "event_ts": ts.isoformat(),
                }
        self.cursor = ts.isoformat()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "side": self.side,
            "price": self.level_price if self.alive else None,
            "pivot_ts": self.pivot_ts,
            "alive": self.alive,
        }
        if self.fixed_side is not None:
            snapshot.update(last_event=self.last_event,
                            last_event_level=self.last_event_level,
                            last_event_at=self.last_event_at)
        return snapshot

    def export_state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "fixed_side": self.fixed_side,
            "side": self.side,
            "pivot_ts": self.pivot_ts,
            "level_price": self.level_price,
            "alive": self.alive,
            "cursor": self.cursor,
            "last_event": self.last_event,
            "last_event_level": self.last_event_level,
            "last_event_at": self.last_event_at,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported STRDefense state version")
        if state.get("fixed_side") != self.fixed_side:
            raise ValueError("STRDefense fixed-side mode does not match the saved state")
        side = state.get("side")
        if side not in {None, "high", "low"}:
            raise ValueError(f"Unsupported STR defense side: {side}")
        level = state.get("level_price")
        self.side = side
        self.pivot_ts = state.get("pivot_ts")
        self.level_price = float(level) if level is not None else None
        self.alive = bool(state.get("alive", False))
        self.cursor = state.get("cursor")
        self.last_event = state.get("last_event")
        level = state.get("last_event_level")
        self.last_event_level = float(level) if level is not None else None
        self.last_event_at = state.get("last_event_at")
        # Restored historical events are state, never newly delivered events.
        self.line_event = None

    @staticmethod
    def _naive_utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp
