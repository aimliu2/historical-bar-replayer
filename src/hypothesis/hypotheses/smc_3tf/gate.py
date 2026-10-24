from __future__ import annotations

from copy import deepcopy
from typing import Any


def target_side(daily: dict[str, Any] | None) -> str | None:
    if not isinstance(daily, dict):
        return None
    status = str(daily.get("of_status") or "").lower()
    direction = str(daily.get("of_direction") or "").lower()
    if status == "active":
        return "top" if direction == "bear" else "bottom" if direction == "bull" else None
    if status == "provisional":
        # In provisional state, of_direction is the c-1 active direction.
        return "bottom" if direction == "bear" else "top" if direction == "bull" else None
    return None


class ArmingSide:
    """Hold the old daily target until the 1H band is first re-entered on 5m."""

    STATE_VERSION = 1

    def __init__(self) -> None:
        self.current_side: str | None = None
        self.desired_side: str | None = None
        self.pending_side: str | None = None
        self.pending_since: str | None = None
        self.last_daily_known_at: str | None = None
        self.last_daily_signature: tuple[str | None, str | None] | None = None

    def step(self, ctx: dict[str, Any]) -> dict[str, Any]:
        daily = ctx.get("1d") or {}
        time = str(ctx.get("time") or "")
        events: list[dict[str, Any]] = []
        known_at = daily.get("known_at")
        signature = (daily.get("of_status"), daily.get("of_direction"))
        if known_at is not None and str(known_at) != self.last_daily_known_at:
            wanted = target_side(daily)
            previous_desired = self.desired_side
            self.last_daily_known_at = str(known_at)
            self.last_daily_signature = signature
            self.desired_side = wanted
            if self.current_side is None and wanted is not None:
                self.current_side = wanted
                self.pending_side = None
                self.pending_since = None
                events.append({"kind": "initialized", "side": wanted, "known_at": str(known_at)})
            elif wanted is None or wanted == self.current_side:
                self.pending_side = None
                self.pending_since = None
                if previous_desired != wanted:
                    events.append(
                        {"kind": "pending_cancelled", "side": self.current_side, "known_at": str(known_at)}
                    )
            else:
                self.pending_side = wanted
                self.pending_since = str(known_at)
                events.append(
                    {
                        "kind": "target_changed",
                        "from": self.current_side,
                        "to": wanted,
                        "known_at": str(known_at),
                    }
                )

        switched = None
        if self.pending_side is not None:
            band = ((((ctx.get("1h") or {}).get("ta") or {}).get("band")) or {})
            upper, lower = band.get("upper"), band.get("lower")
            close = (ctx.get("bar_5m") or {}).get("close")
            if upper is not None and lower is not None and close is not None:
                if float(lower) <= float(close) <= float(upper):
                    old = self.current_side
                    self.current_side = self.pending_side
                    switched = {
                        "kind": "switched",
                        "from": old,
                        "to": self.current_side,
                        "time": time,
                        "daily_known_at": self.last_daily_known_at,
                    }
                    events.append(switched)
                    self.pending_side = None
                    self.pending_since = None

        return {
            "arming_side": self.current_side,
            "desired_side": self.desired_side,
            "pending_side": self.pending_side,
            "pending_since": self.pending_since,
            "switched": switched is not None,
            "events": events,
        }

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "current_side": self.current_side,
            "desired_side": self.desired_side,
            "pending_side": self.pending_side,
            "pending_since": self.pending_since,
            "last_daily_known_at": self.last_daily_known_at,
            "last_daily_signature": list(self.last_daily_signature) if self.last_daily_signature else None,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported ArmingSide state version")
        for key in ("current_side", "desired_side", "pending_side"):
            value = state.get(key)
            if value not in {None, "top", "bottom"}:
                raise ValueError(f"Unsupported ArmingSide {key}: {value}")
            setattr(self, key, value)
        self.pending_since = state.get("pending_since")
        self.last_daily_known_at = state.get("last_daily_known_at")
        signature = state.get("last_daily_signature")
        self.last_daily_signature = tuple(signature) if signature is not None else None

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ArmingSide:
        gate = cls()
        gate.load_state(deepcopy(state))
        return gate
