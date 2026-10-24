from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


PIP_SIZE_DEFAULT = 0.0001


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else _utc(value).isoformat()


class WaveTracker:
    """Incremental `detect_waves` port over the two fixed-side 1D defense lines."""

    STATE_VERSION = 1

    def __init__(self, *, pip_size: float = PIP_SIZE_DEFAULT) -> None:
        if float(pip_size) <= 0:
            raise ValueError("WaveTracker pip_size must be positive")
        self.pip_size = float(pip_size)
        self.lines = {
            side: {"wave": None, "last_established": None, "previous": None}
            for side in ("low", "high")
        }
        self.pivots: dict[str, dict[str, dict[str, Any]]] = {"low": {}, "high": {}}
        self._seen_pivots: set[str] = set()
        self.last_known_at: str | None = None

    @staticmethod
    def _pivot_key(event: dict[str, Any]) -> str:
        event_id = event.get("event_id")
        if event_id is not None:
            return str(event_id)
        return "|".join(
            str(event.get(key, "")) for key in ("side", "pivot_ts", "event_ts", "relation", "price")
        )

    def _collect_pivots(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        collected = []
        for raw in events:
            if not isinstance(raw, dict):
                continue
            side = raw.get("side")
            if side not in {"low", "high"} or raw.get("pivot_ts") is None:
                continue
            key = self._pivot_key(raw)
            if key in self._seen_pivots:
                continue
            self._seen_pivots.add(key)
            event = {
                "event_id": raw.get("event_id"),
                "side": side,
                "relation": raw.get("relation"),
                "price": float(raw["price"]),
                "pivot_ts": _iso(raw["pivot_ts"]),
                "event_ts": _iso(raw.get("event_ts")),
            }
            self.pivots[side][event["pivot_ts"]] = event
            collected.append(event)
        return collected

    def step(self, ctx: dict[str, Any]) -> dict[str, Any] | None:
        daily = ctx.get("1d")
        if not isinstance(daily, dict) or not daily.get("known_at"):
            return None
        new_pivots = self._collect_pivots(list(daily.get("str_pivots") or []))
        known_at = _iso(daily["known_at"])
        if known_at == self.last_known_at:
            for event in new_pivots:
                current = self.lines[event["side"]]["wave"]
                if current is not None and current.get("start_ts") == event["pivot_ts"]:
                    current["start_confirmed"] = event["event_ts"]
            return {"known_at": known_at, "changed": bool(new_pivots), "current": self.current(), "closed": [], "counters": [], "pivot_events": new_pivots}

        closed: list[dict[str, Any]] = []
        counters: list[dict[str, Any]] = []
        fixed_lines = daily.get("str_defense_by_side") or {}
        for fixed_side in ("low", "high"):
            row = fixed_lines.get(fixed_side) or {}
            state = self.lines[fixed_side]
            side = row.get("side") or fixed_side
            pivot_ts = _iso(row.get("pivot_ts"))
            identity = (side, pivot_ts)
            alive = bool(row.get("alive")) and row.get("price") is not None and pivot_ts is not None
            wave = state["wave"]

            if wave is not None and side != wave["side"]:
                state["wave"] = None
                wave = None
            elif wave is not None and identity != state["previous"] and pivot_ts is not None:
                next_price = float(row["price"])
                next_ts = pivot_ts
                extreme = self._opposite_extreme(
                    wave["side"], wave["start_ts"], next_ts, known_at
                )
                lift = (
                    next_price > float(wave["start_price"])
                    if wave["side"] == "low"
                    else next_price < float(wave["start_price"])
                )
                if wave["discontinued_at"] is None and not lift:
                    wave["discontinued_at"] = known_at
                    wave["discontinued_by"] = "wrong-way step"
                outcome = "established" if wave["discontinued_at"] is None else "discontinued"
                closed_wave = {
                    **deepcopy(wave),
                    "fixed_line": fixed_side,
                    "outcome": outcome,
                    "extreme_price": extreme.get("price") if extreme is not None else None,
                    "extreme_ts": extreme.get("pivot_ts") if extreme is not None else None,
                    "extreme_relation": extreme.get("relation") if extreme is not None else None,
                    "extreme_confirmed": extreme.get("event_ts") if extreme is not None else None,
                    "next_price": next_price,
                    "next_ts": next_ts,
                    "next_confirmed": self._confirmation(side, next_ts),
                    "closed_at": known_at,
                }
                closed.append(closed_wave)
                if extreme is not None and outcome == "established":
                    state["last_established"] = deepcopy(closed_wave)
                elif extreme is not None:
                    size = abs(float(extreme["price"]) - next_price)
                    prior = state["last_established"]
                    reference = (
                        abs(float(prior["extreme_price"]) - float(prior["start_price"]))
                        if prior is not None and prior.get("extreme_price") is not None
                        else None
                    )
                    counters.append(
                        {
                            "fixed_line": fixed_side,
                            "from_ts": extreme["pivot_ts"],
                            "from_price": float(extreme["price"]),
                            "from_confirmed": extreme["event_ts"],
                            "to_ts": next_ts,
                            "to_price": next_price,
                            "to_confirmed": closed_wave["next_confirmed"],
                            "pips": size / self.pip_size,
                            "last_established": (
                                None
                                if prior is None
                                else f"{float(prior['start_price']):.5f} -> {float(prior['extreme_price']):.5f}"
                            ),
                            "pct_of_last_established": size / reference * 100 if reference else None,
                            "known_at": known_at,
                        }
                    )
                state["wave"] = None
                wave = None
            elif wave is not None and not alive and wave["discontinued_at"] is None:
                wave["discontinued_at"] = known_at
                wave["discontinued_by"] = "close through"

            if state["wave"] is None and alive:
                confirmation = self._confirmation(side, pivot_ts)
                state["wave"] = {
                    "fixed_line": fixed_side,
                    "side": side,
                    "start_price": float(row["price"]),
                    "start_ts": pivot_ts,
                    "start_confirmed": confirmation,
                    "discontinued_at": None,
                    "discontinued_by": None,
                }
            state["previous"] = identity

        self.last_known_at = known_at
        return {
            "known_at": known_at,
            "changed": True,
            "current": self.current(),
            "closed": closed,
            "counters": counters,
            "pivot_events": new_pivots,
        }

    def _confirmation(self, side: str, pivot_ts: str | None) -> str | None:
        if pivot_ts is None:
            return None
        event = self.pivots.get(side, {}).get(pivot_ts)
        return event.get("event_ts") if event is not None else None

    def _opposite_extreme(
        self, side: str, start_ts: str, end_ts: str, known_at: str
    ) -> dict[str, Any] | None:
        opposite = "high" if side == "low" else "low"
        start, end, known = _utc(start_ts), _utc(end_ts), _utc(known_at)
        candidates = [
            event
            for event in self.pivots[opposite].values()
            if start < _utc(event["pivot_ts"]) < end
            and event.get("event_ts") is not None
            and _utc(event["event_ts"]) <= known
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item["price"]) if side == "low" else min(candidates, key=lambda item: item["price"])

    def current(self) -> dict[str, Any]:
        return {side: deepcopy(state["wave"]) for side, state in self.lines.items()}

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "pip_size": self.pip_size,
            "lines": deepcopy(self.lines),
            "pivots": deepcopy(self.pivots),
            "seen_pivots": sorted(self._seen_pivots),
            "last_known_at": self.last_known_at,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported WaveTracker state version")
        pip_size = float(state.get("pip_size", self.pip_size))
        if pip_size <= 0:
            raise ValueError("WaveTracker saved pip_size must be positive")
        lines = deepcopy(state.get("lines") or {})
        pivots = deepcopy(state.get("pivots") or {})
        if set(lines) != {"low", "high"} or set(pivots) != {"low", "high"}:
            raise ValueError("WaveTracker state must include low and high fixed lines")
        self.pip_size = pip_size
        for line in lines.values():
            if line.get("previous") is not None:
                line["previous"] = tuple(line["previous"])
        self.lines = lines
        self.pivots = pivots
        self._seen_pivots = set(map(str, state.get("seen_pivots") or []))
        self.last_known_at = state.get("last_known_at")

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> WaveTracker:
        tracker = cls(pip_size=float(state.get("pip_size", PIP_SIZE_DEFAULT)))
        tracker.load_state(state)
        return tracker
