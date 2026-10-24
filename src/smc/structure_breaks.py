from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from .pivot_event import PivotEvent


Tier = Literal["str", "itr", "ltr"]
Side = Literal["high", "low"]
LANES: tuple[str, ...] = ("itr", "ltr")
TIERS: tuple[str, ...] = ("str", "itr", "ltr")


@dataclass
class StructureLevel:
    tier: Tier
    side: Side
    price: float
    pivot_ts: pd.Timestamp
    event_ts: pd.Timestamp
    relation: str
    spent: bool = False
    wick_count: int = 0
    first_wicked_at: pd.Timestamp | None = None
    last_wicked_at: pd.Timestamp | None = None
    broken_at: pd.Timestamp | None = None
    break_code: str | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "side": self.side,
            "price": self.price,
            "pivot_ts": self.pivot_ts.isoformat(),
            "event_ts": self.event_ts.isoformat(),
            "relation": self.relation,
            "spent": self.spent,
            "wick_count": self.wick_count,
            "first_wicked_at": (
                self.first_wicked_at.isoformat() if self.first_wicked_at is not None else None
            ),
            "last_wicked_at": (
                self.last_wicked_at.isoformat() if self.last_wicked_at is not None else None
            ),
            "broken_at": self.broken_at.isoformat() if self.broken_at is not None else None,
            "break_code": self.break_code,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> StructureLevel:
        return cls(
            tier=state["tier"],
            side=state["side"],
            price=float(state["price"]),
            pivot_ts=pd.Timestamp(state["pivot_ts"]),
            event_ts=pd.Timestamp(state["event_ts"]),
            relation=str(state["relation"]),
            spent=bool(state.get("spent", False)),
            wick_count=int(state.get("wick_count", 0)),
            first_wicked_at=(
                pd.Timestamp(state["first_wicked_at"])
                if state.get("first_wicked_at") is not None
                else None
            ),
            last_wicked_at=(
                pd.Timestamp(state["last_wicked_at"])
                if state.get("last_wicked_at") is not None
                else None
            ),
            broken_at=(
                pd.Timestamp(state["broken_at"]) if state.get("broken_at") is not None else None
            ),
            break_code=state.get("break_code"),
        )


class StructureBreaks:
    """Tier relation memory and ITR/LTR close-break and wick observers."""

    STATE_VERSION = 2

    def __init__(self) -> None:
        self._levels: dict[str, dict[str, StructureLevel | None]] = {
            tier: {"high": None, "low": None} for tier in TIERS
        }
        self._last_sides: dict[str, str | None] = {tier: None for tier in TIERS}
        self._last_breaks: dict[str, dict[str, Any] | None] = {
            lane: None for lane in LANES
        }
        self._last_wicks: dict[str, dict[str, Any] | None] = {
            lane: None for lane in LANES
        }
        self._closed: list[dict[str, Any]] = []

    def register(self, event: PivotEvent) -> StructureLevel:
        tier = event.tier
        side: Side = "high" if event.pivot_side == 1 else "low"
        if tier not in self._levels:
            raise ValueError(f"Unsupported pivot tier for structure relation: {tier}")

        previous = self._levels[tier][side]
        if previous is None:
            relation = "seed"
        elif side == "high":
            relation = "HH" if event.pivot_price > previous.price else "LH"
        else:
            relation = "HL" if event.pivot_price > previous.price else "LL"

        event.relation = relation
        level = StructureLevel(
            tier=tier,  # type: ignore[arg-type]
            side=side,
            price=float(event.pivot_price),
            pivot_ts=event.pivot_timestamp,
            event_ts=event.event_timestamp,
            relation=relation,
        )
        if tier in LANES and previous is not None and not previous.spent:
            self._closed.append(
                self._lifecycle_row(
                    tier,
                    previous,
                    fate="superseded",
                    fate_ts=event.event_timestamp,
                    break_code=None,
                    superseded_by_ts=event.pivot_timestamp,
                    superseded_by_price=float(event.pivot_price),
                )
            )
        self._levels[tier][side] = level
        self._last_sides[tier] = side
        return level

    def check(
        self, high: float, low: float, close: float, ts: pd.Timestamp
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Emit up to one close break and every qualifying wick per lane."""
        emitted: list[dict[str, Any]] = []
        wicks: list[dict[str, Any]] = []
        for lane in LANES:
            high_level = self._levels[lane]["high"]
            low_level = self._levels[lane]["low"]
            lane_break_emitted = False
            for level in (high_level, low_level):
                if level is None or level.spent:
                    continue
                broken = close > level.price if level.side == "high" else close < level.price
                if broken:
                    if not lane_break_emitted:
                        code = self._break_code(lane, level)
                        event = {
                            "code": code,
                            "lane": lane,
                            "event_ts": ts.isoformat(),
                            "break_direction": "up" if level.side == "high" else "down",
                            "level_price": level.price,
                            "level_ts": level.pivot_ts.isoformat(),
                            "level_relation": level.relation,
                            "level_side": level.side,
                        }
                        level.spent = True
                        level.broken_at = ts
                        level.break_code = code
                        self._last_breaks[lane] = event
                        self._closed.append(
                            self._lifecycle_row(
                                lane,
                                level,
                                fate="broken",
                                fate_ts=ts,
                                break_code=code,
                            )
                        )
                        emitted.append(dict(event))
                        lane_break_emitted = True
                    # A close through takes precedence over a wick on this level.
                    continue

                wicked = (
                    high > level.price and close <= level.price
                    if level.side == "high"
                    else low < level.price and close >= level.price
                )
                if not wicked:
                    continue

                level.wick_count += 1
                if level.first_wicked_at is None:
                    level.first_wicked_at = ts
                level.last_wicked_at = ts
                wick = {
                    "code": "iwick" if lane == "itr" else "lwick",
                    "lane": lane,
                    "event_ts": ts.isoformat(),
                    "direction": "up" if level.side == "high" else "down",
                    "level_price": level.price,
                    "level_ts": level.pivot_ts.isoformat(),
                    "level_relation": level.relation,
                    "level_side": level.side,
                    "wick_extreme": float(high if level.side == "high" else low),
                    "close": float(close),
                    "wick_seq": level.wick_count,
                }
                self._last_wicks[lane] = wick
                wicks.append(dict(wick))
        return emitted, wicks

    @staticmethod
    def _lifecycle_row(
        lane: str,
        level: StructureLevel,
        *,
        fate: str,
        fate_ts: pd.Timestamp | None,
        break_code: str | None,
        superseded_by_ts: pd.Timestamp | None = None,
        superseded_by_price: float | None = None,
    ) -> dict[str, Any]:
        return {
            "lane": lane,
            "side": level.side,
            "pivot_ts": level.pivot_ts.isoformat(),
            "event_ts": level.event_ts.isoformat(),
            "price": level.price,
            "relation": level.relation,
            "fate": fate,
            "fate_ts": fate_ts.isoformat() if fate_ts is not None else None,
            "break_code": break_code,
            "superseded_by_ts": (
                superseded_by_ts.isoformat() if superseded_by_ts is not None else None
            ),
            "superseded_by_price": superseded_by_price,
            "wick_count": level.wick_count,
            "first_wicked_at": (
                level.first_wicked_at.isoformat()
                if level.first_wicked_at is not None
                else None
            ),
            "last_wicked_at": (
                level.last_wicked_at.isoformat() if level.last_wicked_at is not None else None
            ),
        }

    def drain_closed(self) -> list[dict[str, Any]]:
        closed, self._closed = self._closed, []
        return closed

    def active_levels(self) -> list[dict[str, Any]]:
        """Return the current ITR/LTR slot lifetimes at the data boundary."""
        rows: list[dict[str, Any]] = []
        for lane in LANES:
            for side in ("high", "low"):
                level = self._levels[lane][side]
                if level is None or level.spent:
                    continue
                rows.append(
                    self._lifecycle_row(
                        lane,
                        level,
                        fate="active",
                        fate_ts=None,
                        break_code=None,
                    )
                )
        return rows

    @staticmethod
    def _break_code(lane: str, level: StructureLevel) -> str:
        if level.side == "high":
            is_choch = level.relation not in {"HH", "seed"}
        else:
            is_choch = level.relation in {"HL", "seed"}

        if lane == "itr":
            return "ichoch" if is_choch else "isb"
        return "lchoch" if is_choch else "lsb"

    def _watch(self, lane: str, side: Side) -> dict[str, Any]:
        level = self._levels[lane][side]
        if level is None:
            return {
                "price": None,
                "pivot_ts": None,
                "relation": None,
                "code": None,
                "status": "no_level",
                "wick_count": 0,
                "last_wicked_at": None,
            }
        return {
            "price": level.price,
            "pivot_ts": level.pivot_ts.isoformat(),
            "relation": level.relation,
            "code": self._break_code(lane, level),
            "status": "spent" if level.spent else "watching",
            "wick_count": level.wick_count,
            "last_wicked_at": (
                level.last_wicked_at.isoformat() if level.last_wicked_at is not None else None
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        tiers: dict[str, Any] = {}
        for tier, sides in self._levels.items():
            tiers[tier] = {
                side: level.to_state() if level is not None else None
                for side, level in sides.items()
            }
            tiers[tier]["last_side"] = self._last_sides[tier]

        lanes: dict[str, Any] = {}
        for lane in LANES:
            lanes[lane] = {
                "watch_up": self._watch(lane, "high"),
                "watch_down": self._watch(lane, "low"),
                "last_break": (
                    dict(self._last_breaks[lane])
                    if self._last_breaks[lane] is not None
                    else None
                ),
                "last_wick": (
                    dict(self._last_wicks[lane])
                    if self._last_wicks[lane] is not None
                    else None
                ),
            }
        return {"tiers": tiers, "lanes": lanes}

    def export_state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "levels": {
                tier: {
                    side: level.to_state() if level is not None else None
                    for side, level in sides.items()
                }
                for tier, sides in self._levels.items()
            },
            "last_sides": dict(self._last_sides),
            "last_breaks": {
                lane: dict(event) if event is not None else None
                for lane, event in self._last_breaks.items()
            },
            "last_wicks": {
                lane: dict(event) if event is not None else None
                for lane, event in self._last_wicks.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported StructureBreaks state version")
        levels = state.get("levels") or {}
        self._levels = {
            tier: {
                side: (
                    StructureLevel.from_state(levels[tier][side])
                    if levels.get(tier, {}).get(side) is not None
                    else None
                )
                for side in ("high", "low")
            }
            for tier in TIERS
        }
        self._last_sides = {
            tier: (state.get("last_sides") or {}).get(tier) for tier in TIERS
        }
        self._last_breaks = {
            lane: (
                dict((state.get("last_breaks") or {}).get(lane))
                if (state.get("last_breaks") or {}).get(lane) is not None
                else None
            )
            for lane in LANES
        }
        self._last_wicks = {
            lane: (
                dict((state.get("last_wicks") or {}).get(lane))
                if (state.get("last_wicks") or {}).get(lane) is not None
                else None
            )
            for lane in LANES
        }
        self._closed = []
