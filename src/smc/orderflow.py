from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd


Direction = Literal["bull", "bear"]
Side = Literal["high", "low"]
Tier = Literal["itr", "ltr"]


@dataclass
class ProvisionalRecord:
    side: Side
    level: float
    level_relation: str
    level_ts: str
    opened_by: Literal["iwick", "ichoch"]
    opened_at: str
    wick_count: int
    reaffirm_level: float | None
    prev_level: float | None
    new_low: dict[str, Any] | None = None
    level_retired: bool = False
    lh_seen: bool = False
    # Bear provisional is the mirror of the published bull sketch.
    new_high: dict[str, Any] | None = None
    hl_seen: bool = False
    reaffirm_level_ts: str | None = None
    reaffirm_level_relation: str | None = None


class Orderflow:
    """Direction and provisional memory derived from one structure tier."""

    STATE_VERSION = 1
    SHOCK_THRESHOLD = 1.5

    def __init__(
        self,
        *,
        tier: Tier = "itr",
        shock_threshold: float = SHOCK_THRESHOLD,
    ) -> None:
        if tier not in {"itr", "ltr"}:
            raise ValueError("tier must be 'itr' or 'ltr'; STR defense uses its own tracker")
        if shock_threshold <= 0:
            raise ValueError("shock_threshold must be positive")
        self.tier = tier
        self.shock_threshold = float(shock_threshold)
        self.status: Literal["warmup", "active", "provisional"] = "warmup"
        self.direction: Direction | None = None
        self.streak = {"high": 0, "low": 0}
        self.pivots: dict[str, list[dict[str, Any]]] = {"high": [], "low": []}
        self.defended: dict[str, Any] | None = None
        self.defended_source: Literal["reducer", "fallback"] | None = None
        self.fallback_defended: dict[str, Any] | None = None
        self._fallback_set_at: str | None = None
        self.prev_level: float | None = None
        self._warmup_p_fallback = False
        self.provisional: ProvisionalRecord | None = None
        self.exit_path: str | None = None
        self.exited_at: str | None = None
        self.history: list[dict[str, Any]] = []
        self.cursor: dict[str, Any] = {"last_known_at": None}

    @staticmethod
    def _iso(value: Any) -> str:
        return pd.Timestamp(value).isoformat()

    @staticmethod
    def _event_value(event: Any, key: str, alias: str | None = None) -> Any:
        if isinstance(event, dict):
            if key in event:
                return event[key]
            if alias is not None:
                return event[alias]
            raise KeyError(key)
        if hasattr(event, key):
            return getattr(event, key)
        if alias is not None and hasattr(event, alias):
            return getattr(event, alias)
        raise AttributeError(key)

    def _pivot_facts(self, events: list[Any], ts: pd.Timestamp) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for event in events:
            if self._event_value(event, "tier") != self.tier:
                continue
            side = self._event_value(event, "side")
            if side not in {"high", "low"}:
                pivot_side = self._event_value(event, "pivot_side")
                side = "high" if int(pivot_side) == 1 else "low"
            pivot_timestamp = self._event_value(
                event, "pivot_timestamp", "pivot_ts"
            )
            price = self._event_value(event, "pivot_price", "price")
            try:
                event_timestamp = self._event_value(
                    event, "event_timestamp", "event_ts"
                )
            except (AttributeError, KeyError):
                event_timestamp = ts
            facts.append(
                {
                    "side": side,
                    "price": float(price),
                    "ts": self._iso(pivot_timestamp),
                    "relation": self._event_value(event, "relation"),
                    "event_ts": self._iso(event_timestamp),
                }
            )
        return facts

    def nearest_beyond(self, side: Side | str, price: float) -> dict[str, Any] | None:
        """Return the closest standing pivot on the defended side beyond price."""
        side = str(side)
        if side == "high":
            candidates = [pivot for pivot in self.pivots["high"] if pivot["price"] > price]
            return min(candidates, key=lambda pivot: pivot["price"], default=None)
        if side == "low":
            candidates = [pivot for pivot in self.pivots["low"] if pivot["price"] < price]
            return max(candidates, key=lambda pivot: pivot["price"], default=None)
        raise ValueError(f"Unsupported pivot side: {side}")

    def _slot(self, result: dict[str, Any], side: Side, ts: pd.Timestamp) -> dict[str, Any] | None:
        state = result.get("state") or {}
        tiers = state.get("tiers") or {}
        levels = tiers.get(self.tier) or {}
        slot = levels.get(side)
        if slot is None:
            return None
        broken_at = slot.get("broken_at")
        live = not bool(slot.get("spent")) or (
            broken_at is not None and pd.Timestamp(broken_at) == ts
        )
        if not live:
            return None
        return {
            "side": side,
            "price": float(slot["price"]),
            "ts": self._iso(slot.get("pivot_ts", slot.get("ts"))),
            "relation": slot.get("relation"),
            "event_ts": self._iso(slot.get("event_ts", slot.get("pivot_ts", slot.get("ts")))),
        }

    def _latest_slot(self, result: dict[str, Any], side: Side) -> dict[str, Any] | None:
        state = result.get("state") or {}
        slot = ((state.get("tiers") or {}).get(self.tier) or {}).get(side)
        if slot is None:
            return None
        return {
            "side": side,
            "price": float(slot["price"]),
            "ts": self._iso(slot.get("pivot_ts", slot.get("ts"))),
            "relation": slot.get("relation"),
            "event_ts": self._iso(slot.get("event_ts", slot.get("pivot_ts", slot.get("ts")))),
        }

    def _append_pivots(self, facts: list[dict[str, Any]]) -> None:
        for fact in facts:
            pivot = {"price": fact["price"], "ts": fact["ts"]}
            if pivot not in self.pivots[fact["side"]]:
                self.pivots[fact["side"]].append(pivot)

    def _prune_pivots(
        self,
        close: float,
        current_bar_pivots: list[dict[str, Any]],
    ) -> None:
        current = {
            (fact["side"], fact["price"], fact["ts"])
            for fact in current_bar_pivots
        }
        for side in ("high", "low"):
            retained = []
            for pivot in self.pivots[side]:
                identity = (side, float(pivot["price"]), pivot["ts"])
                if identity in current:
                    retained.append(pivot)
                    continue
                closed_through = close > pivot["price"] if side == "high" else close < pivot["price"]
                if not closed_through:
                    retained.append(pivot)
            self.pivots[side] = retained

    def _update_warmup(self, facts: list[dict[str, Any]]) -> None:
        for fact in facts:
            relation = fact["relation"]
            if relation == "seed":
                continue
            if fact["side"] == "high":
                if relation == "LH":
                    self.streak["high"] += 1
                elif relation == "HH":
                    self.streak["high"] = 0
            else:
                if relation == "HL":
                    self.streak["low"] += 1
                elif relation == "LL":
                    self.streak["low"] = 0

    def _warmup_direction(self) -> Direction | None:
        if self.streak["high"] >= 3:
            return "bear"
        if self.streak["low"] >= 3:
            return "bull"
        return None

    def _push_history(
        self,
        status: str,
        ts: pd.Timestamp,
        *,
        exit_path: str | None,
        swept_side: str | None = None,
        swept_level: float | None = None,
    ) -> None:
        of_status = (
            "bullish" if status == "bull" else "bearish" if status == "bear" else status
        )
        self.history.insert(
            0,
            {
                "of_status": of_status,
                "of_since": ts.isoformat(),
                "exit_path": exit_path,
                "swept_side": swept_side,
                "swept_level": swept_level,
            },
        )
        del self.history[3:]

    @staticmethod
    def _transition(
        ts: pd.Timestamp,
        from_status: str,
        to_status: str,
        direction: Direction | None,
        *,
        exit_path: str | None,
        swept_side: str | None = None,
        swept_level: float | None = None,
        opened_by: str | None = None,
    ) -> dict[str, Any]:
        return {
            "known_at": ts.isoformat(),
            "from_status": from_status,
            "to_status": to_status,
            "direction": direction,
            "exit_path": exit_path,
            "swept_side": swept_side,
            "swept_level": swept_level,
            "opened_by": opened_by,
        }

    def _conclude_warmup(
        self, direction: Direction, ts: pd.Timestamp, result: dict[str, Any]
    ) -> dict[str, Any]:
        self.status = "active"
        self.direction = direction
        self.provisional = None
        self.exit_path = "warmup_exit"
        self.exited_at = ts.isoformat()
        self.streak = {"high": 0, "low": 0}
        side: Side = "high" if direction == "bear" else "low"
        self.fallback_defended = None
        self._fallback_set_at = None
        self.defended = self._slot(result, side, ts)
        self.defended_source = "reducer" if self.defended is not None else None
        if self.defended is not None:
            beyond = self.nearest_beyond(side, self.defended["price"])
            if beyond is not None:
                self.prev_level = float(beyond["price"])
                self._warmup_p_fallback = False
            else:
                self.prev_level = self._warmup_extreme(side)
                self._warmup_p_fallback = self.prev_level is not None
        else:
            self.prev_level = self._warmup_extreme(side)
            self._warmup_p_fallback = self.prev_level is not None
        self._push_history(direction, ts, exit_path="warmup_exit")
        return self._transition(
            ts, "warmup", "active", direction, exit_path="warmup_exit"
        )

    def _warmup_extreme(self, side: Side) -> float | None:
        values = [float(pivot["price"]) for pivot in self.pivots[side]]
        if not values:
            return None
        return max(values) if side == "high" else min(values)

    def _set_active_defended(
        self,
        result: dict[str, Any],
        facts: list[dict[str, Any]],
        ts: pd.Timestamp,
    ) -> None:
        previous_defended_key = (
            (self.defended.get("side"), self.defended.get("price"), self.defended.get("ts"))
            if self.defended is not None
            else None
        )
        if self.direction is None:
            self.defended = None
            self.defended_source = None
            self.prev_level = None
            return
        side: Side = "high" if self.direction == "bear" else "low"
        if self.fallback_defended is not None:
            fresh_side_pivot = any(
                fact["side"] == side
                and self._fallback_set_at is not None
                and pd.Timestamp(fact["event_ts"]) > pd.Timestamp(self._fallback_set_at)
                for fact in facts
            )
            if fresh_side_pivot:
                self.fallback_defended = None
                self._fallback_set_at = None

        reducer_slot = self._slot(result, side, ts)
        if reducer_slot is not None:
            self.defended = reducer_slot
            self.defended_source = "reducer"
        elif self.fallback_defended is not None and self.fallback_defended["side"] == side:
            self.defended = dict(self.fallback_defended)
            self.defended_source = "fallback"
        else:
            self.defended = None
            self.defended_source = None
        if self.defended is None:
            if not self._warmup_p_is_standing(side):
                self.prev_level = None
                self._warmup_p_fallback = False
        else:
            current_defended_key = (
                self.defended.get("side"),
                self.defended.get("price"),
                self.defended.get("ts"),
            )
            if current_defended_key != previous_defended_key:
                self._warmup_p_fallback = False
            beyond = self.nearest_beyond(side, self.defended["price"])
            if beyond is not None:
                self.prev_level = float(beyond["price"])
                self._warmup_p_fallback = False
            elif not self._warmup_p_is_standing(side):
                self.prev_level = None
                self._warmup_p_fallback = False

    def _warmup_p_is_standing(self, side: Side) -> bool:
        return self._warmup_p_fallback and self.prev_level is not None and any(
            float(pivot["price"]) == self.prev_level for pivot in self.pivots[side]
        )

    @staticmethod
    def _against_close(direction: Direction, side: Side, close: float, level: float) -> bool:
        if direction == "bull" and side == "low":
            return close < level
        if direction == "bear" and side == "high":
            return close > level
        raise ValueError("Direction and defended side do not match")

    @staticmethod
    def _wicked(side: Side, high: float, low: float, close: float, level: float) -> bool:
        if side == "high":
            return high > level and close <= level
        return low < level and close >= level

    def _open_provisional(
        self,
        *,
        opened_by: Literal["iwick", "ichoch"],
        ts: pd.Timestamp,
        result: dict[str, Any],
        episodes: list[dict[str, Any]],
    ) -> None:
        if self.defended is None or self.direction is None:
            return
        side: Side = self.defended["side"]
        opposite: Side = "low" if side == "high" else "high"
        opposite_level = self._latest_slot(result, opposite)
        beyond = self.nearest_beyond(side, self.defended["price"])
        frozen_p = (
            float(beyond["price"])
            if beyond is not None
            else self.prev_level
            if self._warmup_p_is_standing(side)
            else None
        )
        self.provisional = ProvisionalRecord(
            side=side,
            level=float(self.defended["price"]),
            level_relation=str(self.defended.get("relation") or "seed"),
            level_ts=str(self.defended["ts"]),
            opened_by=opened_by,
            opened_at=ts.isoformat(),
            wick_count=1 if opened_by == "iwick" else 0,
            reaffirm_level=(float(opposite_level["price"]) if opposite_level else None),
            prev_level=frozen_p,
            reaffirm_level_ts=(str(opposite_level["ts"]) if opposite_level else None),
            reaffirm_level_relation=(
                str(opposite_level.get("relation")) if opposite_level else None
            ),
        )
        self.status = "provisional"
        self._push_history(
            "provisional",
            ts,
            exit_path=None,
            swept_side=side,
            swept_level=float(self.defended["price"]),
        )
        episodes.append(
            self._transition(
                ts,
                "active",
                "provisional",
                self.direction,
                exit_path=None,
                swept_side=side,
                swept_level=float(self.defended["price"]),
                opened_by=opened_by,
            )
        )

    def _new_provisional_evidence(self, facts: list[dict[str, Any]]) -> None:
        record = self.provisional
        if record is None:
            return
        for fact in facts:
            if record.side == "low":
                if fact["side"] == "low":
                    if fact["price"] < record.level and (
                        record.new_low is None or fact["price"] < record.new_low["price"]
                    ):
                        record.new_low = {"price": fact["price"], "ts": fact["ts"]}
                elif (
                    record.new_low is not None
                    and fact["relation"] == "LH"
                ):
                    record.lh_seen = True
            else:
                if fact["side"] == "high":
                    if fact["price"] > record.level and (
                        record.new_high is None or fact["price"] > record.new_high["price"]
                    ):
                        record.new_high = {"price": fact["price"], "ts": fact["ts"]}
                elif record.new_high is not None and fact["relation"] == "HL":
                    record.hl_seen = True

    def _resolve_provisional(
        self,
        *,
        target: Direction,
        exit_path: str,
        ts: pd.Timestamp,
        result: dict[str, Any],
        episodes: list[dict[str, Any]],
    ) -> None:
        record = self.provisional
        if record is None:
            return
        target_side: Side = "high" if target == "bear" else "low"
        if exit_path == "reaffirm":
            fallback = {
                "side": record.side,
                "price": record.level,
                "ts": record.level_ts,
                "relation": record.level_relation,
                "event_ts": record.opened_at,
            }
        else:
            fallback = (
                {
                    "side": target_side,
                    "price": float(record.reaffirm_level),
                    "ts": record.reaffirm_level_ts,
                    "relation": record.reaffirm_level_relation,
                    "event_ts": record.opened_at,
                }
                if record.reaffirm_level is not None
                else None
            )

        latest = self._latest_slot(result, target_side)
        if latest is not None and pd.Timestamp(latest["event_ts"]) > pd.Timestamp(record.opened_at):
            fallback = None
        self.fallback_defended = fallback
        self._fallback_set_at = ts.isoformat() if fallback is not None else None
        self._warmup_p_fallback = False
        self.status = "active"
        self.direction = target
        self.provisional = None
        self.exit_path = exit_path
        self.exited_at = ts.isoformat()
        self._push_history(target, ts, exit_path=exit_path)
        episodes.append(
            self._transition(
                ts,
                "provisional",
                "active",
                target,
                exit_path=exit_path,
            )
        )

    def _check_provisional(
        self,
        *,
        ts: pd.Timestamp,
        row: pd.Series,
        shock_ratio: float | None,
        result: dict[str, Any],
        facts: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        same_bar: bool,
    ) -> None:
        record = self.provisional
        if record is None or self.direction is None:
            return
        if not same_bar:
            self._new_provisional_evidence(facts)

        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        shock = shock_ratio is not None and shock_ratio > self.shock_threshold
        through = self._against_close(self.direction, record.side, close, record.level)
        if shock and through:
            target: Direction = "bear" if self.direction == "bull" else "bull"
            self._resolve_provisional(
                target=target,
                exit_path="shock",
                ts=ts,
                result=result,
                episodes=episodes,
            )
            return

        if self.direction == "bull":
            if (
                not same_bar
                and record.reaffirm_level is not None
                and close > record.reaffirm_level
            ):
                self._resolve_provisional(
                    target="bull", exit_path="reaffirm", ts=ts, result=result, episodes=episodes
                )
                return
            if record.prev_level is not None and close < record.prev_level:
                self._resolve_provisional(
                    target="bear", exit_path="expansion", ts=ts, result=result, episodes=episodes
                )
                return
            if (
                not same_bar
                and record.new_low is not None
                and record.lh_seen
                and close < float(record.new_low["price"])
            ):
                self._resolve_provisional(
                    target="bear", exit_path="structure", ts=ts, result=result, episodes=episodes
                )
                return
        else:
            if (
                not same_bar
                and record.reaffirm_level is not None
                and close < record.reaffirm_level
            ):
                self._resolve_provisional(
                    target="bear", exit_path="reaffirm", ts=ts, result=result, episodes=episodes
                )
                return
            if record.prev_level is not None and close > record.prev_level:
                self._resolve_provisional(
                    target="bull", exit_path="expansion", ts=ts, result=result, episodes=episodes
                )
                return
            if (
                not same_bar
                and record.new_high is not None
                and record.hl_seen
                and close > float(record.new_high["price"])
            ):
                self._resolve_provisional(
                    target="bull", exit_path="structure", ts=ts, result=result, episodes=episodes
                )
                return

        if same_bar:
            return
        if record.opened_by != "iwick" or record.level_retired:
            return
        wicked = self._wicked(record.side, high, low, close, record.level)
        if wicked:
            record.wick_count += 1
            return
        if through:
            if record.wick_count >= 2:
                target = "bear" if self.direction == "bull" else "bull"
                self._resolve_provisional(
                    target=target,
                    exit_path="wick_base",
                    ts=ts,
                    result=result,
                    episodes=episodes,
                )
            elif record.wick_count == 1:
                record.level_retired = True

    def _active_step(
        self,
        *,
        ts: pd.Timestamp,
        row: pd.Series,
        shock_ratio: float | None,
        result: dict[str, Any],
        facts: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
    ) -> None:
        if self.direction is None:
            return
        self._set_active_defended(result, facts, ts)
        defended = self.defended
        if defended is None:
            return
        close = float(row["close"])
        shock = shock_ratio is not None and shock_ratio > self.shock_threshold
        through = self._against_close(
            self.direction,
            defended["side"],
            close,
            float(defended["price"]),
        )
        if shock and through:
            old_status = self.status
            target: Direction = "bear" if self.direction == "bull" else "bull"
            target_side: Side = "high" if target == "bear" else "low"
            latest = self._latest_slot(result, target_side)
            self.fallback_defended = latest
            self._fallback_set_at = ts.isoformat() if latest is not None else None
            self._warmup_p_fallback = False
            self.direction = target
            self.exit_path = "shock"
            self.exited_at = ts.isoformat()
            self._push_history(target, ts, exit_path="shock")
            episodes.append(
                self._transition(
                    ts, old_status, "active", target, exit_path="shock"
                )
            )
            self._set_active_defended(result, facts, ts)
            return

        high, low = float(row["high"]), float(row["low"])
        if self._wicked(defended["side"], high, low, close, float(defended["price"])):
            self._open_provisional(
                opened_by="iwick",
                ts=ts,
                result=result,
                episodes=episodes,
            )
        elif through:
            self._open_provisional(
                opened_by="ichoch",
                ts=ts,
                result=result,
                episodes=episodes,
            )

    def step(
        self,
        result: dict[str, Any],
        row: pd.Series,
        *,
        shock_ratio: float | None,
    ) -> dict[str, Any]:
        ts_value = getattr(row, "name", None)
        if ts_value is None and isinstance(row, dict):
            ts_value = row.get("known_at", row.get("timestamp"))
        if ts_value is None:
            raise ValueError("Orderflow rows must carry their known_at timestamp")
        ts = pd.Timestamp(ts_value)
        previous_ts = self.cursor["last_known_at"]
        if previous_ts is not None and ts <= pd.Timestamp(previous_ts):
            raise ValueError(f"Orderflow bars must increase: {ts} <= {previous_ts}")
        if shock_ratio is not None:
            shock_ratio = float(shock_ratio)
            if pd.isna(shock_ratio):
                shock_ratio = None

        facts = self._pivot_facts(result.get("events", []), ts)
        current_bar_pivots = [fact for fact in facts]
        self._append_pivots(facts)
        episodes: list[dict[str, Any]] = []
        status_at_start = self.status
        shock = (
            status_at_start != "warmup"
            and shock_ratio is not None
            and shock_ratio > self.shock_threshold
        )

        if status_at_start == "warmup":
            self._update_warmup(facts)
            direction = self._warmup_direction()
            if direction is None:
                self._prune_pivots(float(row["close"]), current_bar_pivots)
            else:
                # Warmup does not inspect the exit bar as an active test. Prune
                # old levels before choosing P; current-bar pivots remain fresh.
                self._prune_pivots(float(row["close"]), current_bar_pivots)
                episodes.append(self._conclude_warmup(direction, ts, result))
        elif status_at_start == "active":
            self._active_step(
                ts=ts,
                row=row,
                shock_ratio=shock_ratio,
                result=result,
                facts=facts,
                episodes=episodes,
            )
            if self.status == "provisional":
                # A provisional opened on this bar can resolve only through the
                # same-bar shock/expansion check, not same-bar pivot or wick evidence.
                self._check_provisional(
                    ts=ts,
                    row=row,
                    shock_ratio=shock_ratio,
                    result=result,
                    facts=[],
                    episodes=episodes,
                    same_bar=True,
                )
            self._prune_pivots(float(row["close"]), current_bar_pivots)
        else:
            self._check_provisional(
                ts=ts,
                row=row,
                shock_ratio=shock_ratio,
                result=result,
                facts=facts,
                episodes=episodes,
                same_bar=False,
            )
            self._prune_pivots(float(row["close"]), current_bar_pivots)

        if self.status == "active" and status_at_start != "warmup":
            self._set_active_defended(result, facts, ts)

        self.cursor["last_known_at"] = ts.isoformat()
        snapshot = self.snapshot()
        return {
            "snapshot": snapshot,
            "episodes": episodes,
            "shock_ratio": shock_ratio,
            "shock": shock,
        }

    def snapshot(self) -> dict[str, Any]:
        record = asdict(self.provisional) if self.provisional is not None else None
        return {
            "status": self.status,
            "direction": self.direction,
            "streak": dict(self.streak) if self.status == "warmup" else None,
            "defended": (
                {
                    "side": self.defended["side"],
                    "price": self.defended["price"],
                    "ts": self.defended["ts"],
                }
                if self.defended is not None
                else None
            ),
            "defended_source": self.defended_source,
            "prev_level": self.provisional.prev_level if self.provisional else self.prev_level,
            "provisional_open": record,
            "exit_path": self.exit_path,
            "exited_at": self.exited_at,
            "history": deepcopy(self.history),
        }

    def export_state(self) -> dict[str, Any]:
        state = {
            "state_version": self.STATE_VERSION,
            "shock_threshold": self.shock_threshold,
            "status": self.status,
            "direction": self.direction,
            "streak": dict(self.streak),
            "pivots": deepcopy(self.pivots),
            "defended": deepcopy(self.defended),
            "defended_source": self.defended_source,
            "fallback_defended": deepcopy(self.fallback_defended),
            "fallback_set_at": self._fallback_set_at,
            "prev_level": self.prev_level,
            "warmup_p_fallback": self._warmup_p_fallback,
            "provisional": asdict(self.provisional) if self.provisional is not None else None,
            "exit_path": self.exit_path,
            "exited_at": self.exited_at,
            "history": deepcopy(self.history),
            "cursor": dict(self.cursor),
        }
        if self.tier != "itr":
            state["tier"] = self.tier
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        saved_tier = state.get("tier", "itr")
        if saved_tier != self.tier:
            raise ValueError(
                f"Orderflow state tier mismatch: saved {saved_tier!r}, requested {self.tier!r}"
            )
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported Orderflow state version")
        threshold = float(state.get("shock_threshold", self.SHOCK_THRESHOLD))
        if threshold <= 0:
            raise ValueError("shock_threshold must be positive")
        self.shock_threshold = threshold
        status = state.get("status", "warmup")
        if status not in {"warmup", "active", "provisional"}:
            raise ValueError(f"Unsupported orderflow status: {status}")
        direction = state.get("direction")
        if direction not in {None, "bull", "bear"}:
            raise ValueError(f"Unsupported orderflow direction: {direction}")
        self.status = status
        self.direction = direction
        streak = state.get("streak") or {}
        self.streak = {"high": int(streak.get("high", 0)), "low": int(streak.get("low", 0))}
        pivots = state.get("pivots") or {}
        self.pivots = {
            side: [
                {"price": float(item["price"]), "ts": self._iso(item["ts"])}
                for item in pivots.get(side, [])
            ]
            for side in ("high", "low")
        }
        self.defended = deepcopy(state.get("defended"))
        self.defended_source = state.get("defended_source")
        self.fallback_defended = deepcopy(state.get("fallback_defended"))
        self._fallback_set_at = state.get("fallback_set_at")
        prev_level = state.get("prev_level")
        self.prev_level = float(prev_level) if prev_level is not None else None
        self._warmup_p_fallback = bool(state.get("warmup_p_fallback", False))
        provisional = state.get("provisional")
        self.provisional = ProvisionalRecord(**provisional) if provisional is not None else None
        self.exit_path = state.get("exit_path")
        self.exited_at = state.get("exited_at")
        self.history = deepcopy(state.get("history") or [])[:3]
        cursor = state.get("cursor") or {}
        self.cursor = {"last_known_at": cursor.get("last_known_at")}
