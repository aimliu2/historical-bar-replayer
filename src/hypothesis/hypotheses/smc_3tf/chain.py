from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ...base import Trigger


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso(value: Any) -> str | None:
    return None if value is None else _utc(value).isoformat()


def wave_band_exit(ctx: dict[str, Any], wave: str) -> bool:
    """A closed 5m bar has already left the as-of 1H band on the wave side."""
    band = (((ctx.get("1h") or {}).get("ta") or {}).get("band") or {})
    close = (ctx.get("bar_5m") or {}).get("close")
    edge = band.get("lower" if wave == "top" else "upper")
    return close is not None and edge is not None and (
        float(close) < float(edge) if wave == "top" else float(close) > float(edge)
    )


def _ema_gap(hourly: dict[str, Any], wave: str) -> float | None:
    ema = ((hourly.get("ta") or {}).get("ema") or {})
    ema5, ema9 = ema.get("5"), ema.get("9")
    if ema5 is None or ema9 is None:
        return None
    return float(ema5) - float(ema9) if wave == "top" else float(ema9) - float(ema5)


class Chain:
    """One per allowed arm; Layer 4 owns its postponement and 5m search state."""

    STATE_VERSION = 2
    TERMINAL = {"triggered", "wave running", "disarmed", "skipped (shape A)", "merged", "unresolved", "retry_lapsed"}

    def __init__(
        self,
        arm: dict[str, Any],
        ctx: dict[str, Any],
        *,
        settings: dict[str, Any],
        retry: bool = False,
        hypothesis_id: str | None = None,
    ) -> None:
        self.arm = deepcopy(arm)
        self.hypothesis_id = str(hypothesis_id or arm.get("hypothesis") or "")
        if not self.hypothesis_id:
            raise ValueError("Chain requires its owning hypothesis id")
        self.wave = str(arm["wave"])
        self.side = str(arm["side"])
        self.trade_type = str(arm["type"])
        self.arm_time = str(arm["arm_time"])
        self.retry = bool(retry)
        self.retry_policy = arm.get("retry_policy")
        self.postpone_mode = str(settings.get("postpone_mode", "touch"))
        self.skip_shape = str(settings.get("skip_shape", "B")).upper()
        self.near_cross_pips = float(settings.get("near_cross_pips", 1.0))
        self.pip_size = float(ctx.get("pip_size") or 0.0001)
        self.chain_id = str(
            arm.get("chain_id")
            or f"{self.hypothesis_id}:{self.wave}:{self.arm_time}:{self.trade_type}:{(arm.get('pivot_confirmation') or {}).get('event_id', '')}"
        )
        self.state = (
            "triggering"
            if self.retry and self.retry_policy not in {"3T", "3S"}
            else "postponing"
            if self.trade_type in {"2a", "3"} and not self.retry
            else "arming_5m"
        )
        self.bound_at: str | None = None
        self.old_side = False
        self.cut_short = False
        self.share_search = False
        self.cut_short_by_arm_time: str | None = None
        self.landing_at: str | None = None if self.trade_type in {"2a", "3"} and not self.retry else self.arm_time
        # The manager resolves ordinary arm-bar landings after it has added all
        # arms delivered on this close. Retry landings are deliberately kept
        # separate: a retry on a regular arm's bar lapses instead of merging.
        self.landing_resolved = self.landing_at is not None and self.retry
        self.postponed_at: str | None = None
        self.armed_at: str | None = None
        self.armed_pivot: dict[str, Any] | None = None
        self.extreme_high: float | None = None
        self.extreme_low: float | None = None
        if self.retry or self.trade_type in {"1", "2b"}:
            self.extreme_high = float(arm["high"])
            self.extreme_low = float(arm["low"])
        self.type3_hour_landing: str | None = None
        self.type3_hour_high: float | None = None
        self.type3_hour_low: float | None = None
        self.last_hour_seen = str((ctx.get("1h") or {}).get("known_at") or self.arm_time)
        self.last_ema_gap = _ema_gap(ctx.get("1h") or {}, self.wave)
        self._last_output_time: str | None = None

    def _event(self, kind: str, time: Any, **details: Any) -> dict[str, Any]:
        return {"chain_id": self.chain_id, "kind": kind, "time": _iso(time), **details}

    def bind(self, expires_at: Any) -> dict[str, Any] | None:
        if self.old_side or self.state in self.TERMINAL or self.share_search:
            return None
        stamp = _iso(expires_at)
        if stamp is None:
            return None
        if self.bound_at is None or _utc(stamp) < _utc(self.bound_at):
            self.bound_at = stamp
            return self._event("bound", stamp, expires_at=stamp)
        return None

    def mark_old_side(self, time: Any) -> dict[str, Any] | None:
        if self.state in self.TERMINAL:
            return None
        if self.retry_policy == "3S":
            self.state = "retry_lapsed"
            return self._event("retry_lapsed", time, reason="1D gate switch")
        self.old_side = True
        prior = self.bound_at
        self.bound_at = None
        return self._event("old_side", time, cleared_bound=prior)

    def disarm_for_type3(self, time: Any) -> dict[str, Any] | None:
        if self.trade_type != "3" or self.state != "postponing":
            return None
        self.state = "disarmed"
        self.cut_short = True
        self.share_search = True
        return self._event("disarmed", time, reason="next same-side type 3 arm")

    def _postponement_hit(self, hourly: dict[str, Any]) -> bool:
        gap = _ema_gap(hourly, self.wave)
        if gap is None:
            self.last_ema_gap = gap
            return False
        previous = self.last_ema_gap
        touched = (
            previous is not None
            and previous >= self.near_cross_pips * self.pip_size
            and gap < self.near_cross_pips * self.pip_size
        )
        crossed = previous is not None and previous > 0.0 and gap <= 0.0
        self.last_ema_gap = gap
        return crossed if self.postpone_mode == "cross" else (touched or crossed)

    def _land_type2a(self, hourly: dict[str, Any], time: Any, *, shared_search: bool) -> dict[str, Any]:
        self.postponed_at = str(hourly.get("known_at") or _iso(time))
        self.landing_at = self.postponed_at
        self.extreme_high = float(hourly["high"])
        self.extreme_low = float(hourly["low"])
        if not shared_search:
            self.state = "arming_5m"
            self.share_search = False
        return self._event("postponed", time, landing_at=self.landing_at, mode=self.postpone_mode)

    def _land_type3(self, time: Any, *, shared_search: bool) -> dict[str, Any]:
        self.postponed_at = _iso(time)
        self.landing_at = self.type3_hour_landing or _utc(time).ceil("1h").isoformat()
        self.extreme_high = self.type3_hour_high
        self.extreme_low = self.type3_hour_low
        self.type3_hour_landing = None
        self.type3_hour_high = None
        self.type3_hour_low = None
        if not shared_search:
            self.state = "arming_5m"
            self.share_search = False
        return self._event("postponed", time, landing_at=self.landing_at, concept="B")

    def _update_type3_wait_extreme(self, ctx: dict[str, Any]) -> None:
        if self.trade_type != "3":
            return
        waiting = self.state == "postponing" or (self.share_search and self.state in {"unresolved", "disarmed"})
        now = ctx.get("time")
        if not waiting or now is None or _utc(now) <= _utc(self.arm_time):
            return
        landing = _utc(now).ceil("1h").isoformat()
        if landing != self.type3_hour_landing:
            self.type3_hour_landing = landing
            self.type3_hour_high = None
            self.type3_hour_low = None
        bar = ctx.get("bar_5m") or {}
        high, low = bar.get("high"), bar.get("low")
        if high is not None:
            self.type3_hour_high = float(high) if self.type3_hour_high is None else max(self.type3_hour_high, float(high))
        if low is not None:
            self.type3_hour_low = float(low) if self.type3_hour_low is None else min(self.type3_hour_low, float(low))

    def prepare(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Handle 1H postponement / type-3 band return before 5m processing."""
        now = str(ctx.get("time") or "")
        if not now:
            return []
        hourly = ctx.get("1h") or {}
        self._update_type3_wait_extreme(ctx)
        events: list[dict[str, Any]] = []
        share_only = self.state in {"unresolved", "disarmed"} and self.share_search

        if self.state == "postponing" and self.trade_type == "2a":
            if self.bound_at is not None and not self.old_side and _utc(now) >= _utc(self.bound_at):
                self.state = "unresolved"
                self.cut_short = True
                self.share_search = True
                events.append(self._event("unresolved", now, reason="next allowed same-side 1H arm"))
                share_only = True
            else:
                hour_time = str(hourly.get("known_at") or "")
                if hour_time and hour_time != self.last_hour_seen:
                    self.last_hour_seen = hour_time
                    if _utc(hour_time) > _utc(self.arm_time) and self._postponement_hit(hourly):
                        events.append(self._land_type2a(hourly, now, shared_search=False))
                        self.landing_resolved = False
                        return events

        if self.state == "postponing" and self.trade_type == "3":
            if _utc(now) > _utc(self.arm_time):
                band = ((hourly.get("ta") or {}).get("band") or {})
                lower, upper = band.get("lower"), band.get("upper")
                close = (ctx.get("bar_5m") or {}).get("close")
                if lower is not None and upper is not None and close is not None:
                    if float(lower) <= float(close) <= float(upper):
                        events.append(self._land_type3(now, shared_search=False))
                        self.landing_resolved = False
                        return events

        if share_only:
            if self.trade_type == "2a":
                hour_time = str(hourly.get("known_at") or "")
                if hour_time and hour_time != self.last_hour_seen:
                    self.last_hour_seen = hour_time
                    if _utc(hour_time) > _utc(self.arm_time) and self._postponement_hit(hourly):
                        events.append(self._land_type2a(hourly, now, shared_search=True))
                        self.landing_resolved = False
            elif self.trade_type == "3" and _utc(now) > _utc(self.arm_time):
                band = ((hourly.get("ta") or {}).get("band") or {})
                lower, upper = band.get("lower"), band.get("upper")
                close = (ctx.get("bar_5m") or {}).get("close")
                if lower is not None and upper is not None and close is not None:
                    if float(lower) <= float(close) <= float(upper):
                        events.append(self._land_type3(now, shared_search=True))
                        self.landing_resolved = False
            return events

        return events

    def resume_shared_landing(self, next_bound: str | None, time: Any) -> dict[str, Any]:
        prior = self.state
        self.state = "arming_5m"
        self.share_search = False
        self.bound_at = next_bound
        self.landing_resolved = True
        return self._event("shared_landing", time, from_state=prior, landing_at=self.landing_at, expires_at=next_bound)

    def abandon_shared_search(self, time: Any) -> dict[str, Any] | None:
        if not self.share_search:
            return None
        self.share_search = False
        return self._event("shared_landing_not_matched", time, landing_at=self.landing_at)

    def _inside_arm_band(self, relation: str, price: float, hourly: dict[str, Any]) -> bool:
        band = ((hourly.get("ta") or {}).get("band") or {})
        lower, upper = band.get("lower"), band.get("upper")
        if lower is None or upper is None:
            return False
        if self.trade_type in {"1", "2a"}:
            return price > float(upper) if relation == "HL" else price < float(lower)
        return True

    def _valid_pivot(self, event: dict[str, Any], hourly: dict[str, Any], now: Any) -> bool:
        if str(event.get("tier", "")).lower() != "itr":
            return False
        if not self.retry and self.trade_type in {"1", "2a"}:
            cutoff = self.postponed_at if self.trade_type == "2a" and self.postponed_at else self.arm_time
            confirmed_at = event.get("event_ts") or now
            if _utc(confirmed_at) <= _utc(cutoff):
                return False
        wanted_relation = "HL" if self.wave == "top" else "LH"
        if str(event.get("relation", "")).upper() != wanted_relation:
            return False
        price = event.get("price")
        return price is not None and self._inside_arm_band(wanted_relation, float(price), hourly)

    def _valid_break(self, event: dict[str, Any]) -> bool:
        if str(event.get("code", "")).lower() != "ichoch":
            return False
        wanted_relation, wanted_direction = ("HL", "down") if self.wave == "top" else ("LH", "up")
        return (
            str(event.get("level_relation", "")).upper() == wanted_relation
            and str(event.get("break_direction", "")).lower() == wanted_direction
            and event.get("level_price") is not None
        )

    def _update_extreme(self, ctx: dict[str, Any]) -> None:
        bar = ctx.get("bar_5m") or {}
        high, low = bar.get("high"), bar.get("low")
        if high is not None:
            self.extreme_high = float(high) if self.extreme_high is None else max(self.extreme_high, float(high))
        if low is not None:
            self.extreme_low = float(low) if self.extreme_low is None else min(self.extreme_low, float(low))

    def _advance_3s(self, ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], Trigger | None]:
        now = str(ctx["time"])
        if wave_band_exit(ctx, self.wave):
            self.state = "retry_lapsed"
            return [self._event("retry_lapsed", now, reason="band exit")], None
        self._update_extreme(ctx)
        events = []
        lower = ctx.get("5m") or {}
        cutoff = _utc(self.arm["sl_closed_at"])
        for item in lower.get("breaks") or []:
            confirmed = _utc(item.get("event_ts") or now)
            if (str(item.get("lane", "")).lower() != "itr" or not self._valid_break(item)
                    or confirmed <= cutoff or confirmed > _utc(now)):
                continue
            self.state = "triggering"
            self.armed_at = confirmed.isoformat()
            self.armed_pivot = deepcopy(item)
            events.append(self._event("armed_5m", now, pivot=deepcopy(item)))
        if self.armed_pivot is None:
            return events, None
        for pivot in lower.get("pivots") or []:
            confirmed = _utc(pivot.get("event_ts") or now)
            relation = "LL" if self.wave == "top" else "HH"
            if (str(pivot.get("tier", "")).lower() != "itr"
                    or str(pivot.get("relation", "")).upper() != relation
                    or confirmed <= cutoff or confirmed < _utc(self.armed_at) or confirmed > _utc(now)):
                continue
            item = self.armed_pivot
            trigger = Trigger(
                hypothesis=self.hypothesis_id, trade_type=self.trade_type, wave=self.wave, side=self.side,
                trigger_close=now, entry_level=float(item["level_price"]),
                sl_reference=self.extreme_high if self.wave == "top" else self.extreme_low,
                arm_time=self.arm_time, expires_at=self.bound_at, retry=True,
                metadata={"chain_id": self.chain_id, "landing_at": self.landing_at,
                          "postponed_at": self.postponed_at, "armed_at": self.armed_at,
                          "armed_pivot": deepcopy(item), "confirmed_pivot": deepcopy(pivot),
                          "broken_level_ts": item.get("level_ts"),
                          "broken_level_relation": item.get("level_relation"), "old_side": self.old_side,
                          "retry_record_id": self.arm["retry_record_id"], "retry_policy": "3S",
                          "sl_closed_at": self.arm["sl_closed_at"]},
            )
            self.state = "triggered"
            events.append(self._event("triggered", now, trigger=trigger.to_dict()))
            return events, trigger
        return events, None

    def advance_lower(self, ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], Trigger | None]:
        if self.state not in {"arming_5m", "triggering"}:
            return [], None
        now = str(ctx.get("time") or "")
        if self.bound_at is not None and not self.old_side and _utc(now) >= _utc(self.bound_at):
            self.state = "retry_lapsed" if self.retry_policy == "3S" else "wave running"
            return [self._event(self.state, now, reason="exclusive same-side 1H arm bound")], None

        if self.retry_policy == "3S":
            return self._advance_3s(ctx)

        hourly = ctx.get("1h") or {}
        bar_close = (ctx.get("bar_5m") or {}).get("close")
        if self.trade_type == "2a" and self.skip_shape == "A" and self.postponed_at is not None:
            ema200 = ((hourly.get("ta") or {}).get("ema") or {}).get("200")
            if ema200 is not None and bar_close is not None:
                skipped = float(bar_close) < float(ema200) if self.wave == "top" else float(bar_close) > float(ema200)
                if skipped:
                    self.state = "skipped (shape A)"
                    return [self._event("skipped", now, reason="EMA 200 crossed after type 2a postponement")], None

        self._update_extreme(ctx)
        events: list[dict[str, Any]] = []
        lower = ctx.get("5m") or {}
        if self.state == "arming_5m":
            for pivot in lower.get("pivots") or []:
                if self._valid_pivot(pivot, hourly, now):
                    self.state = "triggering"
                    self.armed_at = now
                    self.armed_pivot = deepcopy(pivot)
                    events.append(self._event("armed_5m", now, pivot=deepcopy(pivot)))
                    break

        if self.state != "triggering":
            return events, None
        for item in lower.get("breaks") or []:
            if not self._valid_break(item):
                continue
            trigger = Trigger(
                hypothesis=self.hypothesis_id,
                trade_type=self.trade_type,
                wave=self.wave,
                side=self.side,
                trigger_close=now,
                entry_level=float(item["level_price"]),
                sl_reference=(
                    self.extreme_high if self.wave == "top" else self.extreme_low
                ),
                arm_time=self.arm_time,
                expires_at=self.bound_at,
                retry=self.retry,
                metadata={
                    "chain_id": self.chain_id,
                    "landing_at": self.landing_at,
                    "postponed_at": self.postponed_at,
                    "armed_at": self.armed_at,
                    "armed_pivot": deepcopy(self.armed_pivot),
                    "broken_level_ts": item.get("level_ts"),
                    "broken_level_relation": item.get("level_relation"),
                    "old_side": self.old_side,
                    "retry_record_id": self.arm.get("retry_record_id"),
                    **({"retry_policy": self.retry_policy} if self.retry_policy else {}),
                    **({key: self.arm.get(key) for key in ("rearm_hour", "ema200_breached_at", "ema5x9_touched_at")} if self.retry_policy == "3T" else {}),
                },
            )
            self.state = "triggered"
            events.append(self._event("triggered", now, trigger=trigger.to_dict()))
            return events, trigger
        return events, None

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "hypothesis_id": self.hypothesis_id,
            "arm": deepcopy(self.arm),
            "wave": self.wave,
            "side": self.side,
            "type": self.trade_type,
            "arm_time": self.arm_time,
            "retry": self.retry,
            "postpone_mode": self.postpone_mode,
            "skip_shape": self.skip_shape,
            "near_cross_pips": self.near_cross_pips,
            "pip_size": self.pip_size,
            "chain_id": self.chain_id,
            "state": self.state,
            "bound_at": self.bound_at,
            "old_side": self.old_side,
            "cut_short": self.cut_short,
            "cut_short_by_arm_time": self.cut_short_by_arm_time,
            "share_search": self.share_search,
            "landing_at": self.landing_at,
            "landing_resolved": self.landing_resolved,
            "postponed_at": self.postponed_at,
            "armed_at": self.armed_at,
            "armed_pivot": deepcopy(self.armed_pivot),
            "extreme_high": self.extreme_high,
            "extreme_low": self.extreme_low,
            "type3_hour_landing": self.type3_hour_landing,
            "type3_hour_high": self.type3_hour_high,
            "type3_hour_low": self.type3_hour_low,
            "last_hour_seen": self.last_hour_seen,
            "last_ema_gap": self.last_ema_gap,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Chain:
        if int(state.get("state_version", cls.STATE_VERSION)) != cls.STATE_VERSION:
            raise ValueError("Unsupported Chain state version")
        # Construction sets the common rule defaults; the snapshot then restores the exact run state.
        chain = cls(
            state["arm"],
            {"1h": {"known_at": state.get("last_hour_seen")}, "pip_size": state.get("pip_size")},
            settings={
                "postpone_mode": state.get("postpone_mode", "touch"),
                "skip_shape": state.get("skip_shape", "B"),
                "near_cross_pips": state.get("near_cross_pips", 1.0),
            },
            retry=bool(state.get("retry", False)),
            hypothesis_id=state.get("hypothesis_id") or (state.get("arm") or {}).get("hypothesis"),
        )
        chain.wave = str(state["wave"])
        chain.side = str(state["side"])
        chain.trade_type = str(state["type"])
        chain.arm_time = str(state["arm_time"])
        chain.chain_id = str(state["chain_id"])
        chain.state = str(state["state"])
        chain.bound_at = state.get("bound_at")
        chain.old_side = bool(state.get("old_side", False))
        chain.cut_short = bool(state.get("cut_short", False))
        chain.cut_short_by_arm_time = state.get("cut_short_by_arm_time")
        chain.share_search = bool(state.get("share_search", False))
        chain.landing_at = state.get("landing_at")
        chain.landing_resolved = bool(state.get("landing_resolved", False))
        chain.postponed_at = state.get("postponed_at")
        chain.armed_at = state.get("armed_at")
        chain.armed_pivot = deepcopy(state.get("armed_pivot"))
        chain.extreme_high = float(state["extreme_high"]) if state.get("extreme_high") is not None else None
        chain.extreme_low = float(state["extreme_low"]) if state.get("extreme_low") is not None else None
        chain.type3_hour_landing = state.get("type3_hour_landing")
        chain.type3_hour_high = float(state["type3_hour_high"]) if state.get("type3_hour_high") is not None else None
        chain.type3_hour_low = float(state["type3_hour_low"]) if state.get("type3_hour_low") is not None else None
        chain.last_hour_seen = str(state.get("last_hour_seen") or chain.arm_time)
        chain.last_ema_gap = float(state["last_ema_gap"]) if state.get("last_ema_gap") is not None else None
        return chain


class ChainManager:
    """Apply exclusive same-side bounds and same-landing merges across chains."""

    STATE_VERSION = 1

    def __init__(self, *, settings: dict[str, Any], hypothesis_id: str | None = None) -> None:
        self.settings = deepcopy(settings)
        self.hypothesis_id = str(hypothesis_id) if hypothesis_id is not None else None
        self.chains: list[Chain] = []
        self.allowed_arm_times: dict[str, list[str]] = {"top": [], "bottom": []}

    def mark_old_side(self, side: str, time: Any) -> list[dict[str, Any]]:
        events = []
        for chain in self.chains:
            if chain.wave == side:
                event = chain.mark_old_side(time)
                if event is not None:
                    events.append(event)
        return events

    def _add_arms(self, arms: list[dict[str, Any]], ctx: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        retries: list[dict[str, Any]] = []
        for arm in arms:
            if not arm.get("allowed"):
                continue
            if arm.get("retry"):
                retries.append(arm)
                continue
            key = (str(arm["wave"]), str(arm["arm_time"]))
            grouped.setdefault(key, arm)

        for (wave, arm_time), arm in grouped.items():
            arm_hypothesis = str(arm.get("hypothesis") or self.hypothesis_id or "")
            if not arm_hypothesis:
                raise ValueError("ChainManager arm must identify its hypothesis")
            if self.hypothesis_id is not None and arm_hypothesis != self.hypothesis_id:
                raise ValueError("ChainManager arm belongs to a different hypothesis")
            for existing in list(self.chains):
                if existing.wave != wave or existing.state in Chain.TERMINAL:
                    continue
                if existing.trade_type == "3" and existing.state == "postponing":
                    if str(existing.arm_time) != arm_time and str(arm.get("type")) == "3":
                        event = existing.disarm_for_type3(arm_time)
                        if event is not None:
                            existing.cut_short_by_arm_time = arm_time
                            events.append(event)
                    continue
                event = existing.bind(arm_time)
                if event is not None:
                    existing.cut_short_by_arm_time = arm_time
                    events.append(event)
            self.allowed_arm_times[wave].append(arm_time)
            self.allowed_arm_times[wave].sort(key=_utc)
            chain = Chain(arm, ctx, settings=self.settings, hypothesis_id=arm_hypothesis)
            self.chains.append(chain)
            events.append(
                {
                    "chain_id": chain.chain_id,
                    "kind": "created",
                    "time": arm_time,
                    "wave": wave,
                    "type": chain.trade_type,
                    "state": chain.state,
                }
            )

        regular_keys = set(grouped)
        for arm in retries:
            arm_hypothesis = str(arm.get("hypothesis") or self.hypothesis_id or "")
            if not arm_hypothesis:
                raise ValueError("Retry arm must identify its hypothesis")
            if self.hypothesis_id is not None and arm_hypothesis != self.hypothesis_id:
                raise ValueError("Retry arm belongs to a different hypothesis")
            key = (str(arm["wave"]), str(arm["arm_time"]))
            if key in regular_keys:
                events.append({
                    "chain_id": str(arm.get("chain_id") or ""),
                    "kind": "retry_lapsed",
                    "time": str(arm["arm_time"]),
                    "reason": "regular same-side arm on re-entry close",
                })
                continue
            chain = Chain(arm, ctx, settings=self.settings, retry=True, hypothesis_id=arm_hypothesis)
            chain.bound_at = self._future_bound(chain)
            self.chains.append(chain)
            events.append({
                "chain_id": chain.chain_id,
                "kind": "retry_created",
                "retry_policy": chain.retry_policy,
                "time": chain.arm_time,
                "wave": chain.wave,
                "type": chain.trade_type,
                "record_id": arm.get("retry_record_id"),
                "state": chain.state,
            })
        return events

    def _future_bound(self, chain: Chain) -> str | None:
        if chain.old_side:
            return None
        landing = _utc(chain.landing_at) if chain.landing_at is not None else None
        if landing is None:
            return None
        later = [value for value in self.allowed_arm_times[chain.wave] if _utc(value) > landing]
        return min(later, key=_utc) if later else None

    def _merge_landings(self, now: str) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[Chain]] = {}
        for chain in self.chains:
            if chain.landing_at is None or chain.landing_resolved or chain.state == "merged":
                continue
            groups.setdefault((chain.wave, chain.landing_at), []).append(chain)
        events: list[dict[str, Any]] = []
        for (_, landing), candidates in groups.items():
            active = [chain for chain in candidates if chain.state in {"arming_5m", "triggering"}]
            shared = [chain for chain in candidates if chain.share_search]
            if not active:
                for chain in shared:
                    successor = next(
                        (
                            other
                            for other in self.chains
                            if other.wave == chain.wave
                            and other.arm_time == chain.cut_short_by_arm_time
                        ),
                        None,
                    )
                    if successor is not None and not successor.landing_resolved:
                        continue
                    chain.landing_resolved = True
                    event = chain.abandon_shared_search(now)
                    if event is not None:
                        events.append(event)
                continue
            merged = list({chain.chain_id: chain for chain in [*active, *shared]}.values())
            if len(merged) == 1:
                merged[0].landing_resolved = True
                if merged[0].share_search:
                    event = merged[0].abandon_shared_search(now)
                    if event is not None:
                        events.append(event)
                continue
            winner = min(merged, key=lambda chain: (_utc(chain.arm_time), chain.chain_id))
            newest = max(merged, key=lambda chain: (_utc(chain.arm_time), chain.chain_id))
            winner.extreme_high = newest.extreme_high
            winner.extreme_low = newest.extreme_low
            if winner.share_search:
                events.append(winner.resume_shared_landing(self._future_bound(newest), now))
            else:
                winner.bound_at = self._future_bound(newest)
                winner.landing_resolved = True
            for loser in merged:
                if loser is winner:
                    continue
                loser.state = "merged"
                loser.share_search = False
                loser.landing_resolved = True
                events.append(
                    {
                        "chain_id": loser.chain_id,
                        "kind": "merged",
                        "time": now,
                        "landing_at": landing,
                        "carried_by": winner.chain_id,
                        "carried_type": winner.trade_type,
                    }
                )
        # A cut-short search waits only for its successor to land. If that
        # successor resolves at a different bar, the earlier search is done.
        for chain in self.chains:
            if not chain.share_search or chain.landing_resolved or not chain.cut_short_by_arm_time:
                continue
            successor = next(
                (
                    other
                    for other in self.chains
                    if other.wave == chain.wave and other.arm_time == chain.cut_short_by_arm_time
                ),
                None,
            )
            if successor is None or not successor.landing_resolved:
                continue
            if successor.landing_at == chain.landing_at:
                continue
            chain.landing_resolved = True
            event = chain.abandon_shared_search(now)
            if event is not None:
                events.append(event)
        return events

    def step(
        self,
        ctx: dict[str, Any],
        arms: list[dict[str, Any]],
        *,
        switched_from: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[Trigger]]:
        now = str(ctx.get("time") or "")
        events: list[dict[str, Any]] = []
        triggers: list[Trigger] = []
        if switched_from is not None:
            events.extend(self.mark_old_side(switched_from, now))
        events.extend(self._add_arms(arms, ctx))

        type3_arms = {
            (arm.get("wave"), arm.get("arm_time"))
            for arm in arms
            if arm.get("allowed") and str(arm.get("type")) == "3"
        }
        for chain in list(self.chains):
            if (chain.wave, chain.arm_time) in type3_arms:
                continue
            events.extend(chain.prepare(ctx))
            if chain.trade_type == "3" and chain.landing_at is not None and chain.state == "arming_5m" and not chain.landing_resolved:
                future_bound = self._future_bound(chain)
                if future_bound is not None and (chain.bound_at is None or _utc(future_bound) < _utc(chain.bound_at)):
                    chain.bound_at = future_bound

        events.extend(self._merge_landings(now))
        for chain in list(self.chains):
            if chain.state == "merged" or chain.share_search:
                continue
            new_events, trigger = chain.advance_lower(ctx)
            events.extend(new_events)
            if trigger is not None:
                triggers.append(trigger)

        if len(triggers) > 1 and any(t.metadata.get("retry_policy") == "3S" for t in triggers):
            # IDs and magic are derived from the trigger close. Resolve before geometry.
            winner = min(triggers, key=lambda t: (t.retry, t.metadata.get("retry_policy") != "3S", _utc(t.arm_time), t.metadata["chain_id"]))
            loser_ids = {t.metadata["chain_id"] for t in triggers if t is not winner}
            events = [e for e in events if not (e["kind"] == "triggered" and e["chain_id"] in loser_ids)]
            for chain in self.chains:
                if chain.chain_id in loser_ids:
                    chain.state = "retry_lapsed"
                    events.append(chain._event("retry_lapsed", now, reason="another trigger on this close"))
            triggers = [winner]

        self.chains = [
            chain
            for chain in self.chains
            if chain.state not in Chain.TERMINAL or chain.share_search
        ]
        return events, triggers

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "hypothesis_id": self.hypothesis_id,
            "settings": deepcopy(self.settings),
            "chains": [chain.state_dict() for chain in self.chains],
            "allowed_arm_times": deepcopy(self.allowed_arm_times),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported ChainManager state version")
        saved_hypothesis_id = state.get("hypothesis_id")
        if self.hypothesis_id is not None and saved_hypothesis_id not in {None, self.hypothesis_id}:
            raise ValueError("ChainManager state hypothesis id does not match")
        if self.hypothesis_id is None and saved_hypothesis_id is not None:
            self.hypothesis_id = str(saved_hypothesis_id)
        self.settings = deepcopy(state.get("settings") or self.settings)
        self.chains = [Chain.from_state(item) for item in state.get("chains") or []]
        allowed = deepcopy(state.get("allowed_arm_times") or {"top": [], "bottom": []})
        if set(allowed) != {"top", "bottom"}:
            raise ValueError("ChainManager state must include top and bottom arm times")
        self.allowed_arm_times = allowed

    @classmethod
    def from_state(
        cls, state: dict[str, Any], *, hypothesis_id: str | None = None
    ) -> ChainManager:
        manager = cls(
            settings=state.get("settings") or {},
            hypothesis_id=hypothesis_id or state.get("hypothesis_id"),
        )
        manager.load_state(state)
        return manager
