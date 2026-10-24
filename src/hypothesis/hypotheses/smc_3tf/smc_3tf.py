from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ...base import Hypothesis, Trigger
from .arms import ArmDetector
from .chain import ChainManager, wave_band_exit
from .gate import ArmingSide


def _ema_gap(hourly: dict[str, Any], wave: str) -> float | None:
    ema = ((hourly.get("ta") or {}).get("ema") or {})
    ema5, ema9 = ema.get("5"), ema.get("9")
    if ema5 is None or ema9 is None:
        return None
    return float(ema5) - float(ema9) if wave == "top" else float(ema9) - float(ema5)


def _record_time(record: dict[str, Any]) -> str | None:
    for key in ("sl_closed_at", "sl_time", "closed_at", "exit_time"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


class SMC3TF(Hypothesis):
    """The 1D-gated 1H→5m SMC trade hypothesis."""

    STATE_VERSION = 4
    id = "SMC-3TF"

    def __init__(self, *, settings: dict[str, Any] | None = None, code: int = 1) -> None:
        self.settings = deepcopy(settings or {})
        self.code = int(code)
        self.gate = ArmingSide()
        self.arms = ArmDetector(self.id)
        self.chains = ChainManager(settings=self.settings, hypothesis_id=self.id)
        self.pending_retries: list[dict[str, Any]] = []
        self.live_orders: list[dict[str, Any]] = []
        self.last_step: dict[str, Any] = {}
        self.retry_hour_extreme: dict[str, Any] = {}

    def open_retry(self, record: dict[str, Any]) -> bool:
        """Queue a retry watcher from an eligible shadow SL record.

        The watcher uses the configured policy after the SL close.
        Batch 2 supplies eligibility from the shadow inventory; the defensive
        checks here keep direct callers from opening an exhausted retry.
        """
        if not isinstance(record, dict):
            raise TypeError("retry record must be a mapping")
        trade_type = str(record.get("type", record.get("trade_type", "")))
        wave = str(record.get("wave", ""))
        closed_at = _record_time(record)
        retry_policy = self.settings.get("retry_policy", {key: "5x9" for key in ("1", "2a", "2b", "3")})
        if wave not in {"top", "bottom"}:
            raise ValueError("retry record wave must be 'top' or 'bottom'")
        if trade_type not in {"1", "2a", "2b", "3"}:
            raise ValueError("retry record type must be 1, 2a, 2b or 3")
        if closed_at is None:
            raise ValueError("retry record must include the SL close time")
        if trade_type not in retry_policy or bool(record.get("retry")) or int(record.get("retry_budget", 1)) != 1:
            return False
        if record.get("reached_retry") is not None:
            return False
        record_id = str(record.get("record_id", record.get("magic", record.get("id", closed_at))))
        if any(str(item["record_id"]) == record_id for item in self.pending_retries):
            return False
        if retry_policy[trade_type] == "3S" and (
            self.retry_hour_extreme.get("bar_closed_at") != closed_at
            or self.retry_hour_extreme.get("bar_high") is None
            or self.retry_hour_extreme.get("bar_low") is None
        ):
            raise ValueError("3S requires the observed 5m SL bar at the record close")
        self.pending_retries.append(
            {
                "record_id": record_id,
                "policy": retry_policy[trade_type],
                "record": deepcopy(record),
                "wave": wave,
                "type": trade_type,
                "sl_closed_at": closed_at,
                **({"high": self.retry_hour_extreme["bar_high"], "low": self.retry_hour_extreme["bar_low"]}
                   if retry_policy[trade_type] == "3S" else {}),
                "last_hour_seen": self.retry_hour_extreme.get("known_at") if retry_policy[trade_type] == "3T" else None,
                "last_gap": (
                    self.retry_hour_extreme.get("ema_gap_top")
                    if wave == "top" else -self.retry_hour_extreme["ema_gap_top"]
                ) if retry_policy[trade_type] == "3T" and self.retry_hour_extreme.get("ema_gap_top") is not None else None,
                **({"ema200_breached_at": None, "ema5x9_touched_at": None} if retry_policy[trade_type] == "3T" else {}),
            }
        )
        return True

    def register_order(self, order: dict[str, Any]) -> None:
        metadata = order.get("metadata") or {}
        chain_id = metadata.get("chain_id")
        if chain_id is None:
            raise ValueError("SMC-3TF order is missing its chain_id")
        key = str(chain_id)
        item = next((value for value in self.live_orders if value["chain_id"] == key), None)
        if item is None:
            item = {
                "chain_id": key,
                "wave": str(order["wave"]),
                "trigger_close": str(order["trigger_close"]),
                "type": str(order["type"]),
                "retry": bool(order.get("retry", False)),
                "retry_policy": metadata.get("retry_policy"),
                "record_id": None,
                "magic": None,
            }
            self.live_orders.append(item)
        item["record_id"] = str(order["record_id"])
        item["magic"] = int(order["magic"])

    def release_order(self, chain_id: str) -> None:
        key = str(chain_id)
        self.live_orders = [item for item in self.live_orders if str(item.get("chain_id")) != key]

    def _order_updates(
        self,
        ctx: dict[str, Any],
        arms: list[dict[str, Any]],
        *,
        switched_from: str | None,
        prior_live_orders: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not prior_live_orders:
            return []
        now = str(ctx.get("time") or "")
        days = float(self.settings.get("old_side_expiry_days", 5.0))
        allowed_arms = [arm for arm in arms if arm.get("allowed")]
        updates: list[dict[str, Any]] = []
        for order in prior_live_orders:
            chain_id = str(order["chain_id"])
            if order.get("retry_policy") == "3S" and wave_band_exit(ctx, str(order["wave"])):
                updates.append({"chain_id": chain_id, "expires_at": now, "reason": "band exit"})
            if switched_from is not None and str(order.get("wave")) == switched_from:
                deadline = (pd_timestamp(order["trigger_close"]) + pd.Timedelta(days=days)).isoformat()
                updates.append(
                    {
                        "chain_id": chain_id,
                        "expires_at": deadline,
                        "retry_budget": 0,
                        "reason": "1D gate switch",
                    }
                )

            later_arms = [
                arm
                for arm in allowed_arms
                if str(arm.get("wave")) == str(order.get("wave"))
                and pd_timestamp(arm["arm_time"]) > pd_timestamp(order["trigger_close"])
            ]
            if later_arms:
                bound = min((str(arm["arm_time"]) for arm in later_arms), key=pd_timestamp)
                updates.append(
                    {
                        "chain_id": chain_id,
                        "expires_at": bound,
                        "reason": "next allowed same-side 1H arm",
                    }
                )
        return updates

    @staticmethod
    def _retry_5x9(watcher: dict[str, Any], gap: float, previous_gap: float | None,
                   gap_threshold: float) -> bool:
        touched = previous_gap is not None and float(previous_gap) >= gap_threshold and float(gap) < gap_threshold
        crossed = previous_gap is not None and float(previous_gap) > 0.0 and float(gap) <= 0.0
        return touched or crossed

    def _update_retry_hour_extreme(self, ctx: dict[str, Any]) -> None:
        # Track each closed 5m bar even before an SL opens its watcher. The
        # re-arm hour can start before the SL; S2T needs those earlier bars too.
        now = ctx.get("time")
        if not now:
            return
        hour = pd_timestamp(now).ceil("1h").isoformat()
        if self.retry_hour_extreme.get("hour") != hour:
            self.retry_hour_extreme = {"hour": hour, "high": None, "low": None}
        hourly = ctx.get("1h") or {}
        self.retry_hour_extreme["known_at"] = hourly.get("known_at")
        self.retry_hour_extreme["ema_gap_top"] = _ema_gap(hourly, "top")
        bar = ctx.get("bar_5m") or {}
        self.retry_hour_extreme.update(bar_closed_at=str(ctx["time"]), bar_high=bar.get("high"), bar_low=bar.get("low"))
        for name, reducer in (("high", max), ("low", min)):
            value = bar.get(name)
            if value is not None:
                prior = self.retry_hour_extreme[name]
                self.retry_hour_extreme[name] = float(value) if prior is None else reducer(prior, float(value))

    @staticmethod
    def _retry_3t(watcher: dict[str, Any], ctx: dict[str, Any], touched: bool,
                  hour_time: str) -> bool:
        now = str(ctx["time"])
        if pd_timestamp(now) <= pd_timestamp(watcher["sl_closed_at"]):
            return False
        if touched and watcher["ema5x9_touched_at"] is None:
            watcher["ema5x9_touched_at"] = hour_time
        ema200 = (((ctx.get("1h") or {}).get("ta") or {}).get("ema") or {}).get("200")
        close = (ctx.get("bar_5m") or {}).get("close")
        if ema200 is not None and close is not None and watcher["ema200_breached_at"] is None:
            breached = float(close) > float(ema200) if watcher["wave"] == "top" else float(close) < float(ema200)
            if breached:
                watcher["ema200_breached_at"] = now
        return watcher["ema200_breached_at"] is not None and watcher["ema5x9_touched_at"] is not None

    def _retry_arms(
        self,
        ctx: dict[str, Any],
        regular_arms: list[dict[str, Any]],
        arming_side: str | None,
        *,
        switched_from: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        hourly = ctx.get("1h") or {}
        hour_time = hourly.get("known_at")
        if not hour_time:
            return [], []
        now = str(ctx.get("time") or hour_time)
        self._update_retry_hour_extreme(ctx)
        gap_threshold = float(self.settings.get("near_cross_pips", 1.0)) * float(ctx.get("pip_size") or 0.0001)
        available: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        reentry_keys: set[tuple[str, str]] = set()

        for watcher in self.pending_retries:
            wave, policy = watcher["wave"], watcher["policy"]
            if switched_from == wave:
                events.append({"kind": "retry_cancelled", "time": ctx.get("time"), "record_id": watcher["record_id"], "reason": "1D gate switch"})
                continue
            new_hour = str(hour_time) != str(watcher.get("last_hour_seen"))
            if policy == "5x9" and not new_hour:
                remaining.append(watcher)
                continue

            touched = False
            if new_hour:
                gap = _ema_gap(hourly, wave)
                previous_gap = watcher.get("last_gap")
                watcher["last_hour_seen"] = str(hour_time)
                watcher["last_gap"] = gap
                # Preserve the 5x9 history update even for the latest closed
                # hour at/before the SL, but never count its touch as eligible.
                if gap is None and policy == "5x9":
                    remaining.append(watcher)
                    continue
                if gap is not None and pd_timestamp(hour_time) > pd_timestamp(watcher["sl_closed_at"]):
                    touched = self._retry_5x9(watcher, gap, previous_gap, gap_threshold)

            same_bar_arm = next((arm for arm in regular_arms if arm.get("allowed")
                                 and arm.get("wave") == wave
                                 and str(arm.get("arm_time")) == str(hour_time)), None)
            if same_bar_arm is not None:
                events.append({"kind": "retry_lapsed", "time": str(hour_time), "record_id": watcher["record_id"], "reason": "regular same-side arm on re-entry close"})
                continue

            ready = (pd_timestamp(now) > pd_timestamp(watcher["sl_closed_at"]) if policy == "3S"
                     else touched if policy == "5x9" else self._retry_3t(watcher, ctx, touched, str(hour_time)))
            if not ready:
                remaining.append(watcher)
                continue
            rearm_time = str(hour_time) if policy == "5x9" else watcher["sl_closed_at"] if policy == "3S" else now
            if arming_side != wave:
                events.append({"kind": "retry_blocked", "time": rearm_time, "record_id": watcher["record_id"], "reason": "1D gate"})
                continue
            reentry_key = (wave, now)
            if reentry_key in reentry_keys:
                events.append({"kind": "retry_lapsed", "time": rearm_time, "record_id": watcher["record_id"],
                               "reason": "another retry already re-entered on this 1H close"})
                continue
            reentry_keys.add(reentry_key)

            side = "short" if wave == "top" else "long"
            extreme = watcher if policy == "3S" else self.retry_hour_extreme if policy == "3T" else hourly
            arm = {
                "hypothesis": self.id,
                "wave": wave,
                "side": side,
                "type": watcher["type"],
                "arm_time": rearm_time,
                "available_at": now,
                "close": float((ctx.get("bar_5m") or {})["close"] if policy in {"3T", "3S"} else hourly["close"]),
                "high": float(extreme["high"]),
                "low": float(extreme["low"]),
                "allowed": True,
                "retry": True,
                "chain_id": f"SMC-3TF:retry:{watcher['record_id']}:{rearm_time}",
                "retry_record_id": watcher["record_id"],
                "retry_policy": policy,
                **({"sl_closed_at": watcher["sl_closed_at"]} if policy == "3S" else {}),
                "pivot_confirmation": None,
                **({"rearm_hour": extreme["hour"],
                    "ema200_breached_at": watcher["ema200_breached_at"],
                    "ema5x9_touched_at": watcher["ema5x9_touched_at"]} if policy == "3T" else {}),
            }
            available.append(arm)
            events.append({"kind": "retry_reentry", "time": rearm_time, "record_id": watcher["record_id"], "wave": wave, "type": watcher["type"],
                           **({"retry_policy": policy} if policy == "3S" else {}),
                           **({"retry_policy": policy, "rearm_hour": arm["rearm_hour"],
                               "ema200_breached_at": watcher["ema200_breached_at"],
                               "ema5x9_touched_at": watcher["ema5x9_touched_at"]} if policy == "3T" else {})})

        self.pending_retries = remaining
        return available, events

    def step(self, ctx: dict[str, Any], core_view: dict[str, Any]) -> list[Trigger]:
        prior_live_orders = deepcopy(self.live_orders)
        gate_result = self.gate.step(ctx)
        switched_from = next(
            (event.get("from") for event in gate_result["events"] if event.get("kind") == "switched"),
            None,
        )
        raw_arms = self.arms.step(ctx, gate_result["arming_side"])
        retry_arms, retry_events = self._retry_arms(
            ctx,
            raw_arms,
            gate_result["arming_side"],
            switched_from=switched_from,
        )
        # Only regular allowed arms bound a pending order, as they bound chains;
        # a retry re-entry never cuts another order short.
        order_updates = self._order_updates(
            ctx,
            raw_arms,
            switched_from=switched_from,
            prior_live_orders=prior_live_orders,
        )
        chain_events, triggers = self.chains.step(
            ctx,
            [*raw_arms, *retry_arms],
            switched_from=switched_from,
        )
        for trigger in triggers:
            chain_id = str(trigger.metadata.get("chain_id") or "")
            if not chain_id:
                continue
            self.live_orders.append(
                {
                    "chain_id": chain_id,
                    "wave": trigger.wave,
                    "trigger_close": trigger.trigger_close,
                    "type": trigger.trade_type,
                    "retry": bool(trigger.retry),
                    "retry_policy": trigger.metadata.get("retry_policy"),
                    "record_id": None,
                    "magic": None,
                }
            )
        wave_output = core_view.get("waves")
        self.last_step = {
            "gate": gate_result,
            "arms": deepcopy(raw_arms),
            "retry_events": retry_events,
            "order_updates": order_updates,
            "chains": chain_events,
            "current_waves": deepcopy(core_view.get("current_waves") or {}),
            "wave_update": deepcopy(wave_output),
            "triggers": [trigger.to_dict() for trigger in triggers],
        }
        return triggers

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "id": self.id,
            "code": self.code,
            "settings": deepcopy(self.settings),
            "gate": self.gate.state(),
            "arms": self.arms.state(),
            "chains": self.chains.state(),
            "pending_retries": deepcopy(self.pending_retries),
            "live_orders": deepcopy(self.live_orders),
            "retry_hour_extreme": deepcopy(self.retry_hour_extreme),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported SMC3TF state version")
        if str(state.get("id", self.id)) != self.id:
            raise ValueError("SMC3TF state hypothesis id does not match")
        self.code = int(state.get("code", self.code))
        self.settings = deepcopy(state.get("settings") or self.settings)
        self.gate = ArmingSide.from_state(state.get("gate") or {})
        self.arms = ArmDetector.from_state(state.get("arms") or {}, hypothesis_id=self.id)
        self.chains = ChainManager.from_state(state.get("chains") or {}, hypothesis_id=self.id)
        self.pending_retries = deepcopy(state.get("pending_retries") or [])
        self.live_orders = deepcopy(state.get("live_orders") or [])
        self.retry_hour_extreme = deepcopy(state.get("retry_hour_extreme") or {})

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> SMC3TF:
        instance = cls(settings=state.get("settings") or {}, code=int(state.get("code", 1)))
        instance.load_state(deepcopy(state))
        return instance


def pd_timestamp(value: Any) -> Any:
    """Keep pandas out of the public hypothesis API while comparing UTC times."""
    import pandas as pd

    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
