from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

import pandas as pd

from .base import Hypothesis, Trigger
from .geometry import geometry
from .hypotheses import HYPOTHESIS_REGISTRY
from .shadow import ShadowInventory
from .waves import WaveTracker


class Layer4:
    """Per-asset Layer 4 runtime over one ordered stream of Fusion contexts."""

    STATE_VERSION = 3

    def __init__(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError("Layer4 config must be a mapping")
        self.config = deepcopy(config)
        replay_config = config.get("replay") or {}
        self.symbol = str(
            config.get("symbol") or config.get("asset") or replay_config.get("symbol") or ""
        ).strip().upper()
        assets = config.get("symbols") or {}
        asset_config = deepcopy(assets.get(self.symbol) or config.get("asset_config") or {})
        self.symbol_code = str(config.get("symbol_code") or asset_config.get("code") or "")
        self.pip_size = float(config.get("pip_size") or asset_config.get("pip_size") or 0.0001)
        self.sl_floor_pips = float(config.get("sl_floor_pips") or asset_config.get("sl_floor_pips") or 13.0)

        self.sl_cap_pips = asset_config.get("sl_cap_pips")
        if self.sl_cap_pips is not None:
            if isinstance(self.sl_cap_pips, bool) or not isinstance(self.sl_cap_pips, (int, float)):
                raise ValueError("Layer4 sl_cap_pips must be positive and finite")
            try:
                self.sl_cap_pips = float(self.sl_cap_pips)
            except OverflowError as exc:
                raise ValueError("Layer4 sl_cap_pips must be positive and finite") from exc
            if not isfinite(self.sl_cap_pips) or self.sl_cap_pips <= 0:
                raise ValueError("Layer4 sl_cap_pips must be positive and finite")
            if self.sl_cap_pips < self.sl_floor_pips:
                raise ValueError("Layer4 sl_cap_pips must be at least sl_floor_pips")

        layer4 = deepcopy(config.get("layer4") or {})
        self.enabled = [str(value) for value in layer4.get("hypotheses", ["SMC-3TF"])]
        self.shared_settings = {
            "min_gap_pips": float(layer4.get("min_gap_pips", 2.0)),
            "tp_r": float(layer4.get("tp_r", 3.0)),
            "old_side_expiry_days": float(layer4.get("old_side_expiry_days", 5.0)),
        }
        self.retry_r = float(layer4.get("retry_r", 2.0))
        if self.pip_size <= 0 or self.sl_floor_pips <= 0:
            raise ValueError("Layer4 pip_size and sl_floor_pips must be positive")
        if not isfinite(self.retry_r) or self.retry_r <= 0:
            raise ValueError("Layer4 retry_r must be positive and finite")
        if self.shared_settings["min_gap_pips"] < 0:
            raise ValueError("Layer4 min_gap_pips cannot be negative")
        if self.shared_settings["tp_r"] <= 0 or self.shared_settings["old_side_expiry_days"] <= 0:
            raise ValueError("Layer4 tp_r and old_side_expiry_days must be positive")
        if len(set(self.enabled)) != len(self.enabled):
            raise ValueError("Layer4 hypotheses must not contain duplicates")

        hypotheses: dict[str, Hypothesis] = {}
        codes: dict[int, str] = {}
        for identifier in self.enabled:
            hypothesis_class = HYPOTHESIS_REGISTRY.get(identifier)
            if hypothesis_class is None:
                raise ValueError(f"Unknown Layer4 hypothesis: {identifier}")
            hypothesis_settings = deepcopy(layer4.get(identifier) or {})
            code = int(hypothesis_settings.get("code", 1))
            if not 0 <= code <= 9:
                raise ValueError(f"Layer4 hypothesis code must be one digit: {identifier}={code}")
            if code in codes:
                raise ValueError(
                    f"Layer4 hypotheses {codes[code]} and {identifier} share code {code}"
                )
            codes[code] = identifier
            instance = hypothesis_class(settings=hypothesis_settings, code=code)
            if str(getattr(instance, "id", "")) != identifier:
                raise ValueError(
                    f"Layer4 registry id {identifier} does not match implementation id "
                    f"{getattr(instance, 'id', None)}"
                )
            hypotheses[identifier] = instance
        self.hypotheses = hypotheses
        self.waves = WaveTracker(pip_size=self.pip_size)
        self.shadow = ShadowInventory(asset=self.symbol or str(config.get("symbol") or "UNKNOWN"), retry_r=self.retry_r)
        self.last_time: str | None = None

    def open_retry(self, hypothesis: str, record: dict[str, Any]) -> None:
        """Open a hypothesis-specific retry after the shadow inventory reports an SL."""
        instance = self.hypotheses.get(str(hypothesis))
        if instance is None:
            raise KeyError(f"Layer4 hypothesis is not enabled: {hypothesis}")
        instance.open_retry(record)

    def step(self, ctx: dict[str, Any], *, catch_up: bool = False) -> dict[str, Any]:
        if not isinstance(ctx, dict) or not ctx.get("time"):
            raise ValueError("Layer4.step requires a context_at mapping with a close time")
        now = str(ctx["time"])
        if self.last_time is not None and now <= self.last_time:
            raise ValueError(f"Layer4 contexts must be strictly increasing: {now} <= {self.last_time}")

        enriched = deepcopy(ctx)
        enriched.setdefault("symbol_code", self.symbol_code or None)
        enriched.setdefault("pip_size", self.pip_size)
        waves = self.waves.step(enriched)
        core_view = {"waves": waves, "current_waves": self.waves.current()}
        by_hypothesis: dict[str, Any] = {}
        triggers: list[Trigger] = []
        orders: list[dict[str, Any]] = []
        order_updates: list[dict[str, Any]] = []
        for identifier in self.enabled:
            hypothesis = self.hypotheses[identifier]
            new_triggers = hypothesis.step(enriched, core_view)
            triggers.extend(new_triggers)
            step_view = deepcopy(getattr(hypothesis, "last_step", {}))
            by_hypothesis[identifier] = step_view
            order_updates.extend(deepcopy(step_view.get("order_updates") or []))
            code = int(getattr(hypothesis, "code"))
            settings = (
                (self.config.get("layer4") or {}).get(identifier) or {}
            )
            for trigger in new_triggers:
                order = geometry(
                        trigger,
                        enriched,
                        hypothesis_code=code,
                        min_gap_pips=self.shared_settings["min_gap_pips"],
                        sl_floor_pips=self.sl_floor_pips,
                        sl_cap_pips=self.sl_cap_pips,
                        tp_r=self.shared_settings["tp_r"],
                        retry_3s=settings.get("retry_3s"),
                        old_side_expiry_days=float(
                            settings.get("old_side_expiry_days", self.shared_settings["old_side_expiry_days"])
                        ),
                    )
                trigger_close = pd.Timestamp(order["trigger_close"])
                trigger_close = trigger_close.tz_localize("UTC") if trigger_close.tzinfo is None else trigger_close.tz_convert("UTC")
                order["record_id"] = f"{self.shadow.asset}:{int(order['magic'])}:{trigger_close.isoformat()}"
                orders.append(order)
                register = getattr(hypothesis, "register_order", None)
                if callable(register):
                    register({**order, "asset": self.symbol})

        shadow_result = self.shadow.step(
            enriched,
            new_orders=orders,
            order_updates=order_updates,
        )
        shadow_events = list(shadow_result["events"])
        for record in shadow_result["retry_requests"]:
            hypothesis_id = str(record.get("hypothesis") or "")
            hypothesis = self.hypotheses.get(hypothesis_id)
            if hypothesis is None:
                raise KeyError(f"Layer4 retry owner is not enabled: {hypothesis_id}")
            retry_settings = (self.config.get("layer4") or {}).get(hypothesis_id) or {}
            retry_policy = retry_settings.get("retry_policy", {key: "5x9" for key in ("1", "2a", "2b", "3")})
            if str(record.get("type")) not in retry_policy:
                shadow_events.append(
                    {
                        "kind": "retry_skipped",
                        "time": now,
                        "record_id": record.get("record_id"),
                        "magic": record.get("magic"),
                        "reason": "type has no retry_policy",
                        "record": deepcopy(record),
                    }
                )
                continue
            accepted = bool(hypothesis.open_retry(deepcopy(record)))
            completed = self.shadow.finish_retry(
                str(record["record_id"]),
                attempted=True,
                accepted=accepted,
                record_snapshot=record,
            )
            if completed is not None:
                shadow_events.append(completed)

        for chain_id in shadow_result["released_chain_ids"]:
            for hypothesis in self.hypotheses.values():
                release = getattr(hypothesis, "release_order", None)
                if callable(release):
                    release(chain_id)

        self.last_time = now
        return {
            "time": now,
            "asset": str(enriched.get("asset") or enriched.get("symbol") or self.symbol),
            "catch_up": bool(catch_up),
            "waves": waves,
            "current_waves": deepcopy(self.waves.current()),
            "hypotheses": by_hypothesis,
            "triggers": [trigger.to_dict() for trigger in triggers],
            "orders": orders,
            "order_updates": order_updates,
            "shadow_events": shadow_events,
        }

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "symbol": self.symbol,
            "symbol_code": self.symbol_code,
            "pip_size": self.pip_size,
            "sl_floor_pips": self.sl_floor_pips,
            "sl_cap_pips": self.sl_cap_pips,
            "enabled": list(self.enabled),
            "last_time": self.last_time,
            "waves": self.waves.state(),
            "hypotheses": {name: hypothesis.state() for name, hypothesis in self.hypotheses.items()},
            "shadow": self.shadow.state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported Layer4 state version")
        if str(state.get("symbol", self.symbol)).upper() != self.symbol:
            raise ValueError("Layer4 state symbol does not match this runtime")
        if list(state.get("enabled", self.enabled)) != self.enabled:
            raise ValueError("Layer4 enabled hypothesis list does not match this runtime")
        self.last_time = state.get("last_time")
        self.waves = WaveTracker.from_state(state.get("waves") or {})
        saved_hypotheses = state.get("hypotheses") or {}
        if set(saved_hypotheses) != set(self.hypotheses):
            raise ValueError("Layer4 state hypotheses do not match this runtime")
        for name, hypothesis_state in saved_hypotheses.items():
            self.hypotheses[name].load_state(hypothesis_state)
        shadow_state = state.get("shadow")
        if shadow_state is None:
            raise ValueError("Layer4 state does not contain its shadow inventory")
        self.shadow = ShadowInventory.from_state(shadow_state, asset=self.symbol)

    @classmethod
    def from_state(cls, config: dict[str, Any], state: dict[str, Any]) -> Layer4:
        instance = cls(config)
        instance.load_state(deepcopy(state))
        return instance
