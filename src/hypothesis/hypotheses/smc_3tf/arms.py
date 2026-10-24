from __future__ import annotations

from copy import deepcopy
from typing import Any


def classify_arm(wave: str, close: float, ema200: float, lower: float, upper: float) -> str:
    if wave == "top":
        if close > upper:
            return "1"
        if close >= ema200:
            return "2a"
        if close >= lower:
            return "2b"
        return "3"
    if wave == "bottom":
        if close < lower:
            return "1"
        if close <= ema200:
            return "2a"
        if close <= upper:
            return "2b"
        return "3"
    raise ValueError("wave must be 'top' or 'bottom'")


class ArmDetector:
    """Spend one 1H provisional arm budget per newly confirmed LTR pivot."""

    STATE_VERSION = 1

    def __init__(self, hypothesis_id: str) -> None:
        self.hypothesis_id = str(hypothesis_id)
        self.budgets: dict[str, dict[str, Any] | None] = {"top": None, "bottom": None}
        self.last_hour_known_at: str | None = None
        self._seen_confirmations: set[str] = set()

    @staticmethod
    def _pivot_key(event: dict[str, Any]) -> str:
        if event.get("event_id") is not None:
            return str(event["event_id"])
        return "|".join(str(event.get(key, "")) for key in ("side", "pivot_ts", "event_ts", "price"))

    @staticmethod
    def _type_for(hourly: dict[str, Any], wave: str) -> str | None:
        ta = hourly.get("ta") or {}
        ema200 = (ta.get("ema") or {}).get("200")
        band = ta.get("band") or {}
        lower, upper = band.get("lower"), band.get("upper")
        close = hourly.get("close")
        if any(value is None for value in (ema200, lower, upper, close)):
            return None
        return classify_arm(wave, float(close), float(ema200), float(lower), float(upper))

    def step(self, ctx: dict[str, Any], arming_side: str | None) -> list[dict[str, Any]]:
        hourly = ctx.get("1h")
        if not isinstance(hourly, dict) or not hourly.get("known_at"):
            return []
        known_at = str(hourly["known_at"])
        if known_at == self.last_hour_known_at:
            return []

        confirmations: dict[str, dict[str, Any]] = {}
        for event in hourly.get("pivots") or []:
            if not isinstance(event, dict) or str(event.get("tier", "")).lower() != "ltr":
                continue
            side = event.get("side")
            if side not in {"low", "high"}:
                continue
            key = self._pivot_key(event)
            if key in self._seen_confirmations:
                continue
            self._seen_confirmations.add(key)
            wave = "top" if side == "low" else "bottom"
            confirmations[wave] = {
                "side": side,
                "pivot_ts": event.get("pivot_ts"),
                "event_ts": event.get("event_ts"),
                "price": float(event["price"]) if event.get("price") is not None else None,
                "event_id": event.get("event_id"),
                "used": False,
            }
        self.budgets.update(confirmations)

        flow = (hourly.get("orderflow_by_tier") or {}).get("itr") or {}
        raw: list[dict[str, Any]] = []
        if str(flow.get("status", "")).lower() == "provisional":
            for wave in ("top", "bottom"):
                budget = self.budgets.get(wave)
                if budget is None or budget.get("used"):
                    continue
                budget["used"] = True
                trade_type = self._type_for(hourly, wave)
                allowed = arming_side == wave
                arm = {
                    "hypothesis": self.hypothesis_id,
                    "wave": wave,
                    "side": "short" if wave == "top" else "long",
                    "type": trade_type,
                    "arm_time": known_at,
                    "available_at": str(ctx.get("time") or known_at),
                    "close": float(hourly["close"]),
                    "high": float(hourly["high"]),
                    "low": float(hourly["low"]),
                    "pivot_confirmation": deepcopy(budget),
                    "allowed": bool(allowed and trade_type is not None),
                    "blocked_reason": (
                        "gate" if not allowed else "type_unavailable" if trade_type is None else None
                    ),
                }
                raw.append(arm)
        self.last_hour_known_at = known_at
        return raw

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "hypothesis_id": self.hypothesis_id,
            "budgets": deepcopy(self.budgets),
            "last_hour_known_at": self.last_hour_known_at,
            "seen_confirmations": sorted(self._seen_confirmations),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported ArmDetector state version")
        if str(state.get("hypothesis_id", self.hypothesis_id)) != self.hypothesis_id:
            raise ValueError("ArmDetector state hypothesis id does not match")
        budgets = deepcopy(state.get("budgets") or {"top": None, "bottom": None})
        if set(budgets) != {"top", "bottom"}:
            raise ValueError("ArmDetector state must include top and bottom budgets")
        self.budgets = budgets
        self.last_hour_known_at = state.get("last_hour_known_at")
        self._seen_confirmations = set(map(str, state.get("seen_confirmations") or []))

    @classmethod
    def from_state(cls, state: dict[str, Any], *, hypothesis_id: str) -> ArmDetector:
        detector = cls(hypothesis_id)
        detector.load_state(deepcopy(state))
        return detector
