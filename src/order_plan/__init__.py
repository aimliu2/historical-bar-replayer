from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

def _finite_number(value: Any, key: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Order plan setting '{key}' must be a finite number.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"Order plan setting '{key}' must be a finite number.") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"Order plan setting '{key}' must be a {qualifier} finite number.")
    return number


def order_plan_settings(raw: Any) -> dict[str, float]:
    """Validate and normalize the Layer 5 management settings."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Order plan setting 'order_plan' must be a mapping.")
    unknown = sorted(set(raw) - {"partial_r", "be_plus_pips"})
    if unknown:
        raise ValueError(f"Unknown order plan setting 'order_plan.{unknown[0]}'.")
    return {
        "partial_r": _finite_number(raw.get("partial_r", 2.0), "order_plan.partial_r", positive=True),
        "be_plus_pips": _finite_number(raw.get("be_plus_pips", 3.0), "order_plan.be_plus_pips", positive=False),
    }


class OrderPlan:
    """Stateless Layer 5 management plans for every published Layer 4 order."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.settings = order_plan_settings(config.get("order_plan"))

    def step(self, layer4_output: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "record_id": order["record_id"],
                "magic": order["magic"],
                "hypothesis": order["hypothesis"],
                "comment": order["comment"],
                "plan": {**deepcopy(self.settings), "partial_fraction": 0.5},
            }
            for order in layer4_output.get("orders", [])
        ]
