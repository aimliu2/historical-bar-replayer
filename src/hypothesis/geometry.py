from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

from .base import Trigger


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


def geometry(
    trigger: Trigger,
    ctx: dict[str, Any],
    *,
    hypothesis_code: int,
    min_gap_pips: float,
    sl_floor_pips: float,
    tp_r: float,
    old_side_expiry_days: float = 5.0,
    sl_cap_pips: float | None = None,
    retry_3s: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the decided limit, SL, TP, magic and comment for a trigger."""
    code = str(ctx.get("symbol_code") or "")
    if len(code) != 3 or not code.isdigit():
        raise ValueError("Layer 4 geometry requires a three-digit symbol_code")
    if isinstance(hypothesis_code, bool) or not 0 <= int(hypothesis_code) <= 9:
        raise ValueError("hypothesis_code must fit one decimal digit")
    pip_size = _positive(ctx.get("pip_size"), "pip_size")
    is_3s = trigger.retry and trigger.metadata.get("retry_policy") == "3S"
    settings_3s = retry_3s or {}
    if is_3s:
        min_gap_pips = settings_3s.get("spread_pips", 2.0)
        sl_floor_pips = settings_3s.get("sl_floor_pips", sl_floor_pips)
        sl_cap_pips = settings_3s.get("sl_cap_pips", 25.0)
    gap_pips = _nonnegative(min_gap_pips, "min_gap_pips")
    floor_pips = _positive(sl_floor_pips, "sl_floor_pips")
    cap_pips = None if sl_cap_pips is None else _positive(sl_cap_pips, "sl_cap_pips")
    if cap_pips is not None and cap_pips < floor_pips:
        raise ValueError("sl_cap_pips must be at least sl_floor_pips")
    reward_r = _positive(tp_r, "tp_r")
    close = float(ctx["bar_5m"]["close"])
    entry_level = float(trigger.entry_level)
    if not isfinite(close) or not isfinite(entry_level):
        raise ValueError("Trigger close and entry level must be finite")

    if is_3s:
        entry_level += gap_pips * pip_size * (-1 if trigger.side == "short" else 1)

    if trigger.side == "short":
        entry = max(entry_level, close + gap_pips * pip_size)
    elif trigger.side == "long":
        entry = min(entry_level, close - gap_pips * pip_size)
    else:
        raise ValueError("Trigger side must be 'long' or 'short'")

    hourly_ta = (ctx.get("1h") or {}).get("ta") or {}
    atr_value = ((hourly_ta.get("atr") or {}).get("14"))
    atr_distance = _positive(atr_value, "1H ATR14")
    floor_distance = floor_pips * pip_size
    s2_distance = 0.0
    if trigger.sl_reference is not None:
        reference = float(trigger.sl_reference)
        if not isfinite(reference):
            raise ValueError("sl_reference must be finite when supplied")
        if trigger.side == "short" and reference > entry:
            s2_distance = reference - entry
        elif trigger.side == "long" and reference < entry:
            s2_distance = entry - reference

    distances = [
        ("S2", s2_distance),
        ("ATR14", atr_distance),
        ("floor", floor_distance),
    ]
    sl_source, sl_distance = max(distances, key=lambda item: item[1])
    if cap_pips is not None and sl_distance > cap_pips * pip_size:
        sl_source, sl_distance = "cap", cap_pips * pip_size
    if trigger.side == "short":
        stop_loss = entry + sl_distance
        take_profit = entry - reward_r * sl_distance
    else:
        stop_loss = entry - sl_distance
        take_profit = entry + reward_r * sl_distance

    epoch = int(pd.Timestamp(trigger.trigger_close).timestamp())
    magic_digits = f"{epoch % 100_000_000:08d}"
    magic_text = f"{code}{int(hypothesis_code)}{magic_digits}"
    magic = int(magic_text)
    comment = trigger.metadata["retry_policy"] if trigger.retry and trigger.metadata.get("retry_policy") in {"3T", "3S"} else f"{trigger.trade_type}{'Re' if trigger.retry else ''}"
    expires_at = trigger.expires_at
    if trigger.metadata.get("old_side"):
        expires_at = (pd.Timestamp(trigger.trigger_close) + pd.Timedelta(days=float(old_side_expiry_days))).isoformat()
    return {
        **trigger.to_dict(),
        "expires_at": expires_at,
        "entry": float(entry),
        "limit": float(entry),
        "limit_source": "ichoch level" if entry == entry_level else "close + gap",
        "sl": float(stop_loss),
        "sl_pips": float(sl_distance / pip_size),
        "sl_source": sl_source,
        "s2_distance": float(s2_distance),
        "atr14": float(atr_distance),
        "tp": float(take_profit),
        "tp_r": reward_r,
        "magic": magic,
        "magic_text": magic_text,
        "comment": comment,
    }
