from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd


_EVENT_FIELDS = ("pivots", "breaks", "wicks", "closed_levels")
_OHLC_FIELDS = ("open", "high", "low", "close")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("context_at time must be a valid timestamp")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _indexed_rows(rows: pd.DataFrame | None, name: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    if rows is None:
        empty = pd.DataFrame()
        return empty, pd.DatetimeIndex([], tz="UTC")
    if not isinstance(rows, pd.DataFrame):
        raise TypeError(f"{name} rows must be a pandas DataFrame or None")
    if rows.empty:
        return rows, pd.DatetimeIndex([], tz="UTC")

    if (
        isinstance(rows.index, pd.DatetimeIndex)
        and rows.index.tz is not None
        and str(rows.index.tz) == "UTC"
        and "known_at" not in rows.columns
    ):
        index = rows.index
    else:
        raw_index = rows["known_at"] if "known_at" in rows.columns else rows.index
        index = pd.DatetimeIndex(pd.to_datetime(raw_index, utc=True, errors="coerce"))
    if index.isna().any():
        raise ValueError(f"{name} rows contain an invalid known_at timestamp")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name} rows must have unique, increasing known_at values")
    return rows, index


def _plain(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return _utc(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value


def _row_payload(frame: pd.DataFrame, index: pd.DatetimeIndex, position: int) -> dict[str, Any]:
    row = frame.iloc[position]
    payload = {str(key): _plain(value) for key, value in row.items() if key != "known_at"}
    payload["known_at"] = index[position].isoformat()
    return payload


def _asof_position(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int | None:
    position = int(index.searchsorted(timestamp, side="right")) - 1
    return position if position >= 0 else None


def _exact_position(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int | None:
    position = int(index.searchsorted(timestamp, side="left"))
    return position if position < len(index) and index[position] == timestamp else None


def _events_at(
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    field: str,
) -> list[Any]:
    position = _exact_position(index, timestamp)
    if position is None or field not in frame.columns:
        return []
    value = frame.iloc[position][field]
    return deepcopy(value) if isinstance(value, list) else []


def _bar_payload(bar: Mapping[str, Any] | pd.Series, timestamp: pd.Timestamp) -> dict[str, Any]:
    if isinstance(bar, pd.Series):
        values = bar.to_dict()
    elif isinstance(bar, Mapping):
        values = dict(bar)
    else:
        raise TypeError("bar_5m must be a mapping or pandas Series")

    bar_time = values.get("known_at", values.get("time"))
    if bar_time is not None and _utc(bar_time) != timestamp:
        raise ValueError("bar_5m timestamp must equal the context_at time")
    missing = [field for field in _OHLC_FIELDS if field not in values]
    if missing:
        raise ValueError(f"bar_5m is missing OHLC field(s): {', '.join(missing)}")
    return {field: _plain(values[field]) for field in _OHLC_FIELDS}


def context_at(
    timestamp: Any,
    *,
    asset: str,
    bar_5m: Mapping[str, Any] | pd.Series,
    symbol_code: str | None = None,
    pip_size: float | None = None,
    daily_rows: pd.DataFrame | None = None,
    one_hour_rows: pd.DataFrame | None = None,
    five_minute_rows: pd.DataFrame | None = None,
    event_facts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project one asset's Layer 3 facts as known at a 5m close.

    Inputs are per-close fact rows indexed by ``known_at`` (or carrying a
    ``known_at`` column). States use the latest row at or before the requested
    close. Event lists come only from rows published at that exact close.
    This function is stateless and does not modify any input frame.
    """
    at = _utc(timestamp)
    daily, daily_index = _indexed_rows(daily_rows, "1D")
    hourly, hourly_index = _indexed_rows(one_hour_rows, "1H")
    lower, lower_index = _indexed_rows(five_minute_rows, "5m")

    daily_pos = _asof_position(daily_index, at)
    daily_fact = _row_payload(daily, daily_index, daily_pos) if daily_pos is not None else None
    if daily_fact is not None:
        if "of_status" not in daily_fact and "status" in daily_fact:
            daily_fact["of_status"] = daily_fact["status"]
        if "of_direction" not in daily_fact and "direction" in daily_fact:
            daily_fact["of_direction"] = daily_fact["direction"]
        daily_events = (event_facts or {}).get("1d")
        for field in (*_EVENT_FIELDS, "str_pivots", "str_line_events"):
            if field in daily.columns or (daily_events is not None and field in daily_events):
                daily_fact[field] = (
                    deepcopy(daily_events.get(field) or [])
                    if daily_events is not None and field in daily_events
                    else _events_at(daily, daily_index, at, field)
                )

    hourly_pos = _asof_position(hourly_index, at)
    hourly_fact = _row_payload(hourly, hourly_index, hourly_pos) if hourly_pos is not None else None
    hourly_event_pos = _exact_position(hourly_index, at)
    if hourly_fact is not None:
        hourly_fact["closed_at_t"] = hourly_event_pos is not None
        hourly_events = (event_facts or {}).get("1h")
        for field in _EVENT_FIELDS:
            hourly_fact[field] = (
                deepcopy(hourly_events.get(field) or [])
                if hourly_events is not None and field in hourly_events
                else _events_at(hourly, hourly_index, at, field)
            )

    lower_event_pos = _exact_position(lower_index, at)
    if lower_event_pos is not None:
        lower_fact = _row_payload(lower, lower_index, lower_event_pos)
    else:
        lower_fact = {"known_at": at.isoformat(), "orderflow": None}
    lower_events = (event_facts or {}).get("5m")
    for field in _EVENT_FIELDS:
        lower_fact[field] = (
            deepcopy(lower_events.get(field) or [])
            if lower_events is not None and field in lower_events
            else _events_at(lower, lower_index, at, field)
        )

    normalized_asset = str(asset).strip().upper()
    if symbol_code is not None and (len(str(symbol_code)) != 3 or not str(symbol_code).isdigit()):
        raise ValueError("symbol_code must contain exactly three digits")
    if pip_size is not None and (not isfinite(float(pip_size)) or float(pip_size) <= 0):
        raise ValueError("pip_size must be a positive finite number")

    return {
        "asset": normalized_asset,
        "symbol": normalized_asset,
        "symbol_code": str(symbol_code) if symbol_code is not None else None,
        "pip_size": float(pip_size) if pip_size is not None else None,
        "time": at.isoformat(),
        "bar_5m": _bar_payload(bar_5m, at),
        "1d": daily_fact,
        "1h": hourly_fact,
        "5m": lower_fact,
    }
