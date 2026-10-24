from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from order_plan import order_plan_settings

from bar_replayer.data_source import TIMEFRAME_DURATIONS
from bar_replayer.structure_lane import default_1h_engine_config, default_5m_engine_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
MAX_VISIBLE_BARS = 2_000


def _reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    path: str,
    *,
    pointers: dict[str, str] | None = None,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        key_path = f"{path}.{unknown[0]}"
        pointer = (pointers or {}).get(key_path)
        suffix = f" Use '{pointer}' instead." if pointer else ""
        raise ValueError(f"Unknown bar replayer config key '{key_path}'.{suffix}")


def _window_size(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_VISIBLE_BARS:
        raise ValueError(f"Bar replayer config '{key}' must be a positive integer from 1 to {MAX_VISIBLE_BARS}.")
    return value


def symbol_assets(raw_symbols: Any) -> dict[str, dict[str, Any]]:
    """Normalize symbol settings for the shared config and study overrides."""
    if not isinstance(raw_symbols, dict):
        raise ValueError("Bar replayer config 'symbols' must be a mapping of upper-case symbols to asset mappings.")

    symbols: dict[str, dict[str, Any]] = {}
    for symbol, raw_assets in raw_symbols.items():
        if not isinstance(symbol, str) or not symbol.strip() or symbol != symbol.strip().upper():
            raise ValueError("Bar replayer config 'symbols' keys must be non-empty upper-case symbol names.")
        if not isinstance(raw_assets, dict):
            raise ValueError(f"Bar replayer config 'symbols.{symbol}' must be a mapping.")
        _reject_unknown(
            raw_assets,
            {"code", "pip_size", "sl_floor_pips", "sl_cap_pips", "anchor_set", "anchor_set_1d"},
            f"symbols.{symbol}",
            pointers={f"symbols.{symbol}.daily_export": f"symbols.{symbol}.anchor_set_1d"},
        )

        assets: dict[str, Any] = {}
        if "code" in raw_assets:
            value = raw_assets["code"]
            if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).isdigit() or len(str(value)) != 3:
                raise ValueError(f"Bar replayer config 'symbols.{symbol}.code' must contain exactly three digits.")
            assets["code"] = str(value)
        if "pip_size" in raw_assets:
            value = raw_assets["pip_size"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Bar replayer config 'symbols.{symbol}.pip_size' must be a positive finite number.")
            try:
                pip_size = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    f"Bar replayer config 'symbols.{symbol}.pip_size' must be a positive finite number."
                ) from exc
            if not math.isfinite(pip_size) or pip_size <= 0:
                raise ValueError(f"Bar replayer config 'symbols.{symbol}.pip_size' must be a positive finite number.")
            assets["pip_size"] = pip_size
        for key in ("sl_floor_pips", "sl_cap_pips"):
            if key in raw_assets:
                try:
                    assets[key] = _finite_number(raw_assets[key], f"symbols.{symbol}.{key}", minimum=0, strict=True)
                except ValueError as exc:
                    raise ValueError(f"symbols.{symbol}.{key} must be a positive finite number") from exc
        if "sl_cap_pips" in assets and assets["sl_cap_pips"] < assets.get("sl_floor_pips", 13.0):
            raise ValueError(f"symbols.{symbol}.sl_cap_pips must be at least sl_floor_pips")
        for key in ("anchor_set", "anchor_set_1d"):
            if key in raw_assets:
                value = raw_assets[key]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Bar replayer config 'symbols.{symbol}.{key}' must be a non-empty string.")
                assets[key] = value.strip()
        symbols[symbol] = assets
    return symbols


def _finite_number(value: Any, key: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Bar replayer config '{key}' must be a finite number.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Bar replayer config '{key}' must be a finite number.") from exc
    if not math.isfinite(number) or (minimum is not None and (number <= minimum if strict else number < minimum)):
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"Bar replayer config '{key}' must be finite and {qualifier} {minimum}.")
    return number


def _layer4_config(raw_layer4: Any) -> dict[str, Any]:
    if raw_layer4 is None:
        raw_layer4 = {}
    if not isinstance(raw_layer4, dict):
        raise ValueError("Bar replayer config 'layer4' must be a mapping.")
    _reject_unknown(raw_layer4, {"hypotheses", "min_gap_pips", "tp_r", "retry_r", "SMC-3TF"}, "layer4")

    hypotheses = raw_layer4.get("hypotheses", ["SMC-3TF"])
    if not isinstance(hypotheses, list) or any(not isinstance(item, str) or not item.strip() for item in hypotheses):
        raise ValueError("Bar replayer config 'layer4.hypotheses' must be a list of non-empty names.")
    hypotheses = [item.strip() for item in hypotheses]
    if len(set(hypotheses)) != len(hypotheses):
        raise ValueError("Bar replayer config 'layer4.hypotheses' must not contain duplicates.")

    raw_smc = raw_layer4.get("SMC-3TF", {})
    if not isinstance(raw_smc, dict):
        raise ValueError("Bar replayer config 'layer4.SMC-3TF' must be a mapping.")
    _reject_unknown(
        raw_smc,
        {"code", "postpone_mode", "skip_shape", "near_cross_pips", "retry_policy", "retry_3s", "old_side_expiry_days"},
        "layer4.SMC-3TF",
        pointers={"layer4.SMC-3TF.retry_types": "layer4.SMC-3TF.retry_policy"},
    )
    code = raw_smc.get("code", 1)
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 9:
        raise ValueError("Bar replayer config 'layer4.SMC-3TF.code' must be one decimal digit.")
    postpone_mode = str(raw_smc.get("postpone_mode", "touch")).strip().lower()
    if postpone_mode not in {"touch", "cross"}:
        raise ValueError("Bar replayer config 'layer4.SMC-3TF.postpone_mode' must be 'touch' or 'cross'.")
    skip_shape = str(raw_smc.get("skip_shape", "B")).strip().upper()
    if skip_shape not in {"A", "B"}:
        raise ValueError("Bar replayer config 'layer4.SMC-3TF.skip_shape' must be 'A' or 'B'.")
    retry_policy = raw_smc.get("retry_policy", {key: "5x9" for key in ("1", "2a", "2b", "3")})
    if not isinstance(retry_policy, dict):
        raise ValueError("layer4.SMC-3TF.retry_policy must be a mapping")
    if any(key not in {"1", "2a", "2b", "3"} for key in retry_policy):
        raise ValueError("layer4.SMC-3TF.retry_policy keys must be 1, 2a, 2b or 3")
    if any(value not in ("5x9", "3T", "3S") for value in retry_policy.values()):
        raise ValueError("layer4.SMC-3TF.retry_policy values must be 5x9, 3T or 3S")

    if any(value in ("3T", "3S") and key != "3" for key, value in retry_policy.items()):
        raise ValueError("layer4.SMC-3TF.retry_policy: 3T / 3S are only valid for type 3")

    retry_3s = raw_smc.get("retry_3s", {})
    if not isinstance(retry_3s, dict):
        raise ValueError("layer4.SMC-3TF.retry_3s must be a mapping")
    _reject_unknown(retry_3s, {"spread_pips", "sl_floor_pips", "sl_cap_pips"}, "layer4.SMC-3TF.retry_3s")
    retry_3s = {key: _finite_number(value, f"layer4.SMC-3TF.retry_3s.{key}", minimum=0, strict=True)
                for key, value in retry_3s.items()}
    if retry_3s.get("sl_cap_pips", 25.0) < retry_3s.get("sl_floor_pips", 13.0):
        raise ValueError("retry_3s.sl_cap_pips must be at least sl_floor_pips")
    smc = {
        "code": code,
        "postpone_mode": postpone_mode,
        "skip_shape": skip_shape,
        "near_cross_pips": _finite_number(raw_smc.get("near_cross_pips", 1.0), "layer4.SMC-3TF.near_cross_pips", minimum=0),
        "retry_policy": dict(retry_policy),
        "retry_3s": retry_3s,
        "old_side_expiry_days": _finite_number(raw_smc.get("old_side_expiry_days", 5), "layer4.SMC-3TF.old_side_expiry_days", minimum=0, strict=True),
    }
    return {
        "hypotheses": hypotheses,
        "min_gap_pips": _finite_number(raw_layer4.get("min_gap_pips", 2.0), "layer4.min_gap_pips", minimum=0),
        "retry_r": _finite_number(raw_layer4.get("retry_r", 2.0), "layer4.retry_r", minimum=0, strict=True),
        "tp_r": _finite_number(raw_layer4.get("tp_r", 3.0), "layer4.tp_r", minimum=0, strict=True),
        "SMC-3TF": smc,
    }


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in bar replayer config {config_path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read bar replayer config {config_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"Bar replayer config {config_path} must contain a YAML mapping.")
    _reject_unknown(raw_config, {"data", "replay", "symbols", "layer4", "order_plan"}, "root")

    data = raw_config.get("data")
    replay = raw_config.get("replay")
    if not isinstance(data, dict):
        raise ValueError("Bar replayer config must contain a 'data' mapping.")
    if not isinstance(replay, dict):
        raise ValueError("Bar replayer config must contain a 'replay' mapping.")
    _reject_unknown(
        data,
        {"root"},
        "data",
    )

    legacy_pointers = {
        "replay.dual": (
            "replay.lanes (clock, window_bars, focus_context_bars, enabled, orderflow_5m, ema_plot); "
            "symbols.<SYM>.anchor_set, symbols.<SYM>.anchor_set_1d and symbols.<SYM>.pip_size for symbol assets. "
            "Engine toggles, 1H tier, EMA spans, shock and anchor cadence are fixed in code or by the anchor set"
        ),
        "replay.timeframe": "replay.single.timeframe",
        "replay.initial_fetch_bars": "replay.single.window_bars",
        "replay.lookback_bars": "replay.lanes.window_bars",
    }
    _reject_unknown(
        replay,
        {"mode", "symbol", "start_time", "single", "lanes"},
        "replay",
        pointers=legacy_pointers,
    )

    data_root = data.get("root")
    if not isinstance(data_root, str) or not data_root.strip():
        raise ValueError("Bar replayer config 'data.root' must be a non-empty path string.")

    symbol_value = replay.get("symbol", "EURUSD")
    if not isinstance(symbol_value, str) or not symbol_value.strip():
        raise ValueError("Bar replayer config 'replay.symbol' must be a non-empty string.")

    single = replay.get("single", {})
    lanes = replay.get("lanes", {})
    if not isinstance(single, dict):
        raise ValueError("Bar replayer config 'replay.single' must be a mapping.")
    if not isinstance(lanes, dict):
        raise ValueError("Bar replayer config 'replay.lanes' must be a mapping.")
    _reject_unknown(
        single,
        {"timeframe", "window_bars"},
        "replay.single",
        pointers={
            "replay.single.initial_fetch_bars": "replay.single.window_bars",
        },
    )
    _reject_unknown(
        lanes,
        {"enabled", "clock", "window_bars", "focus_context_bars", "orderflow_5m", "ema_plot", "str_bars_1d"},
        "replay.lanes",
        pointers={
            "replay.lanes.structure": "replay.lanes.enabled",
            "replay.lanes.orderflow": "replay.lanes.orderflow_5m",
            "replay.lanes.lookback_bars": "replay.lanes.window_bars",
            "replay.lanes.anchors": "symbols.<SYM>.anchor_set",
            "replay.lanes.daily": "symbols.<SYM>.anchor_set_1d and symbols.<SYM>.pip_size",
        },
    )

    mode_value = replay.get("mode", "tri")
    if not isinstance(mode_value, str) or mode_value.strip().lower() not in {"single", "dual", "tri"}:
        raise ValueError("Replay mode must be 'single', 'dual' or 'tri'.")
    mode = mode_value.strip().lower()

    timeframe_value = single.get("timeframe", "1d")
    if not isinstance(timeframe_value, str):
        raise ValueError("Bar replayer config 'replay.single.timeframe' must be a string.")
    timeframe = timeframe_value.strip().lower()
    if timeframe not in TIMEFRAME_DURATIONS:
        supported = ", ".join(TIMEFRAME_DURATIONS)
        raise ValueError(
            f"Unsupported default replay timeframe '{timeframe_value}'. Supported values: {supported}."
        )

    start_value = replay.get("start_time", "2026-01-01")
    if not isinstance(start_value, (str, date, datetime)) or not str(start_value).strip():
        raise ValueError("Bar replayer config 'replay.start_time' must be a date or timestamp string.")
    try:
        start_timestamp = pd.Timestamp(start_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Bar replayer config 'replay.start_time' must be a valid UTC date or timestamp."
        ) from exc
    if pd.isna(start_timestamp):
        raise ValueError("Bar replayer config 'replay.start_time' must be a valid UTC date or timestamp.")
    if start_timestamp.tzinfo is None:
        start_timestamp = start_timestamp.tz_localize("UTC")
    else:
        start_timestamp = start_timestamp.tz_convert("UTC")

    single_window_bars = _window_size(
        single.get("window_bars", 200),
        "replay.single.window_bars",
    )

    clock_value = lanes.get("clock", "1h")
    if not isinstance(clock_value, str) or clock_value.strip().lower() not in {"1h", "5m"}:
        raise ValueError("Replay lanes clock must be '1h' or '5m'.")
    clock = clock_value.strip().lower()

    window_value = lanes.get("window_bars", {"1h": 500, "5m": 1000})
    if not isinstance(window_value, dict):
        raise ValueError("Bar replayer config 'replay.lanes.window_bars' must be a mapping.")
    if set(window_value) != {"1h", "5m"}:
        raise ValueError("Bar replayer config 'replay.lanes.window_bars' must contain exactly '1h' and '5m'.")
    windows = {
        timeframe: _window_size(value, f"replay.lanes.window_bars.{timeframe}")
        for timeframe, value in window_value.items()
    }

    focus_context_bars = lanes.get("focus_context_bars", 36)
    if (
        isinstance(focus_context_bars, bool)
        or not isinstance(focus_context_bars, int)
        or not 0 <= focus_context_bars <= MAX_VISIBLE_BARS
    ):
        raise ValueError(
            f"Bar replayer config 'replay.lanes.focus_context_bars' must be an integer from 0 to {MAX_VISIBLE_BARS}."
        )

    str_bars_1d = lanes.get("str_bars_1d", 2)
    if type(str_bars_1d) is not int or str_bars_1d < 1:
        raise ValueError("Bar replayer config 'replay.lanes.str_bars_1d' must be a positive integer.")

    lanes_enabled = lanes.get("enabled", True)
    if not isinstance(lanes_enabled, bool):
        raise ValueError("Bar replayer config 'replay.lanes.enabled' must be a boolean.")

    default_1h = default_1h_engine_config()
    orderflow_5m_default = default_5m_engine_config()["orderflow"]["tier"]
    orderflow_5m = lanes.get("orderflow_5m", orderflow_5m_default)
    if not isinstance(orderflow_5m, str) or orderflow_5m.strip().lower() not in {"itr", "ltr"}:
        raise ValueError("Bar replayer config 'replay.lanes.orderflow_5m' must be 'itr' or 'ltr'.")
    orderflow_5m = orderflow_5m.strip().lower()
    default_5m = default_5m_engine_config(orderflow_5m)

    ema_spans = list(default_1h["ta"]["ema"])
    ema_plot = lanes.get("ema_plot", 100)
    if isinstance(ema_plot, bool) or not isinstance(ema_plot, int) or ema_plot not in ema_spans:
        choices = ", ".join(str(value) for value in ema_spans)
        raise ValueError(f"Bar replayer config 'replay.lanes.ema_plot' must be one of {choices}.")

    symbols = symbol_assets(raw_config.get("symbols", {}))
    layer4 = _layer4_config(raw_config.get("layer4"))
    order_plan = order_plan_settings(raw_config.get("order_plan"))

    return {
        "data_root": data_root.strip(),
        "mode": mode,
        "symbol": symbol_value.strip().upper(),
        "start_time": start_timestamp.isoformat(),
        "start_time_picker": start_timestamp.strftime("%Y-%m-%dT%H:%M"),
        "single": {"timeframe": timeframe, "window_bars": single_window_bars},
        "lanes": {
            "enabled": lanes_enabled,
            "str_bars_1d": str_bars_1d,
            "clock": clock,
            "window_bars": windows,
            "focus_context_bars": focus_context_bars,
            "orderflow_5m": orderflow_5m,
            "ema_plot": ema_plot,
            "structure": {"enabled": lanes_enabled, "lookback_bars": dict(windows)},
            "orderflow": {
                "enabled": bool(default_1h["orderflow"]["enabled"] and default_5m["orderflow"]["enabled"]),
                "tier": {
                    "1h": default_1h["orderflow"]["tier"],
                    "5m": default_5m["orderflow"]["tier"],
                },
                "shock": {
                    "1h": default_1h["orderflow"]["shock"],
                    "5m": default_5m["orderflow"]["shock"],
                },
            },
            "str_defense": {
                "enabled": bool(default_1h["str_defense"]["enabled"]),
                "1h": bool(default_1h["str_defense"]["enabled"]),
            },
            "ta": {
                "enabled": bool(default_1h["ta"]["enabled"]),
                "ema": {"1h": ema_spans},
                "atr": {"1h": default_1h["ta"]["atr"]},
                "band": {"1h": dict(default_1h["ta"]["band"])},
                "plot": {"1h": ema_plot},
            },
        },
        "symbols": symbols,
        "layer4": layer4,
        "order_plan": order_plan,
    }
