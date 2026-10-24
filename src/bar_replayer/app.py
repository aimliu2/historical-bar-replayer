from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from bar_replayer.data_source import (
    DataFileNotFound,
    DataSourceError,
    TIMEFRAME_DURATIONS,
    available_symbols,
    available_timeframes,
    load_bars,
    parquet_path,
    resolve_data_root,
)
from bar_replayer.config import DEFAULT_CONFIG_PATH, load_config
from bar_replayer.session import DualReplaySession, ReplaySession
from bar_replayer.session_store import SessionStore
from smc.anchors import sha256_file


MAX_VISIBLE_BARS = 2_000
DUAL_TIMEFRAMES = {"lower": "5m", "higher": "1h"}


def create_app(
    data_root: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Flask:
    app = Flask(__name__)
    defaults = load_config(config_path)
    environment_root = os.environ.get("BAR_REPLAYER_DATA_ROOT")
    if data_root is not None and str(data_root).strip():
        root_value = data_root
    elif environment_root and environment_root.strip():
        root_value = environment_root
    else:
        root_value = defaults["data_root"]
    resolved_data_root = resolve_data_root(root_value)
    sessions = SessionStore()
    app.config["BAR_REPLAYER_DATA_ROOT"] = resolved_data_root
    app.config["BAR_REPLAYER_DEFAULTS"] = defaults
    app.extensions["bar_replayer_sessions"] = sessions

    def failure(message: str, status: int):
        return jsonify({"error": message}), status

    def get_session(session_id: str) -> ReplaySession | DualReplaySession | tuple[Any, int]:
        try:
            return sessions.get(session_id)
        except KeyError as exc:
            return failure(str(exc.args[0]), 404)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            defaults={
                "symbol": defaults["symbol"],
                "mode": defaults["mode"],
                "timeframe": defaults["single"]["timeframe"],
                "dual_clock": defaults["lanes"]["clock"],
                "start_time": defaults["start_time"],
                "start_time_picker": defaults["start_time_picker"],
                "initial_fetch_bars": defaults["single"]["window_bars"],
                "dual_window_bars": defaults["lanes"]["window_bars"],
                "dual_focus_context_bars": defaults["lanes"]["focus_context_bars"],
                "ta_plot_1h": defaults["lanes"]["ta"]["plot"]["1h"],
            },
        )

    @app.get("/api/symbols")
    def symbols():
        if not resolved_data_root.is_dir():
            return failure(
                "The data directory is unavailable. Create the repository's 0_data link "
                "or set BAR_REPLAYER_DATA_ROOT to a parquet directory.",
                503,
            )
        return jsonify({"symbols": available_symbols(resolved_data_root)})

    @app.get("/api/timeframes")
    def timeframes():
        symbol = request.args.get("symbol", "").strip().upper()
        if not symbol:
            return failure("Choose a symbol before loading timeframes.", 400)
        if symbol not in available_symbols(resolved_data_root):
            return jsonify({"symbol": symbol, "timeframes": []})
        return jsonify(
            {"symbol": symbol, "timeframes": available_timeframes(symbol, resolved_data_root)}
        )

    @app.post("/api/replay")
    def create_replay():
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return failure("Send a JSON object with symbol and timeframe.", 400)

        mode = str(payload.get("mode", defaults["mode"])).strip().lower()
        if mode not in {"single", "dual", "tri"}:
            return failure("mode must be 'single', 'dual' or 'tri'.", 400)

        symbol = str(payload.get("symbol", defaults["symbol"])).strip().upper()
        catalog_symbols = available_symbols(resolved_data_root)
        if "symbol" not in payload and symbol not in catalog_symbols and catalog_symbols:
            symbol = catalog_symbols[0]
        if not symbol:
            return failure("Choose a symbol.", 400)
        if symbol not in catalog_symbols:
            return failure(f"Symbol '{symbol}' is not available under the configured data root.", 404)
        catalog_timeframes = available_timeframes(symbol, resolved_data_root)

        start_time = payload.get("start_time", defaults["start_time"])
        if start_time is not None and not isinstance(start_time, str):
            return failure("start_time must be a timestamp string or empty.", 400)

        try:
            if mode in {"dual", "tri"}:
                pair_keys = {
                    "timeframe", "timeframes", "lower_tf", "higher_tf",
                    "lower_timeframe", "higher_timeframe",
                }
                if pair_keys.intersection(payload):
                    return failure("Dual and Tri replay use the fixed 1H / 5m pair; do not send timeframe fields.", 400)
                clock = payload.get("clock", defaults["lanes"]["clock"])
                try:
                    clock = DualReplaySession._validate_clock(clock)
                except ValueError as exc:
                    return failure(str(exc), 400)
                required = {DUAL_TIMEFRAMES["lower"], DUAL_TIMEFRAMES["higher"]}
                if not required.issubset(catalog_timeframes):
                    return failure(
                        f"Dual replay requires both {DUAL_TIMEFRAMES['higher'].upper()} and {DUAL_TIMEFRAMES['lower'].upper()} data for {symbol}.",
                        404,
                    )
                lower_bars = load_bars(symbol, DUAL_TIMEFRAMES["lower"], resolved_data_root)
                higher_bars = load_bars(symbol, DUAL_TIMEFRAMES["higher"], resolved_data_root)
                lanes = defaults["lanes"]
                symbol_assets = defaults["symbols"].get(symbol, {})
                anchor_set = symbol_assets.get("anchor_set")
                run_lanes = lanes["enabled"] and bool(anchor_set)
                lanes_reason = (
                    "disabled"
                    if not lanes["enabled"]
                    else "no_anchor_set"
                    if not anchor_set
                    else None
                )
                structure_settings = {
                    **lanes["structure"],
                    "enabled": run_lanes,
                }
                anchor_settings = (
                    {"enabled": True, "set_version": anchor_set}
                    if anchor_set
                    else {"enabled": False}
                )
                daily_bars = None
                daily_anchor_settings = {"enabled": False}
                if mode == "tri":
                    if run_lanes and "1d" in catalog_timeframes:
                        daily_bars = load_bars(symbol, "1d", resolved_data_root)
                    daily_anchor_set = symbol_assets.get("anchor_set_1d")
                    if daily_anchor_set:
                        daily_anchor_settings = {"enabled": True, "set_version": daily_anchor_set}
                lane_hashes = (
                    {
                        **({
                            "1d": sha256_file(parquet_path(symbol, "1d", resolved_data_root))
                        } if daily_bars is not None else {}),
                        "1h": sha256_file(
                            parquet_path(symbol, DUAL_TIMEFRAMES["higher"], resolved_data_root)
                        ),
                        "5m": sha256_file(
                            parquet_path(symbol, DUAL_TIMEFRAMES["lower"], resolved_data_root)
                        ),
                    }
                    if run_lanes
                    else None
                )
                session = DualReplaySession(
                    lower_bars,
                    higher_bars,
                    symbol=symbol,
                    clock=clock,
                    start_time=start_time,
                    window_bars=lanes["window_bars"],
                    focus_context_bars=lanes["focus_context_bars"],
                    structure_settings=structure_settings,
                    orderflow_settings=lanes["orderflow"],
                    str_defense_settings=lanes["str_defense"],
                    ta_settings=lanes["ta"],
                    str_bars_1d=lanes["str_bars_1d"],
                    anchor_settings=anchor_settings,
                    data_sha256=lane_hashes,
                    mode=mode,
                    daily_bars=daily_bars,
                    daily_anchor_settings=daily_anchor_settings,
                    symbol_code=symbol_assets.get("code"),
                    pip_size=symbol_assets.get("pip_size"),
                    lanes_reason=lanes_reason,
                )
            else:
                timeframe = str(payload.get("timeframe", defaults["single"]["timeframe"])).strip().lower()
                if "timeframe" not in payload and timeframe not in catalog_timeframes and catalog_timeframes:
                    timeframe = catalog_timeframes[0]
                if timeframe not in TIMEFRAME_DURATIONS:
                    supported = ", ".join(TIMEFRAME_DURATIONS)
                    return failure(f"Choose a supported timeframe: {supported}.", 400)
                visible_bars_value = payload.get("visible_bars", defaults["single"]["window_bars"])
                if isinstance(visible_bars_value, bool):
                    return failure("visible_bars must be a positive integer.", 400)
                try:
                    visible_bars = int(visible_bars_value)
                except (TypeError, ValueError):
                    return failure("visible_bars must be a positive integer.", 400)
                if visible_bars < 1 or visible_bars > MAX_VISIBLE_BARS:
                    return failure(f"visible_bars must be between 1 and {MAX_VISIBLE_BARS}.", 400)
                bars = load_bars(symbol, timeframe, resolved_data_root)
                session = ReplaySession(
                    bars,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    visible_bars=visible_bars,
                )
        except DataFileNotFound as exc:
            return failure(str(exc), 404)
        except DataSourceError as exc:
            return failure(str(exc), 422)
        except ValueError as exc:
            return failure(str(exc), 400)

        sessions.add(session)
        return jsonify({"session_id": session.session_id, "snapshot": session.snapshot()})

    @app.get("/api/replay/<session_id>")
    def snapshot(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        return jsonify({"snapshot": session.snapshot()})

    @app.post("/api/replay/<session_id>/step")
    def step(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        if isinstance(session, DualReplaySession):
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return failure("Send a JSON object with optional lanes.", 400)
            try:
                return jsonify({"delta": session.step(payload.get("lanes"))})
            except ValueError as exc:
                return failure(str(exc), 400)
        return jsonify({"snapshot": session.step()})

    @app.post("/api/replay/<session_id>/back")
    def back(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        return jsonify({"snapshot": session.back()})

    @app.post("/api/replay/<session_id>/seek")
    def seek(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        if isinstance(session, DualReplaySession):
            return failure("seek is available only for single replay; use rewind for dual or Tri replay.", 400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "bar_index" not in payload:
            return failure("Send a JSON object with an integer bar_index.", 400)
        try:
            updated = session.seek(payload["bar_index"])
        except ValueError as exc:
            return failure(str(exc), 400)
        return jsonify({"snapshot": updated})

    @app.post("/api/replay/<session_id>/reset")
    def reset(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        return jsonify({"snapshot": session.reset()})

    @app.post("/api/replay/<session_id>/clock")
    def set_clock(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        if not isinstance(session, DualReplaySession):
            return failure("Clock selection is available only for dual or Tri replay.", 400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "clock" not in payload:
            return failure("Send a JSON object with clock set to '1h' or '5m'.", 400)
        try:
            updated = session.set_clock(payload["clock"])
        except ValueError as exc:
            return failure(str(exc), 400)
        return jsonify({"snapshot": updated})

    @app.post("/api/replay/<session_id>/rewind")
    def rewind(session_id: str):
        session = get_session(session_id)
        if isinstance(session, tuple):
            return session
        if not isinstance(session, DualReplaySession):
            return failure("Rewind to a bar close is available only for dual or Tri replay.", 400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("time"), str):
            return failure("Send a JSON object with a timestamp string in time.", 400)
        try:
            updated = session.rewind(payload["time"])
        except ValueError as exc:
            return failure(str(exc), 400)
        return jsonify({"snapshot": updated})

    return app


def main() -> None:
    host = os.environ.get("BAR_REPLAYER_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("BAR_REPLAYER_PORT", "5055"))
    except ValueError as exc:
        raise SystemExit("BAR_REPLAYER_PORT must be an integer.") from exc

    app = create_app()
    print(f"Bar replayer data root: {app.config['BAR_REPLAYER_DATA_ROOT']}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
