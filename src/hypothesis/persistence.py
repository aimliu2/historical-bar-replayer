from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from .runtime import Layer4


class Layer4SaveError(RuntimeError):
    """A missing, malformed, or incompatible Layer 4 save needs operator action."""


def _symbol(config: dict[str, Any]) -> str:
    replay = config.get("replay") or {}
    return str(config.get("symbol") or config.get("asset") or replay.get("symbol") or "").strip().upper()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _source_digest(repo_root: Path) -> str:
    source_root = repo_root / "src" / "hypothesis"
    if not source_root.is_dir():
        raise Layer4SaveError(f"Layer 4 source directory is missing: {source_root}")
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in paths:
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def layer4_save_key(
    config: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    symbol: str | None = None,
) -> dict[str, str]:
    """Hash Layer 4 source and only the settings that define this asset's state."""
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    asset = str(symbol or _symbol(config)).strip().upper()
    if not asset:
        raise Layer4SaveError("Layer 4 save key requires a configured symbol")
    raw_symbol_config = deepcopy(
        (config.get("symbols") or {}).get(asset) or config.get("asset_config") or {}
    )
    runtime_settings = {
        "symbol": asset,
        "symbol_config": {
            key: raw_symbol_config.get(key)
            for key in ("code", "pip_size", "sl_floor_pips", "sl_cap_pips")
            if key in raw_symbol_config
        },
        "layer4": deepcopy(config.get("layer4") or {}),
    }
    # Daily confirmation changes the facts from which Layer 4 memory was built.
    # Omit the default to preserve the baseline settings hash.
    str_bars_1d = (config.get("lanes") or {}).get("str_bars_1d", 2)
    if str_bars_1d != 2:
        runtime_settings["str_bars_1d"] = str_bars_1d
    source_sha256 = _source_digest(root)
    config_sha256 = hashlib.sha256(_json_bytes(runtime_settings)).hexdigest()
    key_payload = {
        "symbol": asset,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
    }
    return {**key_payload, "sha256": hashlib.sha256(_json_bytes(key_payload)).hexdigest()}


def layer4_save_path(repo_root: str | Path, symbol: str) -> Path:
    asset = str(symbol).strip().upper()
    if not asset or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in asset):
        raise ValueError("Layer 4 save symbol contains unsupported characters")
    root = Path(repo_root).resolve()
    return root / "export" / asset / "layer4" / f"{asset}_layer4_save.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def save_layer4(
    runtime: Layer4,
    *,
    config: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
    save_path: str | Path | None = None,
    inputs: dict[str, Any] | None = None,
) -> Path:
    """Atomically write the complete Layer 4 state after a 5m close."""
    if runtime.last_time is None:
        raise Layer4SaveError("Cannot save Layer 4 before its first 5m close")
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    config_value = config or runtime.config
    key = layer4_save_key(config_value, repo_root=root, symbol=runtime.symbol)
    path = Path(save_path) if save_path is not None else layer4_save_path(root, runtime.symbol)
    if not path.is_absolute():
        path = root / path
    payload = {
        "format_version": 1,
        "symbol": runtime.symbol,
        "key": key,
        "last_close": runtime.last_time,
        "state": runtime.state(),
        "metadata": {
            "output_path": path.resolve().relative_to(root).as_posix()
            if path.resolve().is_relative_to(root)
            else str(path.resolve()),
            "config_sha256": key["config_sha256"],
            "input_files": deepcopy(inputs or {}),
        },
    }
    _atomic_write_json(path, payload)
    return path


def load_layer4(
    config: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    save_path: str | Path | None = None,
    cold_start: bool = False,
) -> tuple[Layer4, dict[str, Any]]:
    """Load a compatible save; missing or stale saves fail unless explicitly cold-started."""
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    symbol = _symbol(config)
    path = Path(save_path) if save_path is not None else layer4_save_path(root, symbol)
    if not path.is_absolute():
        path = root / path
    reason: str | None = None
    payload: dict[str, Any] | None = None
    if not path.is_file():
        reason = f"Layer 4 save is missing: {path}"
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("save root must be a JSON object")
            payload = raw
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            reason = f"Layer 4 save is unreadable: {exc}"

    expected_key = layer4_save_key(config, repo_root=root, symbol=symbol)
    if payload is not None:
        if int(payload.get("format_version", 0)) != 1:
            reason = "Layer 4 save format version does not match this runtime"
        elif str(payload.get("symbol", "")).upper() != symbol:
            reason = "Layer 4 save symbol does not match this runtime"
        elif payload.get("key") != expected_key:
            reason = (
                "Layer 4 save key does not match current source or settings; "
                "rebuild the save or pass cold_start=True"
            )
        elif not isinstance(payload.get("state"), dict):
            reason = "Layer 4 save does not contain a Layer 4 state"
        else:
            try:
                runtime = Layer4.from_state(config, payload["state"])
            except (KeyError, TypeError, ValueError) as exc:
                reason = f"Layer 4 save state is invalid: {exc}"
            else:
                if runtime.last_time is None or str(runtime.last_time) != str(payload.get("last_close")):
                    reason = "Layer 4 save last_close does not match its state"
                else:
                    return runtime, {
                        "resumed": True,
                        "cold_start": False,
                        "reason": None,
                        "last_close": runtime.last_time,
                        "save_path": str(path),
                        "key": expected_key,
                    }

    if not cold_start:
        raise Layer4SaveError(reason or "Layer 4 save could not be loaded; pass cold_start=True to run from genesis")
    return Layer4(config), {
        "resumed": False,
        "cold_start": True,
        "reason": reason,
        "last_close": None,
        "save_path": str(path),
        "key": expected_key,
    }


def stream_contexts(
    runtime: Layer4,
    contexts: Iterable[dict[str, Any]],
    *,
    catch_up: bool,
    save_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Advance every later close in order, mark catch-up outputs, and save after each close."""
    last_time = pd.Timestamp(runtime.last_time) if runtime.last_time is not None else None
    if last_time is not None:
        last_time = last_time.tz_localize("UTC") if last_time.tzinfo is None else last_time.tz_convert("UTC")
    for context in contexts:
        at = pd.Timestamp(context.get("time"))
        at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
        if last_time is not None and at <= last_time:
            continue
        output = runtime.step(context, catch_up=catch_up)
        last_time = at
        if save_path is not None:
            save_layer4(
                runtime,
                config=config,
                repo_root=repo_root,
                save_path=save_path,
                inputs=inputs,
            )
        yield output
