from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .reducer import GENESIS_SEED, StructureReducer
from .orderflow import Orderflow
from .str_defense import STRDefense
from ta.ta_context import TAContext


ANCHOR_SCHEMA_VERSION = 1
ANCHOR_SET_VERSION = "eurusd-1d-quarterly-v3"
ANCHOR_1H_SET_VERSION = "eurusd-1h-500-bars-v1"
ANCHOR_1H5M_SET_VERSION = "eurusd-1h5m-100-bars-v5"
ANCHOR_1D_SET_VERSION = "eurusd-1d-quarterly-v5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value)


def timestamp_key(value: Any) -> str:
    return canonical_timestamp(value).isoformat()


def anchor_filename(symbol: str, timeframe: str, anchor_time: Any) -> str:
    stamp = canonical_timestamp(anchor_time)
    return f"{symbol}_{timeframe}_{stamp.date().isoformat()}.json"


def build_anchor_envelope(
    reducer: StructureReducer,
    *,
    symbol: str,
    timeframe: str,
    genesis_time: Any,
    terrain_anchor_time: Any,
    orderflow: Orderflow | None = None,
    orderflow_by_tier: dict[str, Orderflow] | None = None,
    str_defense: STRDefense | None = None,
    str_defense_by_side: dict[str, STRDefense] | None = None,
    ta_context: TAContext | None = None,
    anchor_set_version: str = ANCHOR_SET_VERSION,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = reducer.export_state()
    probe_time = state["cursor"]["last_known_at"]
    if probe_time is None:
        raise ValueError("Cannot checkpoint a reducer before its first bar")
    layer3_context: dict[str, Any] = {
        "breaks": state["breaks"],
        "first_ready_at": state["first_ready_at"],
    }
    if orderflow is not None:
        layer3_context["orderflow"] = orderflow.export_state()
    if orderflow_by_tier:
        layer3_context["orderflow_by_tier"] = {
            tier: tracker.export_state()
            for tier, tracker in orderflow_by_tier.items()
        }
    if str_defense is not None:
        layer3_context["str_defense"] = str_defense.export_state()
    if str_defense_by_side:
        layer3_context["str_defense_by_side"] = {
            side: defense.export_state()
            for side, defense in str_defense_by_side.items()
        }
    if ta_context is not None:
        layer3_context["ta"] = ta_context.export_state()
    if extra_context:
        layer3_context.update(deepcopy(extra_context))
    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "anchor_set_version": anchor_set_version,
        "symbol": symbol,
        "timeframe": timeframe,
        "genesis_time": timestamp_key(genesis_time),
        "terrain_anchor_time": timestamp_key(terrain_anchor_time),
        "probe_time": probe_time,
        "bar_index": state["cursor"]["bar_index"],
        "layer2_events": {"pivot": state["pivot"]},
        "layer3_context": layer3_context,
    }


def reducer_state_from_envelope(
    envelope: dict[str, Any],
    *,
    timeframe: str = "1h",
) -> dict[str, Any]:
    lane = str(timeframe).strip().lower()
    if lane == "5m":
        try:
            state = envelope["layer3_context"]["5m"]["reducer"]
        except (KeyError, TypeError) as exc:
            raise ValueError("5m_reducer_state_missing") from exc
        return deepcopy(state)
    return {
        "state_version": StructureReducer.STATE_VERSION,
        "pivot": envelope["layer2_events"]["pivot"],
        "breaks": envelope["layer3_context"]["breaks"],
        "cursor": {
            "bar_index": int(envelope["bar_index"]),
            "last_known_at": timestamp_key(envelope["probe_time"]),
        },
        "first_ready_at": envelope["layer3_context"].get("first_ready_at", {}),
    }


def orderflow_state_from_envelope(
    envelope: dict[str, Any],
    *,
    timeframe: str = "1h",
    tier: str | None = None,
) -> dict[str, Any]:
    lane = str(timeframe).strip().lower()
    if lane == "5m":
        try:
            states = envelope["layer3_context"]["5m"]["orderflow"]
            return states[tier or "itr"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"5m_orderflow_state_missing:{tier or 'itr'}") from exc
    layer3 = envelope.get("layer3_context") or {}
    if tier is not None:
        by_tier = layer3.get("orderflow_by_tier") or {}
        if tier in by_tier:
            return deepcopy(by_tier[tier])
    try:
        state = layer3["orderflow"]
        if tier is not None and state.get("tier") != tier:
            raise ValueError(f"1h_orderflow_state_missing:{tier}")
        return deepcopy(state)
    except (KeyError, TypeError) as exc:
        raise ValueError("orderflow_state_missing") from exc


@dataclass
class NearestAnchor:
    envelope: dict[str, Any] | None
    terrain_anchor_time: str | None
    probe_time: str | None
    bar_index: int | None
    used_anchor: bool
    fallback_reason: str | None


def restore_nearest_anchor(
    *,
    window_start: Any,
    probe_time: Any,
    anchor_root: Path,
    manifest_path: Path,
    symbol: str,
    timeframe: str,
    genesis_time: Any,
    data_sha256: str,
    source_hashes: dict[str, str],
    config: dict[str, Any],
    anchor_set_version: str,
) -> NearestAnchor:
    """Load the nearest compatible checkpoint at or before the display window."""
    wanted_start = _naive_utc(window_start)
    wanted_probe = _naive_utc(probe_time)
    lane_key = str(timeframe).strip().lower()
    fallback_reason = "manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("anchor_set_version") != anchor_set_version:
            raise ValueError("anchor_set_version_mismatch")
        if manifest.get("symbol") != symbol:
            raise ValueError("symbol_mismatch")
        if manifest.get("source_hashes") != source_hashes:
            raise ValueError("source_hash_mismatch")

        lane_manifest = (manifest.get("lanes") or {}).get(lane_key)
        if lane_manifest is not None:
            if lane_manifest.get("timeframe", "").lower() != lane_key:
                raise ValueError("timeframe_mismatch")
            if _naive_utc(lane_manifest.get("genesis_time")) != _naive_utc(genesis_time):
                raise ValueError("genesis_time_mismatch")
            if lane_manifest.get("data_sha256") != data_sha256:
                raise ValueError("data_hash_mismatch")
            if lane_manifest.get("engine_config") != _anchor_config(config, lane_key):
                raise ValueError("engine_config_mismatch")
        else:
            # Keep the original single-lane manifest shape readable for existing 1H
            # callers; the combined producer writes a per-lane block above.
            if manifest.get("timeframe") != timeframe:
                raise ValueError("symbol_or_timeframe_mismatch")
            if _naive_utc(manifest.get("genesis_time")) != _naive_utc(genesis_time):
                raise ValueError("genesis_time_mismatch")
            if manifest.get("data_sha256") != data_sha256:
                raise ValueError("data_hash_mismatch")
            if manifest.get("engine_config") != config:
                raise ValueError("engine_config_mismatch")

        candidates = []
        for entry in manifest.get("anchors", []):
            terrain_time = _naive_utc(entry.get("terrain_anchor_time"))
            if terrain_time <= wanted_start:
                candidates.append((terrain_time, entry))
        if not candidates:
            raise ValueError("no_anchor_before_window_start")
        terrain_time, entry = max(candidates, key=lambda item: item[0])

        anchor_path = _resolve_anchor_path(anchor_root, entry["file"])
        if not anchor_path.is_file():
            raise ValueError("anchor_file_missing")
        if sha256_file(anchor_path) != entry.get("sha256"):
            raise ValueError("anchor_file_hash_mismatch")
        envelope = json.loads(anchor_path.read_text(encoding="utf-8"))
        expected_identity = {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "anchor_set_version": anchor_set_version,
            "symbol": symbol,
        }
        expected_timeframe = manifest.get("timeframe") if lane_manifest is not None else timeframe
        if expected_timeframe is not None:
            expected_identity["timeframe"] = expected_timeframe
        if any(envelope.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("anchor_identity_mismatch")
        lane_state = _lane_anchor_state(envelope, lane_key)
        lane_genesis_time = lane_state.get("genesis_time", envelope.get("genesis_time"))
        if _naive_utc(lane_genesis_time) != _naive_utc(genesis_time):
            raise ValueError("genesis_time_mismatch")
        if _naive_utc(envelope.get("terrain_anchor_time")) != terrain_time:
            raise ValueError("terrain_anchor_time_mismatch")

        checkpoint_probe = _naive_utc(lane_state.get("probe_time", envelope["probe_time"]))
        if checkpoint_probe != terrain_time:
            raise ValueError("probe_time_terrain_anchor_time_mismatch")
        lane_bar_index = int(lane_state.get("bar_index", envelope["bar_index"]))
        lane_indices = entry.get("bar_indices") or {}
        manifest_bar_index = int(
            lane_indices.get(lane_key, entry.get("bar_index", -1))
        )
        if manifest_bar_index != lane_bar_index:
            raise ValueError("bar_index_mismatch")
        if entry.get("probe_time") is not None and _naive_utc(entry["probe_time"]) != checkpoint_probe:
            raise ValueError("probe_time_mismatch")
        if checkpoint_probe > wanted_start or checkpoint_probe > wanted_probe:
            raise ValueError("anchor_is_after_requested_window")
        return NearestAnchor(
            envelope=envelope,
            terrain_anchor_time=terrain_time.isoformat(),
            probe_time=checkpoint_probe.isoformat(),
            bar_index=lane_bar_index,
            used_anchor=True,
            fallback_reason=None,
        )
    except FileNotFoundError:
        fallback_reason = "manifest_missing" if not manifest_path.is_file() else "anchor_file_missing"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fallback_reason = str(exc) or type(exc).__name__
    return NearestAnchor(
        envelope=None,
        terrain_anchor_time=None,
        probe_time=None,
        bar_index=None,
        used_anchor=False,
        fallback_reason=fallback_reason,
    )


def _lane_anchor_state(envelope: dict[str, Any], lane: str) -> dict[str, Any]:
    if lane == "5m":
        try:
            return envelope["layer3_context"]["5m"]
        except (KeyError, TypeError) as exc:
            raise ValueError("5m_anchor_state_missing") from exc
    return envelope


def _anchor_config(config: dict[str, Any], lane: str) -> dict[str, Any]:
    normalized = deepcopy(config)
    if lane == "5m":
        # Both modes are stored, so the selected lane mode is not an anchor-set
        # compatibility dimension.
        (normalized.get("orderflow") or {}).pop("tier", None)
        defense = normalized.get("str_defense") or {}
        if not defense.get("enabled", False):
            # 5m has no STR defense. Its default tier follows the selected flow
            # mode, but that disabled tracker must not invalidate shared anchors.
            defense["orderflow_tier"] = "itr"
    return normalized


def _resolve_anchor_path(anchor_root: Path, manifest_file: str) -> Path:
    candidate = Path(manifest_file)
    if candidate.is_absolute():
        return candidate
    for root in (anchor_root, *anchor_root.parents):
        path = root / candidate
        if path.is_file():
            return path
    return anchor_root / candidate


def _naive_utc(value: Any) -> pd.Timestamp:
    if value is None:
        raise ValueError("timestamp_missing")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp_invalid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def save_anchor(path: Path, envelope: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha256_file(path)


@dataclass
class AnchorRestore:
    reducer: StructureReducer
    orderflow: Orderflow
    used_anchor: bool
    fallback_reason: str | None
    processed_through: str | None


def restore_exact_anchor_or_genesis(
    bars: pd.DataFrame,
    *,
    anchor_time: Any,
    probe_time: Any,
    anchor_root: Path,
    manifest_path: Path,
    symbol: str,
    timeframe: str,
    genesis_time: Any,
    data_sha256: str,
    source_hashes: dict[str, str],
    config: dict[str, Any],
    genesis_seed: str = GENESIS_SEED,
    shock_ratios: pd.Series | None = None,
) -> AnchorRestore:
    """Restore one exact anchor or replay from genesis through the requested probe.

    This deliberately does not search for the nearest earlier checkpoint. A bad,
    absent, or incompatible exact key takes the full genesis path.
    """
    wanted_anchor = timestamp_key(anchor_time)
    wanted_probe = canonical_timestamp(probe_time)
    manifest = None
    fallback_reason = "manifest_missing"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("anchor_set_version") != ANCHOR_SET_VERSION:
            raise ValueError("anchor_set_version_mismatch")
        if manifest.get("symbol") != symbol or manifest.get("timeframe") != timeframe:
            raise ValueError("symbol_or_timeframe_mismatch")
        if timestamp_key(manifest.get("genesis_time")) != timestamp_key(genesis_time):
            raise ValueError("genesis_time_mismatch")
        if manifest.get("data_sha256") != data_sha256:
            raise ValueError("data_hash_mismatch")
        if manifest.get("source_hashes") != source_hashes:
            raise ValueError("source_hash_mismatch")
        if manifest.get("engine_config") != config:
            raise ValueError("engine_config_mismatch")

        entry = next(
            (
                item
                for item in manifest.get("anchors", [])
                if timestamp_key(item.get("terrain_anchor_time")) == wanted_anchor
            ),
            None,
        )
        if entry is None:
            raise ValueError("exact_anchor_key_missing")

        anchor_path = _resolve_anchor_path(anchor_root, entry["file"])
        if not anchor_path.is_file():
            raise ValueError("anchor_file_missing")
        if sha256_file(anchor_path) != entry.get("sha256"):
            raise ValueError("anchor_file_hash_mismatch")

        envelope = json.loads(anchor_path.read_text(encoding="utf-8"))
        expected_identity = {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "anchor_set_version": ANCHOR_SET_VERSION,
            "symbol": symbol,
            "timeframe": timeframe,
            "genesis_time": timestamp_key(genesis_time),
            "terrain_anchor_time": wanted_anchor,
        }
        if any(envelope.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("anchor_identity_mismatch")

        checkpoint_probe = canonical_timestamp(envelope["probe_time"])
        if checkpoint_probe > wanted_probe:
            raise ValueError("anchor_is_after_requested_probe")

        reducer = StructureReducer(config, genesis_seed=genesis_seed)
        reducer.load_state(reducer_state_from_envelope(envelope))
        orderflow_state = orderflow_state_from_envelope(envelope)
        orderflow = Orderflow()
        orderflow.load_state(orderflow_state)
        tail = bars.loc[(bars.index > checkpoint_probe) & (bars.index <= wanted_probe)]
        for _, row in tail.iterrows():
            result = reducer.step(row)
            orderflow.step(
                result,
                row,
                shock_ratio=_shock_ratio_for_row(shock_ratios, row),
            )
        return AnchorRestore(
            reducer=reducer,
            orderflow=orderflow,
            used_anchor=True,
            fallback_reason=None,
            processed_through=reducer.cursor["last_known_at"],
        )
    except FileNotFoundError:
        fallback_reason = (
            "manifest_missing" if not manifest_path.is_file() else "anchor_file_missing"
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fallback_reason = str(exc) or type(exc).__name__

    reducer = StructureReducer(config, genesis_seed=genesis_seed)
    orderflow = Orderflow()
    prefix = bars.loc[bars.index <= wanted_probe]
    for _, row in prefix.iterrows():
        result = reducer.step(row)
        orderflow.step(
            result,
            row,
            shock_ratio=_shock_ratio_for_row(shock_ratios, row),
        )
    return AnchorRestore(
        reducer=reducer,
        orderflow=orderflow,
        used_anchor=False,
        fallback_reason=fallback_reason,
        processed_through=reducer.cursor["last_known_at"],
    )


def _shock_ratio_for_row(
    shock_ratios: pd.Series | None, row: pd.Series
) -> float | None:
    if shock_ratios is None:
        return None
    open_time = pd.Timestamp(row.get("timestamp", row.name)).normalize()
    if open_time in shock_ratios.index:
        value = shock_ratios.loc[open_time]
    else:
        value = None
        for key, candidate in shock_ratios.items():
            if pd.Timestamp(key).normalize() == open_time:
                value = candidate
                break
    if value is None or pd.isna(value):
        return None
    return float(value)
