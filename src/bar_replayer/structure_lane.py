from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from bar_replayer.data_source import TIMEFRAME_DURATIONS
from smc.anchors import (
    ANCHOR_1D_SET_VERSION,
    ANCHOR_1H_SET_VERSION,
    ANCHOR_1H5M_SET_VERSION,
    build_anchor_envelope,
    reducer_state_from_envelope,
    orderflow_state_from_envelope,
    restore_nearest_anchor,
    sha256_file,
)
from smc.orderflow import Orderflow
from smc.reducer import DEFAULT_ENGINE_CONFIG, GENESIS_SEED, StructureReducer
from smc.str_defense import STRDefense
from ta.ta_context import TAContext
from quant.har_yz import har_yz_forecasts


LANE_SOURCE_FILES = (
    "src/smc/anchors.py",
    "src/smc/orderflow.py",
    "src/smc/pivot_event.py",
    "src/smc/reducer.py",
    "src/smc/str_defense.py",
    "src/smc/structure_breaks.py",
    "src/ta/ta_context.py",
    "src/bar_replayer/structure_lane.py",
    "src/quant/har_yz.py",
)


def lane_source_hashes(root: Path) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in LANE_SOURCE_FILES}


def default_1h_engine_config() -> dict[str, Any]:
    return {
        "reducer": {**DEFAULT_ENGINE_CONFIG, "timeframe": "1H"},
        "bar_duration": "1h",
        "warmup_start": None,
        "orderflow": {"enabled": True, "tier": "ltr", "shock": None},
        "str_defense": {"enabled": True},
        "ta": {
            "enabled": True,
            "ema": [5, 9, 50, 100, 200],
            "atr": 14,
            "band": {"ema": 200, "mult": 1.5},
        },
    }


def default_5m_engine_config(tier: str = "itr") -> dict[str, Any]:
    if tier not in {"itr", "ltr"}:
        raise ValueError("5m orderflow tier must be 'itr' or 'ltr'")
    return {
        "reducer": {**DEFAULT_ENGINE_CONFIG, "timeframe": "5m"},
        "bar_duration": "5min",
        "warmup_start": None,
        "orderflow": {"enabled": True, "tier": tier, "shock": None},
        "str_defense": {"enabled": False},
        "ta": {"enabled": False, "ema": []},
    }


def default_1d_engine_config() -> dict[str, Any]:
    return {
        "reducer": {**DEFAULT_ENGINE_CONFIG, "timeframe": "1D"},
        "bar_duration": "1d",
        "warmup_start": "2020-01-01",
        "orderflow": {"enabled": True, "tier": "itr", "shock": None},
        "str_defense": {
            "enabled": True,
            "orderflow_tier": "ltr",
            "fixed_sides": ["low", "high"],
        },
        "ta": {"enabled": False, "ema": []},
    }


class StructureLane:
    """Independent closed-bar structure and context memory for one timeframe."""

    def __init__(
        self,
        timeframe: str,
        bars: pd.DataFrame,
        *,
        anchors: dict[str, Any] | None = None,
        engine_config: dict[str, Any] | None = None,
        symbol: str = "EURUSD",
        data_sha256: str | None = None,
        repository_root: Path | None = None,
        lookback_bars: int = 500,
    ) -> None:
        self.timeframe = str(timeframe).strip().upper()
        timeframe_key = self.timeframe.lower()
        if timeframe_key not in TIMEFRAME_DURATIONS:
            raise ValueError(f"Unsupported structure-lane timeframe: {timeframe}")
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            raise ValueError(f"Cannot create an empty {self.timeframe} structure lane")
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError("Structure-lane bars must use a DatetimeIndex of bar-close times")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError("Structure-lane bars must have unique, increasing close times")
        missing = [name for name in ("open", "high", "low", "close") if name not in bars.columns]
        if missing:
            raise ValueError(f"Structure-lane bars are missing OHLC column(s): {', '.join(missing)}")
        if isinstance(lookback_bars, bool) or not isinstance(lookback_bars, int) or lookback_bars < 1:
            raise ValueError("lookback_bars must be a positive integer")

        self.symbol = str(symbol).strip().upper()
        self.bars = bars.copy(deep=False)
        self.bars.index = self._utc_index(bars.index)
        self.bars.index.name = bars.index.name or "bar_close_time"
        self.lookback_bars = lookback_bars
        self.repository_root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
        self.data_sha256 = data_sha256
        self.anchors = deepcopy(anchors or {})
        self.engine_config = self._normalize_engine_config(engine_config, self.timeframe)
        self._duration = TIMEFRAME_DURATIONS[timeframe_key]
        self.genesis_seed = str(self.engine_config.get("genesis_seed", GENESIS_SEED))
        self.reducer_config = deepcopy(self.engine_config["reducer"])
        self.reducer = self._new_reducer()
        self.orderflow_by_tier = self._new_orderflows()
        self.orderflow = self._selected_orderflow()
        self.str_defense = STRDefense() if self.engine_config["str_defense"]["enabled"] else None
        self.str_defense_by_side = self._new_fixed_defenses()
        self.ta = self._new_ta()
        self.daily_forecasts = har_yz_forecasts(self.bars) if self.timeframe == "1D" else None
        self._daily_forecasts_by_origin = (
            self.daily_forecasts.set_index("origin", drop=False)
            if self.daily_forecasts is not None and not self.daily_forecasts.empty
            else None
        )
        self._bar_facts: list[dict[str, Any]] = []
        self._last_stepped: pd.Timestamp | None = None
        self.cursor: pd.Timestamp | None = None
        self.anchor_restore: dict[str, Any] = {
            "used_anchor": False,
            "fallback_reason": "not_built",
            "terrain_anchor_time": None,
            "probe_time": None,
        }

    @staticmethod
    def _normalize_engine_config(config: dict[str, Any] | None, timeframe: str) -> dict[str, Any]:
        if config is not None and not isinstance(config, dict):
            raise ValueError("Lane engine config must be a mapping")
        supplied = deepcopy(config or {})
        reducer_keys = set(DEFAULT_ENGINE_CONFIG)
        lane_keys = {"reducer", "bar_duration", "warmup_start", "genesis_seed", "orderflow", "str_defense", "ta"}
        unknown = sorted(set(supplied) - reducer_keys - lane_keys)
        if unknown:
            raise ValueError(f"Unknown lane engine config key '{unknown[0]}'")

        supplied_reducer = supplied.pop("reducer", None)
        if supplied_reducer is not None and not isinstance(supplied_reducer, dict):
            raise ValueError("Lane reducer settings must be a mapping")
        reducer_values = {key: supplied.pop(key) for key in reducer_keys if key in supplied}
        reducer_config = deepcopy(DEFAULT_ENGINE_CONFIG)
        reducer_config.update(reducer_values)
        reducer_config.update(deepcopy(supplied_reducer or {}))
        unknown_reducer = sorted(set(reducer_config) - reducer_keys)
        if unknown_reducer:
            raise ValueError(f"Unknown reducer config key '{unknown_reducer[0]}'")
        reducer_config["timeframe"] = timeframe

        defaults = {
            "1D": default_1d_engine_config,
            "1H": default_1h_engine_config,
        }.get(timeframe, default_5m_engine_config)()
        bar_duration = pd.Timedelta(
            supplied.get("bar_duration", defaults["bar_duration"])
        )
        if pd.isna(bar_duration) or bar_duration <= pd.Timedelta(0):
            raise ValueError("Lane bar_duration must be a positive duration")
        warmup_start = supplied.get("warmup_start", defaults["warmup_start"])
        genesis_seed = supplied.get("genesis_seed", GENESIS_SEED)
        if genesis_seed != GENESIS_SEED:
            raise ValueError("Lane genesis_seed must be 'low'")
        orderflow = supplied.get("orderflow", defaults["orderflow"])
        ta = supplied.get("ta", defaults["ta"])
        str_defense = supplied.get("str_defense", defaults["str_defense"])
        if not isinstance(orderflow, dict) or not isinstance(ta, dict) or not isinstance(str_defense, dict):
            raise ValueError("Lane orderflow, str_defense and ta settings must be mappings")
        unknown_orderflow = sorted(set(orderflow) - {"enabled", "tier", "shock", "shock_threshold"})
        unknown_ta = sorted(set(ta) - {"enabled", "ema", "atr", "band"})
        unknown_defense = sorted(set(str_defense) - {"enabled", "orderflow_tier", "fixed_sides"})
        if unknown_orderflow or unknown_ta or unknown_defense:
            unknown_lane_keys = (
                [f"orderflow.{name}" for name in unknown_orderflow]
                + [f"ta.{name}" for name in unknown_ta]
                + [f"str_defense.{name}" for name in unknown_defense]
            )
            key = min(unknown_lane_keys)
            raise ValueError(f"Unknown lane engine config key '{key}'")

        orderflow_enabled = orderflow.get("enabled", False)
        if not isinstance(orderflow_enabled, bool):
            raise ValueError("Lane orderflow enabled must be a boolean")
        orderflow_tier = orderflow.get("tier", "itr")
        defense_tier = str_defense.get("orderflow_tier", orderflow_tier)
        fixed_sides = str_defense.get("fixed_sides", ())
        shock = orderflow.get("shock")
        if orderflow_tier not in {"itr", "ltr"}:
            raise ValueError("Lane orderflow tier must be 'itr' or 'ltr'")
        if defense_tier not in {"itr", "ltr"}:
            raise ValueError("STR defense orderflow_tier must be 'itr' or 'ltr'")
        if not isinstance(fixed_sides, (list, tuple)) or any(side not in {"high", "low"} for side in fixed_sides):
            raise ValueError("STR defense fixed_sides must contain 'high' and/or 'low'")
        if len(set(fixed_sides)) != len(fixed_sides):
            raise ValueError("STR defense fixed_sides must be unique")
        if shock not in {None, "none"}:
            raise ValueError("Lane orderflow shock must be none")
        if not isinstance(str_defense.get("enabled", False), bool):
            raise ValueError("Lane str_defense enabled must be a boolean")
        if not isinstance(ta.get("enabled", False), bool):
            raise ValueError("Lane ta enabled must be a boolean")
        if str_defense.get("enabled", False) and not orderflow_enabled:
            raise ValueError("STR defense requires enabled orderflow on the same lane")
        if str_defense.get("enabled", False) and timeframe == "1H" and defense_tier != "ltr":
            raise ValueError("1H STR defense is keyed to LTR orderflow")
        if fixed_sides and timeframe != "1D":
            raise ValueError("Fixed-side STR defense is supported only on 1D")
        if timeframe == "5M" and str_defense.get("enabled", False):
            raise ValueError("5m STR defense is not supported")
        if timeframe == "5M" and ta.get("enabled", False):
            raise ValueError("5m TA is not supported")
        raw_ema_spans = ta.get("ema", ()) if ta.get("enabled", False) else ()
        if any(isinstance(span, bool) or not isinstance(span, int) for span in raw_ema_spans):
            raise ValueError("Lane EMA spans must be unique positive integers")
        ema_spans = tuple(raw_ema_spans)
        if any(span < 1 for span in ema_spans) or len(set(ema_spans)) != len(ema_spans):
            raise ValueError("Lane EMA spans must be unique positive integers")
        atr_period = ta.get("atr") if ta.get("enabled", False) else None
        band = ta.get("band") if ta.get("enabled", False) else None
        ta_context = (
            TAContext(ema_spans, atr_period=atr_period, band=band)
            if ta.get("enabled", False)
            else None
        )
        config_normalized = {
            "reducer": reducer_config,
            "bar_duration": str(bar_duration),
            "warmup_start": warmup_start,
            "genesis_seed": genesis_seed,
            "orderflow": {
                "enabled": orderflow_enabled,
                "tier": orderflow_tier,
                "shock": None,
                "shock_threshold": float(orderflow.get("shock_threshold", Orderflow.SHOCK_THRESHOLD)),
            },
            "str_defense": {
                "enabled": bool(str_defense.get("enabled", False)),
                "orderflow_tier": defense_tier,
                "fixed_sides": list(fixed_sides),
            },
            "ta": {
                "enabled": bool(ta.get("enabled", False)),
                "ema": list(ema_spans),
                "atr": ta_context.atr_period if ta_context is not None else None,
                "band": deepcopy(ta_context.band) if ta_context is not None else None,
            },
        }
        return config_normalized

    def _new_ta(self) -> TAContext | None:
        config = self.engine_config["ta"]
        if not config["enabled"]:
            return None
        return TAContext(config["ema"], atr_period=config["atr"], band=config["band"])

    def _new_reducer(self) -> StructureReducer:
        return StructureReducer(
            self.reducer_config,
            genesis_seed=self.genesis_seed,
            bar_duration=self.engine_config["bar_duration"],
            warmup_start=self.engine_config["warmup_start"],
        )

    def _new_orderflow(self) -> Orderflow | None:
        config = self.engine_config["orderflow"]
        if not config["enabled"]:
            return None
        return Orderflow(tier=config["tier"], shock_threshold=config["shock_threshold"])

    def _new_orderflows(self) -> dict[str, Orderflow]:
        config = self.engine_config["orderflow"]
        if not config["enabled"]:
            return {}
        tiers = ("ltr", "itr") if self.timeframe in {"1H", "1D"} else (config["tier"],)
        return {
            tier: Orderflow(tier=tier, shock_threshold=config["shock_threshold"])
            for tier in tiers
        }

    def _selected_orderflow(self) -> Orderflow | None:
        return self.orderflow_by_tier.get(self.engine_config["orderflow"]["tier"])

    def _new_fixed_defenses(self) -> dict[str, STRDefense]:
        config = self.engine_config["str_defense"]
        if not config["enabled"]:
            return {}
        return {side: STRDefense(fixed_side=side) for side in config["fixed_sides"]}

    @staticmethod
    def _utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if index.tz is None:
            return index.tz_localize("UTC")
        return index.tz_convert("UTC")

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @classmethod
    def _naive_utc(cls, value: Any) -> pd.Timestamp:
        return cls._utc(value).tz_localize(None)

    @classmethod
    def _iso_utc(cls, value: Any) -> str | None:
        if value is None:
            return None
        return cls._utc(value).isoformat()

    def _window_start(self, cursor: pd.Timestamp | None) -> pd.Timestamp | None:
        if cursor is None:
            return None
        end = int(self.bars.index.searchsorted(cursor, side="right"))
        start = max(0, end - self.lookback_bars)
        return self.bars.index[start] if start < end else None

    def _anchor_paths(self) -> tuple[Path, Path]:
        root_value = self.anchors.get("root")
        anchor_root = Path(root_value) if root_value else self.repository_root / "export" / self.symbol / "anchor"
        default_set = ANCHOR_1D_SET_VERSION if self.timeframe == "1D" else ANCHOR_1H5M_SET_VERSION
        set_version = str(self.anchors.get("set_version", default_set))
        if not root_value and (
            self.timeframe == "1D" or set_version != "eurusd-1h5m-100-bars-v3"
        ):
            anchor_root = anchor_root / set_version
        if not anchor_root.is_absolute():
            anchor_root = self.repository_root / anchor_root
        manifest_value = self.anchors.get("manifest")
        manifest_name = f"{self.symbol}_1D_manifest.json" if self.timeframe == "1D" else f"{self.symbol}_1H5M_manifest.json"
        manifest_path = Path(manifest_value) if manifest_value else anchor_root / manifest_name
        if not manifest_path.is_absolute():
            manifest_path = self.repository_root / manifest_path
        return anchor_root, manifest_path

    def _source_hashes(self) -> dict[str, str]:
        return lane_source_hashes(self.repository_root)

    def rebuild(self, cursor: Any | None) -> dict[str, Any]:
        self.reducer = self._new_reducer()
        self.orderflow_by_tier = self._new_orderflows()
        self.orderflow = self._selected_orderflow()
        self.str_defense = STRDefense() if self.engine_config["str_defense"]["enabled"] else None
        self.str_defense_by_side = self._new_fixed_defenses()
        self.ta = self._new_ta()
        self._bar_facts = []
        self._last_advance_outputs: list[dict[str, Any]] = []
        self._last_stepped = None
        self.cursor = self._utc(cursor) if cursor is not None else None
        if self.cursor is None:
            self.anchor_restore = {
                "used_anchor": False,
                "fallback_reason": "no_cursor",
                "terrain_anchor_time": None,
                "probe_time": None,
            }
            return self.facts()

        window_start = self._window_start(self.cursor)
        anchor_result = None
        if (
            self.anchors.get("enabled", True)
            and self.data_sha256
            and window_start is not None
        ):
            anchor_root, manifest_path = self._anchor_paths()
            anchor_result = restore_nearest_anchor(
                window_start=window_start,
                probe_time=self.cursor,
                anchor_root=anchor_root,
                manifest_path=manifest_path,
                symbol=self.symbol,
                timeframe=self.timeframe,
                genesis_time=self.bars.index[0],
                data_sha256=self.data_sha256,
                source_hashes=self._source_hashes(),
                config=self.engine_config,
                anchor_set_version=str(
                    self.anchors.get(
                        "set_version",
                        ANCHOR_1D_SET_VERSION if self.timeframe == "1D" else ANCHOR_1H5M_SET_VERSION,
                    )
                ),
            )

        start_after: pd.Timestamp | None = None
        self.anchor_restore = {
            "used_anchor": bool(anchor_result and anchor_result.used_anchor),
            "fallback_reason": (
                anchor_result.fallback_reason if anchor_result is not None else "anchor_unavailable"
            ),
            "terrain_anchor_time": anchor_result.terrain_anchor_time if anchor_result else None,
            "probe_time": anchor_result.probe_time if anchor_result else None,
        }
        if anchor_result is not None and anchor_result.envelope is not None:
            try:
                self._load_anchor(anchor_result.envelope)
                start_after = self._utc(anchor_result.probe_time)
                last_facts = self._anchor_last_facts(anchor_result.envelope)
                if last_facts and start_after >= window_start:
                    self._bar_facts.append(deepcopy(last_facts))
            except (KeyError, TypeError, ValueError) as exc:
                self.reducer = self._new_reducer()
                self.orderflow_by_tier = self._new_orderflows()
                self.orderflow = self._selected_orderflow()
                self.str_defense = STRDefense() if self.engine_config["str_defense"]["enabled"] else None
                self.str_defense_by_side = self._new_fixed_defenses()
                self.ta = self._new_ta()
                self._bar_facts = []
                self._last_stepped = None
                self.anchor_restore = {
                    "used_anchor": False,
                    "fallback_reason": f"anchor_state_invalid:{exc}",
                    "terrain_anchor_time": anchor_result.terrain_anchor_time,
                    "probe_time": anchor_result.probe_time,
                }
                start_after = None

        start = int(self.bars.index.searchsorted(start_after, side="right")) if start_after is not None else 0
        end = int(self.bars.index.searchsorted(self.cursor, side="right"))
        for position in range(start, end):
            self._step_position(position)
        self._last_stepped = self.bars.index[end - 1] if end else start_after
        self._prune_facts(window_start)
        return self.facts(window_start=window_start)

    def _load_anchor(self, envelope: dict[str, Any]) -> None:
        lane = self.timeframe.lower()
        self.reducer.load_state(reducer_state_from_envelope(envelope, timeframe=lane))
        layer3 = envelope.get("layer3_context") or {}
        if lane == "5m":
            if self.orderflow is not None:
                state = orderflow_state_from_envelope(
                    envelope, timeframe="5m", tier=self.orderflow.tier
                )
                self.orderflow.load_state(state)
        else:
            for tier, tracker in self.orderflow_by_tier.items():
                tracker.load_state(
                    orderflow_state_from_envelope(envelope, timeframe=lane, tier=tier)
                )
        if self.str_defense is not None:
            self.str_defense.load_state(layer3["str_defense"])
        if self.str_defense_by_side:
            fixed_states = layer3["str_defense_by_side"]
            for side, defense in self.str_defense_by_side.items():
                defense.load_state(fixed_states[side])
        if self.ta is not None:
            self.ta.load_state(layer3["ta"])

    def _anchor_last_facts(self, envelope: dict[str, Any]) -> dict[str, Any] | None:
        layer3 = envelope.get("layer3_context") or {}
        if self.timeframe == "5M":
            lane_context = layer3.get("5m") or {}
            by_tier = lane_context.get("last_facts_by_tier") or {}
            facts = by_tier.get(self.engine_config["orderflow"]["tier"])
            if facts is not None:
                return deepcopy(facts)
            return deepcopy(lane_context.get("last_facts"))
        return deepcopy(layer3.get("last_facts"))

    def advance(self, cursor: Any) -> dict[str, Any]:
        target = self._utc(cursor)
        if self.cursor is not None and target < self.cursor:
            return self.rebuild(target)
        previous_cursor = self.cursor
        previous = self._last_stepped
        start = int(self.bars.index.searchsorted(previous, side="right")) if previous is not None else 0
        end = int(self.bars.index.searchsorted(target, side="right"))
        outputs: list[dict[str, Any]] = []
        for position in range(start, end):
            outputs.append(self._step_position(position))
        if end:
            self._last_stepped = self.bars.index[end - 1]
        self._last_advance_outputs = deepcopy(outputs)
        self.cursor = target
        self._prune_facts(self._window_start(target))
        return self._collect(
            outputs,
            since=previous_cursor,
            window_start=None,
            include_levels=True,
        )

    def _step_position(self, position: int) -> dict[str, Any]:
        timestamp = self.bars.index[position]
        row = self.bars.iloc[position].copy(deep=False)
        row.name = self._naive_utc(timestamp)
        result = self.reducer.step(row)
        flow_result = None
        flow_snapshot = None
        flow_results: dict[str, dict[str, Any]] = {}
        bar_open = timestamp - self._duration
        target_forecast = self._forecast_by_target(bar_open)
        shock_ratio = (
            float(target_forecast["shock_ratio"])
            if target_forecast is not None and pd.notna(target_forecast.get("shock_ratio"))
            else None
        )
        for tier, tracker in self.orderflow_by_tier.items():
            tier_result = tracker.step(result, row, shock_ratio=shock_ratio)
            flow_results[tier] = tier_result
            if tracker is self.orderflow:
                flow_result = tier_result
                flow_snapshot = tier_result["snapshot"]
        str_snapshot = None
        defense_tier = self.engine_config["str_defense"]["orderflow_tier"]
        defense_flow = (flow_results.get(defense_tier) or {}).get("snapshot") or {}
        if self.str_defense is not None:
            str_snapshot = self.str_defense.step(
                result,
                defense_flow,
                close=float(row["close"]),
                known_at=row.name,
            )
        fixed_snapshots = {
            side: defense.step(
                result,
                {"direction": "bull" if side == "low" else "bear"},
                close=float(row["close"]),
                known_at=row.name,
            )
            for side, defense in self.str_defense_by_side.items()
        }
        ta_snapshot = self.ta.step(row) if self.ta is not None else None

        facts = {
            "known_at": timestamp.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "pivots": [self._pivot_fact(event) for event in result["events"]],
            "breaks": [self._serialize_times(item) for item in result["breaks"]],
            "wicks": [self._serialize_times(item) for item in result["wicks"]],
            "closed_levels": [self._serialize_times(item) for item in result["closed_levels"]],
            "orderflow": self._orderflow_fact(flow_snapshot, flow_result, timestamp),
            "str_defense": (
                {"time": timestamp.isoformat(), **str_snapshot}
                if str_snapshot is not None
                else None
            ),
            "str_defense_by_side": {
                side: {"time": timestamp.isoformat(), **snapshot}
                for side, snapshot in fixed_snapshots.items()
            },
            "ta": {"time": timestamp.isoformat(), **ta_snapshot} if ta_snapshot is not None else None,
        }
        if self.timeframe in {"1H", "1D"}:
            facts["orderflow_by_tier"] = {
                tier: self._orderflow_fact(tier_result["snapshot"], tier_result, timestamp)
                for tier, tier_result in flow_results.items()
        }
        if self.timeframe == "1D":
            facts["str_line_events"] = [
                deepcopy(defense.line_event)
                for defense in self.str_defense_by_side.values()
                if defense.line_event is not None
            ]
            facts["str_pivots"] = [
                pivot for pivot in facts["pivots"] if pivot.get("tier") == "str"
            ]
            facts.update(self._daily_contract(bar_open, flow_snapshot))
        self._bar_facts.append(facts)
        return facts

    def _orderflow_fact(
        self,
        snapshot: dict[str, Any] | None,
        result: dict[str, Any] | None,
        timestamp: pd.Timestamp,
    ) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        fact = self._serialize_times(snapshot)
        fact["time"] = timestamp.isoformat()
        fact["episodes"] = [self._serialize_times(item) for item in (result or {}).get("episodes", [])]
        fact["shock_ratio"] = (result or {}).get("shock_ratio")
        fact["shock"] = (result or {}).get("shock", False)
        return fact

    def _forecast_by_target(self, bar_open: Any) -> pd.Series | None:
        if self.daily_forecasts is None or self.daily_forecasts.empty:
            return None
        target = self._utc(bar_open)
        position = int(self.daily_forecasts.index.get_indexer([target])[0])
        return self.daily_forecasts.iloc[position] if position >= 0 else None

    def _forecast_by_origin(self, bar_open: Any) -> pd.Series | None:
        if self._daily_forecasts_by_origin is None:
            return None
        origin = self._utc(bar_open)
        position = int(self._daily_forecasts_by_origin.index.get_indexer([origin])[0])
        return self._daily_forecasts_by_origin.iloc[position] if position >= 0 else None

    @staticmethod
    def _published_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(f"{float(value):.6g}")

    def _daily_contract(
        self,
        bar_open: pd.Timestamp,
        orderflow_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        snapshot = orderflow_snapshot or {}
        defended = snapshot.get("defended") or {}
        provisional = snapshot.get("provisional_open") or {}
        history = snapshot.get("history") or []
        contract: dict[str, Any] = {
            "bar_open": self._naive_utc(bar_open).isoformat(),
            "of_status": snapshot.get("status"),
            "of_direction": snapshot.get("direction"),
            "of_defended_side": defended.get("side"),
            "of_defended_price": defended.get("price"),
            "of_prov_level": provisional.get("level"),
            "of_prov_opened_by": provisional.get("opened_by"),
            "of_prov_wick_count": provisional.get("wick_count"),
            "of_prov_R": provisional.get("reaffirm_level"),
            "of_prov_P": provisional.get("prev_level"),
        }
        stage_fields = {
            "status": "of_status",
            "since": "of_since",
            "exit_path": "exit_path",
            "swept_side": "swept_side",
            "swept_level": "swept_level",
        }
        for stage in range(3):
            entry = history[stage] if stage < len(history) else {}
            for suffix, source in stage_fields.items():
                value = entry.get(source)
                contract[f"of_c{stage}_{suffix}"] = value

        next_forecast = self._forecast_by_origin(bar_open)
        target_day = next_forecast["target_day"] if next_forecast is not None else None
        if target_day is not None and pd.isna(target_day):
            target_day = None
        contract.update(
            {
                "vol_target_day": target_day.strftime("%Y-%m-%d") if target_day is not None else None,
                "vol_sigma_hat": self._published_float(next_forecast.get("sigma_hat")) if next_forecast is not None else None,
                "vol_sigma_base": self._published_float(next_forecast.get("sigma_base")) if next_forecast is not None else None,
                "vol_gate": bool(next_forecast["gate"]) if next_forecast is not None else None,
            }
        )
        return contract

    @classmethod
    def _pivot_fact(cls, event: Any) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_code": event.event_code,
            "tier": event.tier,
            "side": event.side,
            "relation": event.relation,
            "pivot_ts": cls._iso_utc(event.pivot_timestamp),
            "event_ts": cls._iso_utc(event.event_timestamp),
            "price": float(event.pivot_price),
            "timeframe": event.timeframe,
        }

    @classmethod
    def _serialize_times(cls, value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if item is not None and (
                    key.endswith("_ts")
                    or key in {"known_at", "of_since", "opened_at", "exited_at", "last_known_at", "cursor"}
                ):
                    try:
                        result[key] = cls._iso_utc(item)
                        continue
                    except (TypeError, ValueError):
                        pass
                result[key] = cls._serialize_times(item)
            return result
        if isinstance(value, list):
            return [cls._serialize_times(item) for item in value]
        if isinstance(value, pd.Timestamp):
            return cls._iso_utc(value)
        return value

    def _prune_facts(self, window_start: pd.Timestamp | None) -> None:
        if window_start is None:
            return
        start = self._utc(window_start)
        self._bar_facts = [item for item in self._bar_facts if self._utc(item["known_at"]) >= start]

    def _collect(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        since: Any | None,
        window_start: Any | None,
        include_levels: bool,
    ) -> dict[str, Any]:
        facts = rows if rows is not None else self._bar_facts
        lower = self._utc(window_start) if window_start is not None else None
        after = self._utc(since) if since is not None else None
        selected = [
            item
            for item in facts
            if (lower is None or self._utc(item["known_at"]) >= lower)
            and (after is None or self._utc(item["known_at"]) > after)
        ]
        collected: dict[str, Any] = {
            "pivots": [],
            "breaks": [],
            "wicks": [],
            "levels": [],
            "orderflow": [],
            "str_defense": [],
            "str_defense_by_side": {},
            "ta": [],
        }
        if self.timeframe in {"1H", "1D"}:
            collected["orderflow_by_tier"] = {
                tier: [] for tier in self.orderflow_by_tier
            }
        if self.timeframe == "1D":
            collected["str_defense_by_side"] = {
                side: [] for side in self.str_defense_by_side
            }
        for item in selected:
            collected["pivots"].extend(deepcopy(item["pivots"]))
            collected["breaks"].extend(deepcopy(item["breaks"]))
            collected["wicks"].extend(deepcopy(item["wicks"]))
            collected["levels"].extend(deepcopy(item["closed_levels"]))
            for key in ("orderflow", "str_defense", "ta"):
                if item.get(key) is not None:
                    collected[key].append(deepcopy(item[key]))
            for side, fact in (item.get("str_defense_by_side") or {}).items():
                collected["str_defense_by_side"].setdefault(side, []).append(deepcopy(fact))
            if "orderflow_by_tier" in collected:
                for tier, fact in (item.get("orderflow_by_tier") or {}).items():
                    if fact is not None:
                        collected["orderflow_by_tier"].setdefault(tier, []).append(deepcopy(fact))
        if include_levels:
            collected["levels"].extend(self._active_levels(after))
        return collected

    def _active_levels(
        self,
        since: pd.Timestamp | None,
    ) -> list[dict[str, Any]]:
        rows = [self._serialize_times(item) for item in self.reducer.active_levels()]
        active = []
        for row in rows:
            event_ts = self._utc(row["event_ts"])
            if since is not None and event_ts <= since:
                continue
            active.append(row)
        return active

    def facts(
        self,
        since: Any | None = None,
        window_start: Any | None = None,
    ) -> dict[str, Any]:
        return self._collect(
            since=since,
            window_start=window_start,
            include_levels=True,
        )

    def bar_facts_frame(self) -> pd.DataFrame:
        """Return copied per-close lane facts indexed by their UTC known_at time."""
        if not self._bar_facts:
            empty = pd.DataFrame()
            empty.index = pd.DatetimeIndex([], tz="UTC", name="known_at")
            return empty
        frame = pd.DataFrame(deepcopy(self._bar_facts))
        known_at = pd.DatetimeIndex(pd.to_datetime(frame.pop("known_at"), utc=True))
        frame.index = known_at
        frame.index.name = "known_at"
        return frame

    def anchor_envelope(
        self,
        *,
        terrain_anchor_time: Any,
        anchor_set_version: str | None = None,
    ) -> dict[str, Any]:
        return build_anchor_envelope(
            self.reducer,
            symbol=self.symbol,
            timeframe=self.timeframe,
            genesis_time=self.bars.index[0],
            terrain_anchor_time=terrain_anchor_time,
            orderflow=self.orderflow,
            orderflow_by_tier=self.orderflow_by_tier,
            str_defense=self.str_defense,
            str_defense_by_side=self.str_defense_by_side,
            ta_context=self.ta,
            anchor_set_version=anchor_set_version
            or (ANCHOR_1D_SET_VERSION if self.timeframe == "1D" else ANCHOR_1H_SET_VERSION),
            extra_context={"last_facts": self._bar_facts[-1] if self._bar_facts else None},
        )

    @property
    def last_stepped(self) -> pd.Timestamp | None:
        return self._last_stepped
