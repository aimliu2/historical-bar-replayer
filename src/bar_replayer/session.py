from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from bar_replayer.data_source import OHLC_COLUMNS, TIMEFRAME_DURATIONS
from bar_replayer.structure_lane import (
    StructureLane,
    default_1d_engine_config,
    default_1h_engine_config,
    default_5m_engine_config,
)
from fusion.context import context_at as project_context_at


class ReplaySession:
    """A cursor over one symbol/timeframe dataset."""

    def __init__(
        self,
        bars: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        start_time: str | None = None,
        visible_bars: int = 200,
    ) -> None:
        if bars.empty:
            raise ValueError("Cannot create a replay session from an empty dataset.")
        if visible_bars < 1:
            raise ValueError("visible_bars must be a positive integer.")

        self.session_id = uuid4().hex
        self.bars = bars
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.visible_bars = int(visible_bars)
        self.start_index = self._resolve_start_index(start_time)
        self.cursor_index = self.start_index - 1
        self._at_reset_position = True

    def _resolve_start_index(self, start_time: str | None) -> int:
        if start_time is None or not str(start_time).strip():
            return max(0, len(self.bars) - self.visible_bars)

        try:
            timestamp = pd.Timestamp(str(start_time).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "start_time must be a timestamp such as 2024-01-01T12:00:00Z."
            ) from exc
        if pd.isna(timestamp):
            raise ValueError("start_time must be a valid timestamp.")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        index = int(self.bars.index.searchsorted(timestamp, side="left"))
        return min(max(index, 0), len(self.bars) - 1)

    @property
    def done(self) -> bool:
        return self.cursor_index >= len(self.bars) - 1

    def step(self) -> dict:
        if not self.done:
            self.cursor_index += 1
            self._at_reset_position = False
        return self.snapshot()

    def back(self) -> dict:
        if self.cursor_index == self.start_index - 1 and self._at_reset_position:
            return self.snapshot()
        if self.cursor_index > self.start_index:
            self.cursor_index -= 1
        elif self.cursor_index == self.start_index:
            self.cursor_index = self.start_index - 1
            self._at_reset_position = True
        elif self.cursor_index > 0:
            self.cursor_index -= 1
        return self.snapshot()

    def seek(self, bar_index: int) -> dict:
        if isinstance(bar_index, bool) or not isinstance(bar_index, int):
            raise ValueError("bar_index must be an integer.")
        self.cursor_index = min(max(bar_index, 0), len(self.bars) - 1)
        self._at_reset_position = False
        return self.snapshot()

    def reset(self) -> dict:
        self.cursor_index = self.start_index - 1
        self._at_reset_position = True
        return self.snapshot()

    def _serialize_bar(self, bar_index: int) -> dict:
        timestamp = self.bars.index[bar_index]
        row = self.bars.iloc[bar_index]
        return {
            "bar_index": bar_index,
            "time": timestamp.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }

    def snapshot(self) -> dict:
        cursor_time = None
        current_bar = None
        visible = []
        if self.cursor_index >= 0:
            cursor_time = self.bars.index[self.cursor_index].isoformat()
            current_bar = self._serialize_bar(self.cursor_index)
            first_visible = max(0, self.cursor_index - self.visible_bars + 1)
            visible = [
                self._serialize_bar(index)
                for index in range(first_visible, self.cursor_index + 1)
            ]

        return {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "cursor_index": self.cursor_index,
            "cursor_time": cursor_time,
            "current_bar": current_bar,
            "bars": visible,
            "done": self.done,
        }


class DualReplaySession:
    """A shared close-time cursor over the fixed 1H / 5m replay lanes."""

    LOWER_TIMEFRAME = "5m"
    HIGHER_TIMEFRAME = "1h"
    CLOCKS = ("1h", "5m")

    def __init__(
        self,
        lower_bars: pd.DataFrame,
        higher_bars: pd.DataFrame,
        *,
        symbol: str,
        lower_tf: str = LOWER_TIMEFRAME,
        higher_tf: str = HIGHER_TIMEFRAME,
        clock: str = "1h",
        start_time: str | None = None,
        window_bars: dict[str, int] | None = None,
        focus_context_bars: int = 36,
        structure_settings: dict[str, Any] | None = None,
        orderflow_settings: dict[str, Any] | None = None,
        str_defense_settings: dict[str, Any] | None = None,
        ta_settings: dict[str, Any] | None = None,
        anchor_settings: dict[str, Any] | None = None,
        daily_anchor_settings: dict[str, Any] | None = None,
        data_sha256: str | dict[str, str | None] | None = None,
        repository_root: Path | None = None,
        mode: str = "dual",
        daily_bars: pd.DataFrame | None = None,
        daily_lookback_bars: int = 500,
        str_bars_1d: int = 2,
        symbol_code: str | None = None,
        pip_size: float | None = None,
        lanes_reason: str | None = None,
    ) -> None:
        if type(str_bars_1d) is not int or str_bars_1d < 1:
            raise ValueError("str_bars_1d must be a positive integer")
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"dual", "tri"}:
            raise ValueError("DualReplaySession mode must be 'dual' or 'tri'.")
        self.mode = normalized_mode
        self.daily_bars = daily_bars if normalized_mode == "tri" else None
        if self.daily_bars is not None:
            self._validate_bars(self.daily_bars, "1d")
        if isinstance(daily_lookback_bars, bool) or not isinstance(daily_lookback_bars, int) or daily_lookback_bars < 1:
            raise ValueError("daily_lookback_bars must be a positive integer")
        self.daily_lookback_bars = daily_lookback_bars
        if lanes_reason not in {None, "disabled", "no_anchor_set"}:
            raise ValueError("lanes_reason must be null, 'disabled' or 'no_anchor_set'.")
        self.lanes_reason = lanes_reason
        self.lower_tf = str(lower_tf).strip().lower()
        self.higher_tf = str(higher_tf).strip().lower()
        if (self.lower_tf, self.higher_tf) != (self.LOWER_TIMEFRAME, self.HIGHER_TIMEFRAME):
            raise ValueError("Dual replay uses the fixed 1H / 5m timeframe pair.")
        self.clock = self._validate_clock(clock)
        self._validate_bars(lower_bars, "5m")
        self._validate_bars(higher_bars, "1h")

        windows = {"1h": 500, "5m": 1000} if window_bars is None else window_bars
        if not isinstance(windows, dict) or set(windows) != {"1h", "5m"}:
            raise ValueError("dual window_bars must contain exactly the keys '1h' and '5m'.")
        self.window_bars = {}
        for timeframe, value in windows.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"dual window_bars['{timeframe}'] must be a positive integer.")
            self.window_bars[timeframe] = value
        if isinstance(focus_context_bars, bool) or not isinstance(focus_context_bars, int) or focus_context_bars < 0:
            raise ValueError("focus_context_bars must be a non-negative integer.")

        self.session_id = uuid4().hex
        self.symbol = str(symbol).strip().upper()
        if symbol_code is not None and (len(str(symbol_code)) != 3 or not str(symbol_code).isdigit()):
            raise ValueError("symbol_code must contain exactly three digits")
        if pip_size is not None and (not np.isfinite(float(pip_size)) or float(pip_size) <= 0):
            raise ValueError("pip_size must be a positive finite number")
        self.symbol_code = str(symbol_code) if symbol_code is not None else None
        self.pip_size = float(pip_size) if pip_size is not None else None
        self.lower_bars = lower_bars
        self.higher_bars = higher_bars
        self._daily_index = (
            pd.DatetimeIndex(pd.to_datetime(self.daily_bars.index, utc=True))
            if self.daily_bars is not None
            else pd.DatetimeIndex([], tz="UTC")
        )
        self.focus_context_bars = focus_context_bars
        self._lower_index = lower_bars.index
        self._higher_index = higher_bars.index
        self._lower_duration = TIMEFRAME_DURATIONS[self.lower_tf]
        self._higher_duration = TIMEFRAME_DURATIONS[self.higher_tf]
        if self._higher_duration % self._lower_duration != pd.Timedelta(0):
            raise ValueError("The higher timeframe duration must be a whole multiple of the lower timeframe.")
        self._validate_alignment()

        self._overlap_end = min(self._lower_index[-1], self._higher_index[-1])
        self._clock_times = self._index_for_clock(self.clock)
        self.start_pos = self._resolve_start_pos(start_time)
        self.cursor_time = self._clock_times[self.start_pos - 1] if self.start_pos > 0 else None
        self._initial_cursor_time = self.cursor_time
        self._at_reset_position = True
        self._last_revealed = {"lower": 0, "higher": 0}
        structure = structure_settings or {"enabled": True}
        if self.lanes_reason is None and not structure.get("enabled", False):
            self.lanes_reason = "disabled"
        flow = orderflow_settings or {"enabled": True, "tier": {"1h": "ltr"}, "shock": {"1h": None}}
        defense = str_defense_settings or {"enabled": True, "1h": True}
        ta = ta_settings or {
            "enabled": True,
            "ema": {"1h": [5, 9, 50, 100, 200]},
            "atr": {"1h": 14},
            "band": {"1h": {"ema": 200, "mult": 1.5}},
        }
        anchor_config = anchor_settings or {"enabled": True}
        daily_anchor_config = daily_anchor_settings or anchor_config
        self.structure_lanes: dict[str, StructureLane] = {}
        # Preserve the Batch 1 attribute used by existing callers and study tests.
        self.structure_lane: StructureLane | None = None
        self.lower_structure_lane: StructureLane | None = None
        self.daily_structure_lane: StructureLane | None = None
        if structure.get("enabled", False):
            lane_engine_config = default_1h_engine_config()
            lane_engine_config["orderflow"] = {
                "enabled": bool(flow.get("enabled", False)),
                "tier": flow.get("tier", {}).get("1h", "ltr"),
                "shock": flow.get("shock", {}).get("1h"),
            }
            lane_engine_config["str_defense"] = {
                "enabled": bool(defense.get("enabled", False))
                and bool(defense.get("1h", False)),
            }
            lane_engine_config["ta"] = {
                "enabled": bool(ta.get("enabled", False)),
                "ema": ta.get("ema", {}).get("1h", lane_engine_config["ta"]["ema"]),
                "atr": ta.get("atr", {}).get("1h", lane_engine_config["ta"]["atr"]),
                "band": ta.get("band", {}).get("1h", lane_engine_config["ta"]["band"]),
            }
            self.structure_lane = StructureLane(
                "1H",
                self.higher_bars,
                anchors=anchor_config,
                engine_config=lane_engine_config,
                symbol=self.symbol,
                data_sha256=self._lane_data_hash(data_sha256, "1h"),
                repository_root=repository_root,
                lookback_bars=self._structure_lookback(structure, "1h", self.window_bars),
            )
            self.lower_structure_lane = StructureLane(
                "5m",
                self.lower_bars,
                anchors=anchor_config,
                engine_config={
                    **default_5m_engine_config(flow.get("tier", {}).get("5m", "itr")),
                    "orderflow": {
                        "enabled": bool(flow.get("enabled", False)),
                        "tier": flow.get("tier", {}).get("5m", "itr"),
                        "shock": None,
                    },
                },
                symbol=self.symbol,
                data_sha256=self._lane_data_hash(data_sha256, "5m"),
                repository_root=repository_root,
                lookback_bars=self._structure_lookback(structure, "5m", self.window_bars),
            )
            if self.daily_bars is not None:
                daily_engine_config = default_1d_engine_config()
                daily_engine_config["reducer"]["str_bars"] = str_bars_1d
                self.daily_structure_lane = StructureLane(
                    "1D",
                    self.daily_bars,
                    anchors=daily_anchor_config,
                    engine_config=daily_engine_config,
                    symbol=self.symbol,
                    data_sha256=self._lane_data_hash(data_sha256, "1d"),
                    repository_root=repository_root,
                    lookback_bars=self.daily_lookback_bars,
                )
            self.structure_lanes = {"1d": self.daily_structure_lane, "1h": self.structure_lane, "5m": self.lower_structure_lane}
            self.structure_lanes = {key: lane for key, lane in self.structure_lanes.items() if lane is not None}
            self._rebuild_structure_lanes(self.cursor_time)
        self._stream_current: dict[str, dict[str, Any] | None] = {"1d": None, "1h": None, "5m": None}
        self._stream_pending_events = {timeframe: self._empty_event_facts() for timeframe in ("1d", "1h")}
        self._last_contexts: list[dict[str, Any]] = []
        self._last_fusion_context: dict[str, Any] | None = None
        self._reset_stream_projection(self.cursor_time)

    @classmethod
    def _validate_clock(cls, clock: str) -> str:
        normalized = str(clock).strip().lower()
        if normalized not in cls.CLOCKS:
            raise ValueError("clock must be '1h' or '5m'.")
        return normalized

    @staticmethod
    def _lane_data_hash(value: str | dict[str, str | None] | None, timeframe: str) -> str | None:
        if isinstance(value, dict):
            return value.get(timeframe)
        # Backward compatibility for callers that supplied only the Batch 1 hash.
        return value if timeframe == "1h" else None

    def _rebuild_structure_lanes(self, cursor_time: pd.Timestamp | None) -> None:
        for timeframe in ("1d", "1h", "5m"):
            lane = self.structure_lanes.get(timeframe)
            if lane is not None:
                lane.rebuild(cursor_time)

    def _advance_structure_lanes(
        self, cursor_time: pd.Timestamp | None
    ) -> dict[str, dict[str, Any]]:
        facts: dict[str, dict[str, Any]] = {}
        # Advance higher lanes first so the lower-close stream can align their facts.
        for timeframe in ("1d", "1h", "5m"):
            lane = self.structure_lanes.get(timeframe)
            if lane is not None:
                facts[timeframe] = lane.advance(cursor_time)
                facts[timeframe]["structure_tier"] = self._structure_tier(timeframe, lane)
                facts[timeframe]["step_outputs"] = lane._last_advance_outputs
        return facts

    @staticmethod
    def _public_lane_facts(facts: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in facts.items() if key != "step_outputs"}

    @staticmethod
    def _structure_tier(timeframe: str, lane: StructureLane) -> str:
        if timeframe == "1d":
            return "str"
        if timeframe == "1h":
            return "str"
        return str(lane.engine_config.get("orderflow", {}).get("tier", "itr"))

    @staticmethod
    def _structure_lookback(
        structure_settings: dict[str, Any],
        timeframe: str,
        windows: dict[str, int],
    ) -> int:
        value = structure_settings.get("lookback_bars", windows)
        if isinstance(value, dict):
            return int(value.get(timeframe, windows[timeframe]))
        # A legacy scalar config only ever controlled the 1H lane.
        return int(value) if timeframe == "1h" else windows[timeframe]

    @staticmethod
    def _validate_bars(bars: pd.DataFrame, timeframe: str) -> None:
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            raise ValueError(f"Cannot create a dual replay session from an empty {timeframe} dataset.")
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError(f"The {timeframe} dataset must use a DatetimeIndex of bar-close times.")
        if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
            raise ValueError(f"The {timeframe} dataset must have unique, increasing bar-close times.")
        missing = [column for column in OHLC_COLUMNS if column not in bars.columns]
        if missing:
            raise ValueError(f"The {timeframe} dataset is missing OHLC column(s): {', '.join(missing)}.")

    def _validate_alignment(self) -> None:
        """Reject overlapping lane data that does not aggregate from the 5m lane."""
        lower_ns = self._lower_index.asi8
        higher_ns = self._higher_index.asi8
        higher_step = int(self._higher_duration.value)
        lower_step = int(self._lower_duration.value)

        # A 5m close belongs to the first whole-hour close at or after it.
        bucket_ns = ((lower_ns + higher_step - 1) // higher_step) * higher_step
        bucket_index = pd.DatetimeIndex(pd.to_datetime(bucket_ns, utc=True))
        grouped = self.lower_bars.groupby(bucket_index, sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            count=("close", "count"),
        )

        lower_first_open = self._lower_index[0].value - lower_step
        lower_last_close = self._lower_index[-1].value
        fully_covered = [
            pd.Timestamp(bucket_close)
            for bucket_close in grouped.index
            if bucket_close.value - higher_step >= lower_first_open
            and bucket_close.value <= lower_last_close
            and int(grouped.at[bucket_close, "count"]) == int(higher_step // lower_step)
        ]
        if not fully_covered:
            # Sparse feeds are valid. Only compare buckets that contain every 5m
            # close; incomplete hours are delivered on their first available close.
            return

        higher_positions = self._higher_index.get_indexer(fully_covered)
        comparable = [offset for offset, position in enumerate(higher_positions) if position >= 0]
        if not comparable:
            return
        comparable_times = [fully_covered[offset] for offset in comparable]
        expected = grouped.loc[comparable_times, list(OHLC_COLUMNS)].to_numpy(dtype=float)
        actual = self.higher_bars.iloc[higher_positions[comparable]][list(OHLC_COLUMNS)].to_numpy(dtype=float)
        tolerance = np.maximum(1e-12, np.abs(expected) * 1e-10)
        mismatched = np.any(np.abs(expected - actual) > tolerance, axis=1)
        if mismatched.any():
            bucket_close = comparable_times[int(np.flatnonzero(mismatched)[0])]
            raise ValueError(
                f"The 1H OHLC at {bucket_close.isoformat()} does not match the 5m bars aggregated over that hour."
            )

    def _index_for_clock(self, clock: str) -> pd.DatetimeIndex:
        index = self._higher_index if clock == "1h" else self._lower_index
        return index[index <= self._overlap_end]

    def _resolve_start_pos(self, start_time: str | None) -> int:
        if start_time is None or not str(start_time).strip():
            return max(0, len(self._clock_times) - self.window_bars[self.clock])
        try:
            timestamp = pd.Timestamp(str(start_time).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError("start_time must be a timestamp such as 2024-01-01T12:00:00Z.") from exc
        if pd.isna(timestamp):
            raise ValueError("start_time must be a valid timestamp.")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        index = int(self._clock_times.searchsorted(timestamp, side="left"))
        return min(max(index, 0), len(self._clock_times) - 1)

    @property
    def done(self) -> bool:
        return self._next_clock_pos() >= len(self._clock_times)

    def _next_clock_pos(self) -> int:
        if self.cursor_time is None:
            return 0
        return int(self._clock_times.searchsorted(self.cursor_time, side="right"))

    def _bar_end(self, timeframe: str, cursor_time: pd.Timestamp | None) -> int:
        if cursor_time is None:
            return 0
        index = {
            "1d": self._daily_index,
            "1h": self._higher_index,
            "5m": self._lower_index,
        }[timeframe]
        return int(index.searchsorted(cursor_time, side="right"))

    @staticmethod
    def _serialize_range(frame: pd.DataFrame, start: int, end: int) -> list[dict]:
        if end <= start:
            return []
        timestamps = frame.index[start:end]
        values = frame.iloc[start:end][list(OHLC_COLUMNS)].to_numpy(dtype=float, copy=False)
        return [
            {
                "bar_index": start + offset,
                "time": timestamp.isoformat(),
                "open": float(row[0]),
                "high": float(row[1]),
                "low": float(row[2]),
                "close": float(row[3]),
            }
            for offset, (timestamp, row) in enumerate(zip(timestamps, values))
        ]

    def _forming_higher_bar(self, cursor_time: pd.Timestamp | None) -> dict | None:
        bounds = self._forming_higher_bounds(cursor_time)
        if bounds is None:
            return None
        next_higher_pos, next_close, start, end = bounds
        values = self.lower_bars.iloc[start:end][list(OHLC_COLUMNS)]
        return {
            "bar_index": next_higher_pos,
            "time": next_close.isoformat(),
            "open": float(values["open"].iloc[0]),
            "high": float(values["high"].max()),
            "low": float(values["low"].min()),
            "close": float(values["close"].iloc[-1]),
            "forming": True,
        }

    def _forming_higher_bounds(
        self,
        cursor_time: pd.Timestamp | None,
    ) -> tuple[int, pd.Timestamp, int, int] | None:
        if cursor_time is None:
            return None
        next_higher_pos = int(self._higher_index.searchsorted(cursor_time, side="right"))
        if next_higher_pos >= len(self._higher_index):
            return None
        next_close = self._higher_index[next_higher_pos]
        closed_before = int(self._higher_index.searchsorted(cursor_time, side="right")) - 1
        previous_close = (
            self._higher_index[closed_before]
            if closed_before >= 0
            else next_close - self._higher_duration
        )
        start = int(self._lower_index.searchsorted(previous_close, side="right"))
        end = int(self._lower_index.searchsorted(cursor_time, side="right"))
        if end <= start:
            return None
        return next_higher_pos, next_close, start, end

    def _window_start(self, timeframe: str, cursor_time: pd.Timestamp | None) -> int:
        end = self._bar_end(timeframe, cursor_time)
        slots = self.window_bars[timeframe]
        if timeframe == "1h" and self._forming_higher_bounds(cursor_time) is not None:
            slots = max(0, slots - 1)
        return max(0, end - slots)

    def _is_visible_closed_bar(self, timestamp: pd.Timestamp) -> bool:
        for timeframe, index in (("5m", self._lower_index), ("1h", self._higher_index)):
            position = int(index.get_indexer([timestamp])[0])
            if position < 0:
                continue
            end = self._bar_end(timeframe, self.cursor_time)
            if self._window_start(timeframe, self.cursor_time) <= position < end:
                return True
        return False

    def _snapshot_bars(self, timeframe: str, cursor_time: pd.Timestamp | None) -> list[dict]:
        frame = self.higher_bars if timeframe == "1h" else self.lower_bars
        end = self._bar_end(timeframe, cursor_time)
        bars = self._serialize_range(frame, max(0, end - self.window_bars[timeframe]), end)
        if timeframe == "1h":
            forming = self._forming_higher_bar(cursor_time)
            if forming is not None:
                bars = bars[-max(0, self.window_bars[timeframe] - 1):] if self.window_bars[timeframe] > 1 else []
                bars.append(forming)
        return bars

    def _focus(self, cursor_time: pd.Timestamp | None) -> dict | None:
        if self.clock != "1h" or cursor_time is None:
            return None
        end = self._bar_end("1h", cursor_time)
        if end <= 0:
            return None
        hour_close = self._higher_index[end - 1]
        return {
            "hour_open": (hour_close - self._higher_duration).isoformat(),
            "hour_close": hour_close.isoformat(),
        }

    def _structure_window_start(
        self, timeframe: str, cursor_time: pd.Timestamp | None
    ) -> pd.Timestamp | None:
        if cursor_time is None or timeframe not in self.structure_lanes:
            return None
        index = {
            "1d": self._daily_index,
            "1h": self._higher_index,
            "5m": self._lower_index,
        }[timeframe]
        start = self._window_start(timeframe, cursor_time)
        end = self._bar_end(timeframe, cursor_time)
        return index[start] if start < end else None

    def _structure_snapshot(
        self, timeframe: str, cursor_time: pd.Timestamp | None
    ) -> dict[str, Any] | None:
        lane = self.structure_lanes.get(timeframe)
        if lane is None:
            return None
        result = lane.facts(window_start=self._structure_window_start(timeframe, cursor_time))
        result["structure_tier"] = self._structure_tier(timeframe, lane)
        result["anchor_restore"] = dict(lane.anchor_restore)
        return result

    def _daily_window_start(self, cursor_time: pd.Timestamp | None) -> pd.Timestamp | None:
        if cursor_time is None:
            return None
        start = self._window_start("1h", cursor_time)
        end = self._bar_end("1h", cursor_time)
        return self._higher_index[start].normalize() if start < end else None

    @staticmethod
    def _empty_event_facts() -> dict[str, list[Any]]:
        return {field: [] for field in ("pivots", "breaks", "wicks", "closed_levels", "str_pivots", "str_line_events")}

    @staticmethod
    def _fact_time(fact: dict[str, Any]) -> pd.Timestamp:
        return pd.Timestamp(fact["known_at"]).tz_convert("UTC")

    @staticmethod
    def _events_from_fact(fact: dict[str, Any]) -> dict[str, list[Any]]:
        return {
            "pivots": list(fact.get("pivots") or []),
            "breaks": list(fact.get("breaks") or []),
            "wicks": list(fact.get("wicks") or []),
            "closed_levels": list(fact.get("closed_levels") or []),
            "str_pivots": list(fact.get("str_pivots") or []),
            "str_line_events": list(fact.get("str_line_events") or []),
        }

    def _lane_frame(self, timeframe: str) -> pd.DataFrame | None:
        lane = self.structure_lanes.get(timeframe)
        return lane.bar_facts_frame() if lane is not None else None

    def _project_fusion_context(
        self,
        timestamp: pd.Timestamp,
        *,
        event_facts: dict[str, dict[str, list[Any]]] | None = None,
        fact_frames: dict[str, pd.DataFrame | None] | None = None,
    ) -> dict[str, Any]:
        lower_position = int(self._lower_index.get_indexer([timestamp])[0])
        if lower_position < 0:
            raise ValueError("Fusion context requires a matching closed 5m bar")
        bar = self.lower_bars.iloc[lower_position].to_dict()
        bar["known_at"] = timestamp
        frames = fact_frames if fact_frames is not None else {}
        return project_context_at(
            timestamp,
            asset=self.symbol,
            symbol_code=self.symbol_code,
            pip_size=self.pip_size,
            bar_5m=bar,
            daily_rows=frames.get("1d") if "1d" in frames else self._lane_frame("1d"),
            one_hour_rows=frames.get("1h") if "1h" in frames else self._lane_frame("1h"),
            five_minute_rows=frames.get("5m") if "5m" in frames else self._lane_frame("5m"),
            event_facts=event_facts,
        )

    @staticmethod
    def _append_fact_outputs(
        base: pd.DataFrame | None,
        outputs: list[dict[str, Any]],
    ) -> pd.DataFrame | None:
        if not outputs:
            return base
        addition = pd.DataFrame(deepcopy(outputs))
        index = pd.DatetimeIndex(pd.to_datetime(addition.pop("known_at"), utc=True))
        addition.index = index
        addition.index.name = "known_at"
        if base is None or base.empty:
            return addition.sort_index()
        combined = pd.concat([base, addition])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    def _reset_stream_projection(self, cursor_time: pd.Timestamp | None) -> None:
        self._last_contexts = []
        self._last_fusion_context = None
        self._stream_pending_events = {timeframe: [] for timeframe in ("1d", "1h")}
        self._stream_fact_frames = {
            timeframe: self._lane_frame(timeframe)
            for timeframe in ("1d", "1h", "5m")
        }
        if cursor_time is None or "5m" not in self.structure_lanes:
            self._stream_current = {"1d": None, "1h": None, "5m": None}
            return

        lower_end = int(self._lower_index.searchsorted(cursor_time, side="right"))
        last_lower_time = self._lower_index[lower_end - 1] if lower_end else None
        self._stream_current = {"1d": None, "1h": None, "5m": None}
        for timeframe in ("1d", "1h", "5m"):
            frame = self._lane_frame(timeframe)
            if frame is None or frame.empty:
                continue
            state_time = cursor_time if timeframe == "5m" else last_lower_time
            if state_time is not None:
                position = int(frame.index.searchsorted(state_time, side="right")) - 1
                if position >= 0:
                    self._stream_current[timeframe] = frame.iloc[position].to_dict()

        if last_lower_time is None:
            return
        for timeframe in ("1d", "1h"):
            frame = self._lane_frame(timeframe)
            if frame is None or frame.empty:
                continue
            pending = frame.loc[(frame.index > last_lower_time) & (frame.index <= cursor_time)]
            self._stream_pending_events[timeframe] = [
                {"known_at": index.isoformat(), **row.to_dict()}
                for index, row in pending.iterrows()
            ]

        previous_lower_time = self._lower_index[lower_end - 2] if lower_end > 1 else None
        delivered: dict[str, dict[str, list[Any]]] = {}
        for timeframe in ("1d", "1h"):
            frame = self._lane_frame(timeframe)
            if frame is None or frame.empty:
                continue
            start = frame.index > previous_lower_time if previous_lower_time is not None else frame.index <= last_lower_time
            crossed = frame.loc[start & (frame.index <= last_lower_time)]
            if crossed.empty:
                continue
            payload = self._empty_event_facts()
            for index, row in crossed.iterrows():
                fact = {"known_at": index.isoformat(), **row.to_dict()}
                for field, values in self._events_from_fact(fact).items():
                    payload[field].extend(values)
            delivered[timeframe] = payload
        self._last_fusion_context = self._project_fusion_context(
            last_lower_time,
            event_facts=delivered or None,
            fact_frames=self._stream_fact_frames,
        )

    def _stream_contexts(self, lane_facts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if "5m" not in self.structure_lanes:
            return []
        high_outputs: dict[str, list[dict[str, Any]]] = {}
        for timeframe in ("1d", "1h"):
            high_outputs[timeframe] = list((lane_facts.get(timeframe) or {}).get("step_outputs") or [])
            self._stream_pending_events[timeframe].extend(high_outputs[timeframe])
            self._stream_pending_events[timeframe].sort(key=self._fact_time)
        lower_outputs = list((lane_facts.get("5m") or {}).get("step_outputs") or [])
        fact_frames = dict(self._stream_fact_frames)
        for timeframe, outputs in (
            ("1d", high_outputs["1d"]),
            ("1h", high_outputs["1h"]),
            ("5m", lower_outputs),
        ):
            fact_frames[timeframe] = self._append_fact_outputs(fact_frames.get(timeframe), outputs)
        contexts: list[dict[str, Any]] = []
        for lower_fact in lower_outputs:
            timestamp = self._fact_time(lower_fact)
            delivered: dict[str, dict[str, list[Any]]] = {}
            for timeframe in ("1d", "1h"):
                queue = self._stream_pending_events[timeframe]
                eligible = [fact for fact in queue if self._fact_time(fact) <= timestamp]
                self._stream_pending_events[timeframe] = [
                    fact for fact in queue if self._fact_time(fact) > timestamp
                ]
                if eligible:
                    payload = self._empty_event_facts()
                    for fact in eligible:
                        for field, values in self._events_from_fact(fact).items():
                            payload[field].extend(values)
                    delivered[timeframe] = payload
            context = self._project_fusion_context(
                timestamp,
                event_facts=delivered or None,
                fact_frames=fact_frames,
            )
            contexts.append(context)
            self._stream_current["5m"] = lower_fact
            for timeframe in ("1d", "1h"):
                frame = fact_frames.get(timeframe)
                if frame is not None and not frame.empty:
                    position = int(frame.index.searchsorted(timestamp, side="right")) - 1
                    self._stream_current[timeframe] = frame.iloc[position].to_dict() if position >= 0 else None
        self._last_contexts = contexts
        if contexts:
            self._last_fusion_context = contexts[-1]
        for timeframe, frame in fact_frames.items():
            lane = self.structure_lanes.get(timeframe)
            if lane is not None and frame is not None and not frame.empty:
                window_start = lane._window_start(self.cursor_time)
                if window_start is not None:
                    frame = frame.loc[frame.index >= window_start]
                self._stream_fact_frames[timeframe] = frame
        return contexts

    def _daily_snapshot(self, cursor_time: pd.Timestamp | None) -> dict[str, Any]:
        lane = self.structure_lanes.get("1d")
        if lane is None:
            return {
                "state": "none",
                "reason": "No runtime 1D structure lane is available for this Tri replay.",
                "reason_code": "no_1d_lane",
                "export_path": None,
                "export_sha256": None,
                "coverage": None,
                "as_of": None,
                "rows": [],
            }
        facts = lane.bar_facts_frame()
        if facts.empty or cursor_time is None:
            return {
                "state": "none",
                "reason": "No daily row is known at this cursor.",
                "reason_code": "no_row",
                "export_path": None,
                "export_sha256": None,
                "coverage": None,
                "as_of": None,
                "rows": [],
            }
        position = int(facts.index.searchsorted(cursor_time, side="right")) - 1
        if position < 0:
            return {
                "state": "none",
                "reason": "No daily row is known at this cursor.",
                "reason_code": "no_row",
                "export_path": None,
                "export_sha256": None,
                "coverage": {
                    "first_known_at": self._daily_index[0].isoformat(),
                    "last_known_at": self._daily_index[-1].isoformat(),
                },
                "as_of": None,
                "rows": [],
            }

        def record(row_time: pd.Timestamp, row: pd.Series, changed: bool) -> dict[str, Any]:
            hourly_position = int(self._higher_index.get_indexer([row_time])[0])
            close = float(self.higher_bars.iloc[hourly_position]["close"]) if hourly_position >= 0 else None
            sigma = row.get("vol_sigma_hat")
            sigma = float(sigma) if sigma is not None and pd.notna(sigma) else None
            pips = sigma * close / self.pip_size if sigma is not None and close is not None and self.pip_size else None
            return {
                "known_at": row_time.isoformat(),
                "bar_open": row.get("bar_open"),
                "stage_c": {"status": row.get("of_c0_status"), "exit_path": row.get("of_c0_exit_path")},
                "stage_c1": {"status": row.get("of_c1_status"), "exit_path": row.get("of_c1_exit_path")},
                "stage_c2": {"status": row.get("of_c2_status"), "exit_path": row.get("of_c2_exit_path")},
                "vol_target_day": row.get("vol_target_day"),
                "vol_sigma_hat": sigma,
                "sigma_hat_pips": float(pips) if pips is not None and np.isfinite(pips) else None,
                "state_changed": bool(changed),
            }

        row_time = facts.index[position]
        previous_changed = (
            facts.iloc[position - 1].get("of_c0_status") != facts.iloc[position].get("of_c0_status")
            if position > 0 else False
        )
        all_records = []
        for offset, (timestamp, row) in enumerate(facts.iloc[: position + 1].iterrows()):
            changed = (
                offset > 0
                and facts.iloc[offset - 1].get("of_c0_status") != row.get("of_c0_status")
            )
            all_records.append(record(timestamp, row, changed))
        start = self._daily_window_start(cursor_time)
        visible = [
            item
            for item in all_records
            if start is not None
            and pd.Timestamp(item["known_at"]) - pd.Timedelta(days=1) >= start
        ]
        return {
            "state": "available",
            "reason": None,
            "reason_code": None,
            "export_path": None,
            "export_sha256": None,
            "coverage": {
                "first_known_at": self._daily_index[0].isoformat(),
                "last_known_at": self._daily_index[-1].isoformat(),
            },
            "as_of": record(row_time, facts.iloc[position], previous_changed),
            "rows": visible,
        }

    def _daily_signature(self, cursor_time: pd.Timestamp | None) -> tuple[str, str | None] | None:
        if self.mode != "tri" or "1d" not in self.structure_lanes:
            return None
        if cursor_time is None:
            return "none", None
        position = int(self._daily_index.searchsorted(cursor_time, side="right")) - 1
        return ("available", self._daily_index[position].isoformat()) if position >= 0 else ("none", None)

    def snapshot(self) -> dict:
        result = {
            "mode": self.mode,
            "lanes_reason": self.lanes_reason,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "lower_tf": self.lower_tf,
            "higher_tf": self.higher_tf,
            "clock": self.clock,
            "cursor_time": self.cursor_time.isoformat() if self.cursor_time is not None else None,
            "done": self.done,
            "lower_bars": self._snapshot_bars("5m", self.cursor_time),
            "higher_bars": self._snapshot_bars("1h", self.cursor_time),
            "revealed": dict(self._last_revealed),
            "focus": self._focus(self.cursor_time),
            "window_bars": dict(self.window_bars),
            "focus_context_bars": self.focus_context_bars,
            "higher_facts": self._structure_snapshot("1h", self.cursor_time),
            "lower_facts": self._structure_snapshot("5m", self.cursor_time),
            "fusion_context": self._last_fusion_context,
        }
        if self.mode == "tri":
            result["daily_facts"] = self._daily_snapshot(self.cursor_time)
        return result

    def context_at(
        self,
        timestamp: Any | None = None,
        *,
        daily_rows: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Return the shared Layer 3 → 4 projection at this session's cursor."""
        if self.cursor_time is None:
            raise ValueError("context_at requires a replay cursor")
        at = pd.Timestamp(self.cursor_time if timestamp is None else timestamp)
        at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
        if at != self.cursor_time:
            raise ValueError("context_at can only project the current session cursor")
        lower_position = int(self._lower_index.get_indexer([at])[0])
        if lower_position < 0:
            if timestamp is None and self._last_fusion_context is not None:
                return deepcopy(self._last_fusion_context)
            raise ValueError("context_at requires a matching closed 5m bar")

        for context in reversed(self._last_contexts):
            if context.get("time") == at.isoformat():
                return deepcopy(context)

        if daily_rows is None:
            daily_rows = self._lane_frame("1d")
        hourly_rows = (
            self._lane_frame("1h")
            if "1h" in self.structure_lanes
            else None
        )
        lower_rows = (
            self._lane_frame("5m")
            if "5m" in self.structure_lanes
            else None
        )
        bar = self.lower_bars.iloc[lower_position].to_dict()
        bar["known_at"] = at
        return project_context_at(
            at,
            asset=self.symbol,
            symbol_code=self.symbol_code,
            pip_size=self.pip_size,
            bar_5m=bar,
            daily_rows=daily_rows,
            one_hour_rows=hourly_rows,
            five_minute_rows=lower_rows,
        )

    def step(self, lanes: list[str] | None = None) -> dict:
        if lanes is not None and (
            not isinstance(lanes, list) or any(not isinstance(lane, str) for lane in lanes)
        ):
            raise ValueError("lanes must be a list containing '1h' and/or '5m'.")
        requested_lanes = {"1h", "5m"} if lanes is None else set(lanes)
        if not requested_lanes or not requested_lanes.issubset({"1h", "5m"}):
            raise ValueError("lanes must contain '1h' and/or '5m'.")
        base_cursor = self.cursor_time.isoformat() if self.cursor_time is not None else None
        if self.done:
            self._last_revealed = {"lower": 0, "higher": 0}
            self._last_contexts = []
            lane_facts = self._advance_structure_lanes(self.cursor_time)
            result = self._empty_delta(base_cursor, requested_lanes)
            if "1h" in requested_lanes:
                result["higher"]["forming"] = self._forming_higher_bar(self.cursor_time)
                if "1h" in lane_facts:
                    result["higher"]["facts"] = self._public_lane_facts(lane_facts["1h"])
            if "5m" in requested_lanes and "5m" in lane_facts:
                result["lower"]["facts"] = self._public_lane_facts(lane_facts["5m"])
            result["fusion_context"] = self._last_fusion_context
            return result

        previous_time = self.cursor_time
        previous_daily_signature = self._daily_signature(previous_time)
        before_starts = {
            "lower": self._window_start("5m", previous_time),
            "higher": self._window_start("1h", previous_time),
        }
        cursor_pos = self._next_clock_pos()
        self.cursor_time = self._clock_times[cursor_pos]
        self._at_reset_position = False
        self._last_revealed = {
            "lower": self._bar_end("5m", self.cursor_time) - self._bar_end("5m", previous_time),
            "higher": self._bar_end("1h", self.cursor_time) - self._bar_end("1h", previous_time),
        }
        lane_facts = self._advance_structure_lanes(self.cursor_time)
        fusion_contexts = self._stream_contexts(lane_facts)

        result = self._empty_delta(base_cursor, requested_lanes)
        result["cursor_time"] = self.cursor_time.isoformat()
        result["done"] = self.done
        result["revealed"] = dict(self._last_revealed)
        result["focus"] = self._focus(self.cursor_time)
        result["fusion_context"] = fusion_contexts[-1] if fusion_contexts else self._last_fusion_context
        for timeframe, lane_name in (("1h", "higher"), ("5m", "lower")):
            if timeframe not in requested_lanes:
                continue
            frame = self.higher_bars if timeframe == "1h" else self.lower_bars
            before = int(frame.index.searchsorted(previous_time, side="right")) if previous_time is not None else 0
            after = int(frame.index.searchsorted(self.cursor_time, side="right"))
            appended = self._serialize_range(frame, before, after)
            current_start = self._window_start(timeframe, self.cursor_time)
            lane = {"bars": appended, "drop_left": max(0, current_start - before_starts[lane_name])}
            if timeframe == "1h":
                lane["forming"] = self._forming_higher_bar(self.cursor_time)
            if timeframe in lane_facts:
                lane["facts"] = self._public_lane_facts(lane_facts[timeframe])
            if (
                timeframe == "1h"
                and self.mode == "tri"
                and "1h" in requested_lanes
                and self._daily_signature(self.cursor_time) != previous_daily_signature
            ):
                lane["daily_facts"] = self._daily_snapshot(self.cursor_time)
            result[lane_name] = lane
        return result

    def _empty_delta(self, base_cursor: str | None, lanes: set[str]) -> dict:
        result = {
            "mode": self.mode,
            "clock": self.clock,
            "base_cursor": base_cursor,
            "cursor_time": self.cursor_time.isoformat() if self.cursor_time is not None else None,
            "done": self.done,
            "revealed": dict(self._last_revealed),
            "focus": self._focus(self.cursor_time),
        }
        for timeframe, lane_name in (("1h", "higher"), ("5m", "lower")):
            if timeframe in lanes:
                result[lane_name] = {"bars": [], "drop_left": 0}
                if timeframe == "1h":
                    result[lane_name]["forming"] = None
        return result

    def back(self) -> dict:
        if self._at_reset_position:
            self._last_revealed = {"lower": 0, "higher": 0}
            return self.snapshot()
        pos = int(self._clock_times.searchsorted(self.cursor_time, side="left")) - 1
        if pos < 0:
            self.cursor_time = None
        else:
            self.cursor_time = self._clock_times[pos]
        self._at_reset_position = self.cursor_time == self._initial_cursor_time
        self._last_revealed = {"lower": 0, "higher": 0}
        self._rebuild_structure_lanes(self.cursor_time)
        self._reset_stream_projection(self.cursor_time)
        return self.snapshot()

    def reset(self) -> dict:
        self.cursor_time = self._initial_cursor_time
        self._at_reset_position = True
        self._last_revealed = {"lower": 0, "higher": 0}
        self._rebuild_structure_lanes(self.cursor_time)
        self._reset_stream_projection(self.cursor_time)
        return self.snapshot()

    def set_clock(self, clock: str) -> dict:
        normalized = self._validate_clock(clock)
        self.clock = normalized
        self._clock_times = self._index_for_clock(normalized)
        self._last_revealed = {"lower": 0, "higher": 0}
        return self.snapshot()

    def rewind(self, time_value: str) -> dict:
        try:
            timestamp = pd.Timestamp(time_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("time must be a valid bar-close timestamp.") from exc
        if pd.isna(timestamp):
            raise ValueError("time must be a valid bar-close timestamp.")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        if self.cursor_time is None or timestamp > self.cursor_time:
            raise ValueError("Rewind time must be at or before the current cursor.")
        known = self._lower_index.get_indexer([timestamp])[0] >= 0 or self._higher_index.get_indexer([timestamp])[0] >= 0
        if not known:
            raise ValueError("Rewind time must match a closed 1H or 5m bar.")
        if not self._is_visible_closed_bar(timestamp):
            raise ValueError("Rewind time must match a closed bar currently shown in one of the panes.")
        self.cursor_time = timestamp
        self._at_reset_position = False
        self._last_revealed = {"lower": 0, "higher": 0}
        self._rebuild_structure_lanes(self.cursor_time)
        self._reset_stream_projection(self.cursor_time)
        return self.snapshot()
