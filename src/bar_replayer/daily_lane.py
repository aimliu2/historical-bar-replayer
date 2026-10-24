from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bar_replayer.data_source import DataSourceError, repository_root


DAILY_STAGE_COLUMNS = {
    "c": ("of_c0_status", "of_c0_exit_path"),
    "c1": ("of_c1_status", "of_c1_exit_path"),
    "c2": ("of_c2_status", "of_c2_exit_path"),
}
DAILY_REQUIRED_COLUMNS = {
    "known_at",
    "bar_open",
    "vol_target_day",
    "vol_sigma_hat",
    *(column for pair in DAILY_STAGE_COLUMNS.values() for column in pair),
}


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


class DailyFactLane:
    """Read-only as-of projection of the published daily export."""

    def __init__(
        self,
        *,
        export_path: str,
        export_sha256: str | None,
        pip_size: float,
        hourly_bars: pd.DataFrame,
        frame: pd.DataFrame | None = None,
        context_frame: pd.DataFrame | None = None,
        missing_reason: str | None = None,
        missing_reason_code: str | None = None,
    ) -> None:
        self.export_path = str(export_path)
        self.export_sha256 = export_sha256
        self.pip_size = float(pip_size)
        self.hourly_bars = hourly_bars
        self.frame = frame
        self.context_frame = context_frame
        self.missing_reason = missing_reason
        self.missing_reason_code = missing_reason_code

    @classmethod
    def load(
        cls,
        export_path: str | Path,
        *,
        pip_size: float,
        hourly_bars: pd.DataFrame,
        expected_symbol: str | None = None,
        root: Path | None = None,
    ) -> DailyFactLane:
        configured_path = str(export_path)
        path = Path(export_path).expanduser()
        if not path.is_absolute():
            path = (root or repository_root()) / path
        path = path.resolve()
        if not path.is_file():
            return cls(
                export_path=configured_path,
                export_sha256=None,
                pip_size=pip_size,
                hourly_bars=hourly_bars,
                missing_reason=f"Configured 1D export is missing: {configured_path}",
                missing_reason_code="export_missing",
            )

        try:
            payload = path.read_bytes()
            frame = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise DataSourceError(f"Could not read daily export {path}: {exc}") from exc

        missing_columns = sorted(DAILY_REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            raise DataSourceError(
                f"Daily export {path} is missing required column(s): {', '.join(missing_columns)}."
            )
        if frame.empty:
            raise DataSourceError(f"Daily export {path} is empty.")

        if expected_symbol:
            manifest_path = path.with_suffix(".manifest.json")
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise DataSourceError(f"Could not read daily export manifest {manifest_path}: {exc}") from exc
                source_symbol = str(manifest.get("symbol", "")).strip().upper()
                if source_symbol and source_symbol != expected_symbol.strip().upper():
                    return cls(
                        export_path=configured_path,
                        export_sha256=hashlib.sha256(payload).hexdigest(),
                        pip_size=pip_size,
                        hourly_bars=hourly_bars,
                        missing_reason=(
                            f"Configured 1D export is for {source_symbol}, not {expected_symbol.strip().upper()}"
                        ),
                        missing_reason_code="symbol_mismatch",
                    )

        known_at = pd.DatetimeIndex(pd.to_datetime(frame["known_at"], utc=True, errors="coerce"))
        bar_open = pd.DatetimeIndex(pd.to_datetime(frame["bar_open"], utc=True, errors="coerce"))
        if known_at.isna().any() or bar_open.isna().any():
            raise DataSourceError(f"Daily export {path} contains an invalid known_at or bar_open timestamp.")
        if known_at.has_duplicates or not known_at.is_monotonic_increasing:
            raise DataSourceError(f"Daily export {path} known_at values must be unique and increasing.")
        if not (bar_open + pd.Timedelta(days=1)).equals(known_at):
            raise DataSourceError(f"Daily export {path} must have known_at equal to bar_open + one day.")

        sigma = pd.to_numeric(frame["vol_sigma_hat"], errors="coerce")
        if sigma.isna().any() or (~sigma.map(lambda value: pd.notna(value) and abs(float(value)) != float("inf"))).any():
            raise DataSourceError(f"Daily export {path} contains invalid vol_sigma_hat values.")
        if (sigma <= 0).any():
            raise DataSourceError(f"Daily export {path} vol_sigma_hat values must be positive.")

        selected = pd.DataFrame(index=known_at)
        selected.index.name = "known_at"
        selected["bar_open"] = bar_open
        selected["vol_target_day"] = frame["vol_target_day"].astype(str).to_numpy()
        selected["vol_sigma_hat"] = sigma.astype(float).to_numpy()
        for fields in DAILY_STAGE_COLUMNS.values():
            for column in fields:
                selected[column] = frame[column].where(frame[column].notna(), None).to_numpy()
        selected["state_changed"] = selected["of_c0_status"].ne(
            selected["of_c0_status"].shift()
        )
        selected.iloc[0, selected.columns.get_loc("state_changed")] = False

        context_frame = frame.copy(deep=True)
        context_frame.index = known_at
        context_frame.index.name = "known_at"
        context_frame = context_frame.drop(columns=["known_at"])

        return cls(
            export_path=configured_path,
            export_sha256=hashlib.sha256(payload).hexdigest(),
            pip_size=pip_size,
            hourly_bars=hourly_bars,
            frame=selected,
            context_frame=context_frame,
        )

    @property
    def available(self) -> bool:
        return self.frame is not None

    def _base_payload(self) -> dict[str, Any]:
        coverage = None
        if self.frame is not None:
            coverage = {
                "first_known_at": self.frame.index[0].isoformat(),
                "last_known_at": self.frame.index[-1].isoformat(),
            }
        return {
            "state": "none",
            "reason": self.missing_reason,
            "reason_code": self.missing_reason_code,
            "export_path": self.export_path,
            "export_sha256": self.export_sha256,
            "coverage": coverage,
            "as_of": None,
            "rows": [],
        }

    @staticmethod
    def _stage_record(row: pd.Series, key: str) -> dict[str, str | None]:
        status_column, path_column = DAILY_STAGE_COLUMNS[key]
        status = row[status_column]
        exit_path = row[path_column]
        return {
            "status": str(status) if pd.notna(status) else None,
            "exit_path": str(exit_path) if pd.notna(exit_path) else None,
        }

    def _record(self, known_at: pd.Timestamp, row: pd.Series) -> dict[str, Any]:
        close = None
        if known_at in self.hourly_bars.index:
            close = float(self.hourly_bars.at[known_at, "close"])
        sigma_hat = float(row["vol_sigma_hat"])
        sigma_hat_pips = sigma_hat * close / self.pip_size if close is not None else None
        if not pd.isna(sigma_hat_pips) and not abs(float(sigma_hat_pips)) < float("inf"):
            sigma_hat_pips = None
        return {
            "known_at": known_at.isoformat(),
            "bar_open": pd.Timestamp(row["bar_open"]).isoformat(),
            "stage_c": self._stage_record(row, "c"),
            "stage_c1": self._stage_record(row, "c1"),
            "stage_c2": self._stage_record(row, "c2"),
            "vol_target_day": str(row["vol_target_day"]),
            "vol_sigma_hat": sigma_hat,
            "sigma_hat_pips": float(sigma_hat_pips) if sigma_hat_pips is not None else None,
            "state_changed": bool(row["state_changed"]),
        }

    def signature(self, cursor_time: pd.Timestamp | None) -> tuple[str, str | None]:
        if self.frame is None:
            return "none", None
        if cursor_time is None:
            return "none", None
        cursor = _utc_timestamp(cursor_time)
        first_day = self.frame.index[0].normalize()
        last_day = self.frame.index[-1].normalize()
        if cursor.normalize() < first_day or cursor.normalize() > last_day:
            return "none", None
        position = int(self.frame.index.searchsorted(cursor, side="right")) - 1
        if position < 0:
            return "none", None
        return "available", self.frame.index[position].isoformat()

    def snapshot(
        self,
        cursor_time: pd.Timestamp | None,
        *,
        window_start: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload()
        if self.frame is None:
            return payload
        if cursor_time is None:
            payload["reason"] = "No replay cursor is available yet."
            payload["reason_code"] = "no_cursor"
            return payload

        cursor = _utc_timestamp(cursor_time)
        first_day = self.frame.index[0].normalize()
        last_day = self.frame.index[-1].normalize()
        if cursor.normalize() < first_day or cursor.normalize() > last_day:
            payload["reason"] = (
                f"Cursor is outside the export range {first_day.date()} through {last_day.date()}."
            )
            payload["reason_code"] = "out_of_range"
            return payload

        position = int(self.frame.index.searchsorted(cursor, side="right")) - 1
        if position < 0:
            payload["reason"] = "No daily row is known at this cursor."
            payload["reason_code"] = "no_row"
            return payload
        asof_time = self.frame.index[position]
        if asof_time > cursor:
            raise RuntimeError("Daily as-of lookup exposed a row after the replay cursor.")
        payload.update(
            {
                "state": "available",
                "reason": None,
                "reason_code": None,
                "as_of": self._record(asof_time, self.frame.iloc[position]),
            }
        )

        if window_start is None:
            visible = self.frame.iloc[0:0]
        else:
            start_open = _utc_timestamp(window_start) - pd.Timedelta(hours=1)
            first_day_in_view = start_open.normalize()
            visible = self.frame.loc[
                (self.frame["bar_open"] >= first_day_in_view)
                & (self.frame.index <= cursor)
            ]
        payload["rows"] = [
            self._record(timestamp, row)
            for timestamp, row in visible.iterrows()
        ]
        return payload
