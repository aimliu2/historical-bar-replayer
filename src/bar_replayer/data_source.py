from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


TIMEFRAME_ORDER = ("1d", "4h", "1h", "15m", "5m", "1m")
TIMEFRAME_DURATIONS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
OHLC_COLUMNS = ("open", "high", "low", "close")
_DATA_FILENAME = re.compile(r"^(?P<symbol>.+)-sorted-(?P<timeframe>1m|5m|15m|1h|4h|1d)\.parquet$", re.I)
_SAFE_SYMBOL = re.compile(r"^[A-Z0-9._-]+$")


class DataSourceError(ValueError):
    """A readable error for missing or malformed replay data."""


class DataFileNotFound(DataSourceError):
    """The selected symbol/timeframe parquet file does not exist."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root(root: str | Path | None = None) -> Path:
    """Resolve the data root without depending on the caller's working directory."""
    configured = root if root is not None else os.environ.get("BAR_REPLAYER_DATA_ROOT")
    if configured is None or not str(configured).strip():
        return (repository_root() / "0_data").resolve()

    if str(configured).strip() == "@0_data":
        return (repository_root() / "0_data").resolve()

    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = repository_root() / path
    return path.resolve()


def available_symbols(root: str | Path | None = None) -> list[str]:
    data_root = resolve_data_root(root)
    if not data_root.is_dir():
        return []

    symbols: set[str] = set()
    for symbol_dir in data_root.iterdir():
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name.upper()
        for parquet_path in symbol_dir.glob("*-sorted-*.parquet"):
            match = _DATA_FILENAME.fullmatch(parquet_path.name)
            if match and match.group("symbol").upper() == symbol:
                symbols.add(symbol)
                break
    return sorted(symbols)


def available_timeframes(symbol: str, root: str | Path | None = None) -> list[str]:
    resolved_symbol = str(symbol).strip().upper()
    if not resolved_symbol:
        return []
    data_root = resolve_data_root(root)
    symbol_dir = data_root / resolved_symbol
    if not symbol_dir.is_dir():
        return []

    found: set[str] = set()
    for parquet_path in symbol_dir.glob("*-sorted-*.parquet"):
        match = _DATA_FILENAME.fullmatch(parquet_path.name)
        if (
            match
            and match.group("symbol").upper() == resolved_symbol
            and match.group("timeframe").lower() in TIMEFRAME_DURATIONS
        ):
            found.add(match.group("timeframe").lower())
    return [timeframe for timeframe in TIMEFRAME_ORDER if timeframe in found]


def parquet_path(symbol: str, timeframe: str, root: str | Path | None = None) -> Path:
    resolved_symbol = str(symbol).strip().upper()
    resolved_timeframe = str(timeframe).strip().lower()
    if not resolved_symbol:
        raise DataSourceError("Choose a symbol.")
    if resolved_symbol in {".", ".."} or not _SAFE_SYMBOL.fullmatch(resolved_symbol):
        raise DataSourceError("Symbol may contain only letters, numbers, dots, underscores, and hyphens.")
    if resolved_timeframe not in TIMEFRAME_DURATIONS:
        supported = ", ".join(TIMEFRAME_DURATIONS)
        raise DataSourceError(f"Unsupported timeframe '{timeframe}'. Choose one of: {supported}.")

    return (
        resolve_data_root(root)
        / resolved_symbol
        / f"{resolved_symbol}-sorted-{resolved_timeframe}.parquet"
    )


def load_bars(
    symbol: str,
    timeframe: str,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Load one parquet file and return sorted UTC bar-close OHLC rows."""
    resolved_symbol = str(symbol).strip().upper()
    resolved_timeframe = str(timeframe).strip().lower()
    path = parquet_path(resolved_symbol, resolved_timeframe, root)
    if not path.is_file():
        data_root = path.parents[1]
        raise DataFileNotFound(
            f"No data file for {resolved_symbol} {resolved_timeframe.upper()} at {path}. "
            f"Check the 0_data link or set BAR_REPLAYER_DATA_ROOT (currently resolved to {data_root})."
        )

    try:
        raw = pd.read_parquet(path)
    except Exception as exc:
        raise DataSourceError(f"Could not read parquet file {path}: {exc}") from exc

    if raw.empty:
        raise DataSourceError(f"The parquet file is empty: {path}")

    timestamp_column = next(
        (column for column in raw.columns if str(column).strip().lower() == "timestamp"),
        None,
    )
    if timestamp_column is not None:
        timestamps = pd.to_datetime(raw[timestamp_column], utc=True, errors="coerce")
    elif isinstance(raw.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(raw.index, utc=True, errors="coerce")
    else:
        raise DataSourceError(
            f"{path} needs a 'timestamp' column or a DatetimeIndex containing bar-open times."
        )

    if timestamps.isna().any():
        bad_rows = int(timestamps.isna().sum())
        raise DataSourceError(f"{path} contains {bad_rows} invalid or missing timestamp value(s).")

    columns_by_name: dict[str, object] = {}
    for column in raw.columns:
        normalized = str(column).strip().lower()
        if normalized in OHLC_COLUMNS:
            if normalized in columns_by_name:
                raise DataSourceError(f"{path} contains more than one '{normalized}' column.")
            columns_by_name[normalized] = column

    missing_columns = [column for column in OHLC_COLUMNS if column not in columns_by_name]
    if missing_columns:
        raise DataSourceError(
            f"{path} is missing required OHLC column(s): {', '.join(missing_columns)}."
        )

    prices = pd.DataFrame(
        {
            column: pd.to_numeric(raw[columns_by_name[column]], errors="coerce")
            for column in OHLC_COLUMNS
        }
    )
    finite_rows = prices.notna().all(axis=1) & ~prices.isin([float("inf"), float("-inf")]).any(axis=1)
    if not finite_rows.all():
        bad_rows = int((~finite_rows).sum())
        raise DataSourceError(f"{path} contains {bad_rows} missing or non-finite OHLC row(s).")

    prices.index = pd.DatetimeIndex(timestamps) + TIMEFRAME_DURATIONS[resolved_timeframe]
    prices.index.name = "bar_close_time"
    prices = prices.sort_index(kind="stable")

    if prices.index.has_duplicates:
        duplicate_count = int(prices.index.duplicated(keep=False).sum())
        raise DataSourceError(f"{path} contains {duplicate_count} rows with duplicate bar-close times.")

    invalid_ohlc = (
        (prices["high"] < prices[["open", "close", "low"]].max(axis=1))
        | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        bad_rows = int(invalid_ohlc.sum())
        raise DataSourceError(
            f"{path} contains {bad_rows} invalid OHLC row(s): high must contain open/close/low "
            "and low must contain open/close/high."
        )

    return prices.astype(float)
