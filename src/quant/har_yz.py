from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LAGS = {"d": 1, "w": 5, "m": 22}
BOOTSTRAP_END = pd.Timestamp("2020-06-30")
T_MIN = 2.0
YZ_K = 0.34 / 2.34


def _close_index(bars: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("HAR-YZ bars must have a DatetimeIndex of bar-close times")
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("HAR-YZ bar-close times must be unique and increasing")
    index = bars.index
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.c_[np.ones(len(x)), x]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    residual_variance = residuals @ residuals / (len(y) - design.shape[1])
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    return coefficients, covariance, float(residual_variance)


def har_yz_forecasts(
    bars: pd.DataFrame,
    *,
    bootstrap_end: Any = BOOTSTRAP_END,
) -> pd.DataFrame:
    """Return causal one-day-ahead HAR-YZ forecasts, indexed by target bar open.

    Each expanding OLS fit uses only forecast-origin rows preceding its target.
    `sigma_true` is the current target day's observed single-day YZ sigma; the
    resulting ratio is supplied to orderflow at that day's close.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"HAR-YZ bars are missing OHLC columns: {sorted(missing)}")
    if bars.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="target_day"))

    close_index = _close_index(bars)
    open_index = close_index - pd.Timedelta(days=1)
    prices = bars.loc[:, ["open", "high", "low", "close"]].astype(float)
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("HAR-YZ OHLC values must be finite and positive")

    log_open, log_high, log_low, log_close = (
        np.log(prices[column].to_numpy(dtype=float))
        for column in ("open", "high", "low", "close")
    )
    overnight = log_open - np.r_[np.nan, log_close[:-1]]
    oc = log_close - log_open
    rs = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (log_low - log_open)
    yz_var = overnight**2 + YZ_K * oc**2 + (1.0 - YZ_K) * rs
    yz_var[~np.isfinite(yz_var)] = np.nan
    variance = pd.Series(yz_var, index=open_index, dtype=float).dropna()
    log_variance = np.log(variance)

    data = pd.DataFrame({key: log_variance.rolling(period).mean() for key, period in LAGS.items()})
    data["y"] = log_variance.shift(-1)
    data["target_day"] = pd.Series(log_variance.index, index=log_variance.index).shift(-1)
    data["base"] = variance.rolling(LAGS["m"]).mean()
    # Keep the newest feature row even though its next-day target is not known
    # yet. Historical rows below retain the original complete-case selection.
    origins = data.dropna(subset=[*LAGS, "base"]).reset_index(names="origin")
    data = origins.dropna().reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="target_day"))

    boot_end = pd.Timestamp(bootstrap_end)
    boot_end = boot_end.tz_localize("UTC") if boot_end.tzinfo is None else boot_end.tz_convert("UTC")
    boot_end = boot_end.normalize()
    start = int((data["target_day"] <= boot_end).sum())
    # The production EURUSD history has 365 bootstrap rows. Short synthetic
    # lanes still publish their bars, but do not fit an unstable tiny sample.
    if start < 30 or start >= len(data):
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="target_day"))

    features = data[list(LAGS)].to_numpy(dtype=float)
    targets = data["y"].to_numpy(dtype=float)
    target_days = pd.DatetimeIndex(pd.to_datetime(data["target_day"], utc=True))
    output: list[dict[str, Any]] = []
    ones = np.array([0.0, 1.0, 1.0, 1.0])

    def forecast(
        position: int,
        origin: Any,
        x: np.ndarray,
        base: float,
        target_day: Any,
    ) -> dict[str, Any]:
        coefficients, covariance, residual_variance = _ols(
            features[:position], targets[:position]
        )
        mu = coefficients[0] + x @ coefficients[1:]
        sigma_hat = float(np.sqrt(np.exp(mu + residual_variance / 2.0)))
        t_bm = float(coefficients[3] / np.sqrt(covariance[3, 3]))
        t_sum = float(ones @ coefficients / np.sqrt(ones @ covariance @ ones))
        coefficient_sum = float(coefficients[1:].sum())
        gate = bool(
            np.all(coefficients[1:] > 0)
            and coefficient_sum < 1.0
            and t_sum > T_MIN
        )
        return {
            "target_day": target_day,
            "origin": pd.Timestamp(origin),
            "n_fit": position,
            "b0": float(coefficients[0]),
            "bd": float(coefficients[1]),
            "bw": float(coefficients[2]),
            "bm": float(coefficients[3]),
            "t_bm": t_bm,
            "t_sum": t_sum,
            "sigma_hat": sigma_hat,
            "sigma_base": float(np.sqrt(base)),
            "sigma_true": np.nan,
            "shock_ratio": np.nan,
            "sum_b": coefficient_sum,
            "gate": gate,
        }

    for position in range(start, len(data)):
        row = forecast(
            position,
            data.at[position, "origin"],
            features[position],
            float(data.at[position, "base"]),
            target_days[position],
        )
        row["sigma_true"] = float(np.exp(targets[position] / 2.0))
        row["shock_ratio"] = row["sigma_true"] / row["sigma_hat"]
        output.append(row)

    edge_positions = origins.index[origins["target_day"].isna()]
    if len(edge_positions):
        edge_position = int(edge_positions[-1])
        # Training ends at the last origin whose next-day target is realized;
        # the feature at the newest origin is forecast-only.
        if len(origins) == len(data) + 1 and edge_position == len(data) and edge_position >= start:
            edge = origins.iloc[edge_position]
            output.append(
                forecast(
                    edge_position,
                    edge["origin"],
                    edge[list(LAGS)].to_numpy(dtype=float),
                    float(edge["base"]),
                    pd.NaT,
                )
            )
    forecasts = pd.DataFrame(output).set_index("target_day", drop=False)
    forecasts.index.name = "target_day"
    return forecasts
