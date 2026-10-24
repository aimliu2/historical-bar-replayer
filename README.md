# py-historical-bar-replayer

Step through historical MT5 bars in the browser, one bar at a time.

## Quick start

Run everything from the repo root.

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Convert MT5 CSV exports to parquet**

```bash
python3 tools/parse_all.py -i ./test_data -o ./data
```

CSV names must follow `{SYMBOL}_{TIMEFRAME}_{anything}.csv` (e.g.
`EURUSD_Daily_201901020000_202604160000.csv`). Output lands in
`data/{SYMBOL}/{SYMBOL}-sorted-{tf}.parquet`.
→ Details: [docs/convert-mt5-csv.md](docs/convert-mt5-csv.md)

**3. Check the config**

`config/config.yaml` sets the defaults — single mode, EURUSD, daily bars from `data/`:

```yaml
data:
  root: data
replay:
  mode: single
  symbol: EURUSD
  single:
    timeframe: 1d
```

Dual mode needs `1h` + `5m` data; tri mode needs `1h` + `5m` + `1d`.

**4. Start the replayer**

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m bar_replayer.app
```

Open <http://127.0.0.1:5055>. Stop with `Ctrl+C`.
→ Options and troubleshooting: [docs/start-replayer.md](docs/start-replayer.md)

> **Using it from another project?** Symlink the whole `src/` folder:
> `ln -s ~/Documents/git-no-drive/py-historical-bar-replayer/src ./src` (use your
> real checkout path). Config and data still
> resolve to this repo unless you set `BAR_REPLAYER_DATA_ROOT`.
> → [docs/symlink-replayer.md](docs/symlink-replayer.md)

> **Technical indicators:** `src/ta` provides SMA, EMA, KAMA, ATR, RSI, SuperTrend
> and SuperTrend flip/stability/step-count. They are numba kernels that match the
> System C runtime exactly, so `numba` must be installed.
> → [docs/ta-indicators.md](docs/ta-indicators.md)

---

## Docs

- [Convert MT5 CSV to Parquet](docs/convert-mt5-csv.md) — filename rules, supported timeframes, output columns
- [Start the Bar Replayer](docs/start-replayer.md) — config, mode data requirements, environment variables, troubleshooting
- [Use the Replayer from Another Project](docs/symlink-replayer.md) — symlink `src/`, where config and data resolve
- [TA Indicators](docs/ta-indicators.md) — every indicator in `src/ta`, formulas, `TAContext`
