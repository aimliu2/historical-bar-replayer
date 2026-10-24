# Convert MT5 CSV to Parquet

`tools/parse_all.py` converts MetaTrader 5 bar exports (tab-separated CSV) into
one parquet file per symbol and timeframe.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run from the repo root:

```bash
python3 tools/parse_all.py -i ./test_data -o ./data
```

| Flag | Meaning | Default |
|---|---|---|
| `-i`, `--input` | Folder containing MT5 CSV files | `tools/fx-csv-chart` |
| `-o`, `--output` | Folder to write parquet files into | `tools/fx-parquet` |

Relative paths resolve against the directory you run the command from.

## Example

Input:

```
test_data/
├── EURUSD_Daily_201901020000_202604160000.csv
└── EURUSD_H4_201901020000_202604160000.csv
```

Console output:

```
  OK    EURUSD Daily  → data/EURUSD/EURUSD-sorted-1d.parquet  (1,890 bars)
  OK    EURUSD H4     → data/EURUSD/EURUSD-sorted-4h.parquet  (11,335 bars)
```

Result:

```
data/
└── EURUSD/
    ├── EURUSD-sorted-1d.parquet
    └── EURUSD-sorted-4h.parquet
```

## Input requirements

File names must follow `{SYMBOL}_{TIMEFRAME}_{anything}.csv`. Supported
timeframes and their output suffix:

| MT5 | Output |
|---|---|
| `Daily` | `1d` |
| `H4` | `4h` |
| `H1` | `1h` |
| `M15` | `15m` |
| `M5` | `5m` |
| `M1` | `1m` |

- Files that don't match are printed as `SKIP` and ignored.
- Two files for the same symbol and timeframe stop the run with an error.
- Columns are the standard MT5 export header: `<DATE> <TIME> <OPEN> <HIGH> <LOW>
  <CLOSE> <TICKVOL> <VOL> <SPREAD>`. Daily exports have no `<TIME>` column.

## Output columns

| Column | Type | Notes |
|---|---|---|
| `timestamp` | datetime | Bar open time, as exported by MT5 |
| `open`, `high`, `low`, `close` | float | |
| `tick_vol` | int | From `<TICKVOL>` |
| `spread` | int | From `<SPREAD>` |
| `bar_close_bull` | bool | `close > open` |

Rows are sorted by `timestamp`. `<VOL>` is dropped.
