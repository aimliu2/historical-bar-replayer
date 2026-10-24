# Start the Bar Replayer

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare data

The replayer reads parquet files from `data/{SYMBOL}/{SYMBOL}-sorted-{tf}.parquet`.
Create them from MT5 CSV exports — see [convert-mt5-csv.md](convert-mt5-csv.md):

```bash
python3 tools/parse_all.py -i ./test_data -o ./data
```

## 3. Check the config

`config/config.yaml` sets the defaults. For single mode on EURUSD daily:

```yaml
data:
  root: data
replay:
  mode: single
  symbol: EURUSD
  start_time: "2026-01-01"
  single:
    timeframe: 1d
    window_bars: 200
```

| Mode | Needs data |
|---|---|
| `single` | any one timeframe for the symbol |
| `dual` | `1h` and `5m` |
| `tri` | `1h`, `5m`, and `1d` |

## 4. Run

From the repo root:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m bar_replayer.app
```

Expected output:

```
Bar replayer data root: .../py-historical-bar-replayer/data
 * Running on http://127.0.0.1:5055
```

Open <http://127.0.0.1:5055> in a browser. Stop with `Ctrl+C`.

## Options

| Environment variable | Purpose | Default |
|---|---|---|
| `BAR_REPLAYER_HOST` | Bind address | `127.0.0.1` |
| `BAR_REPLAYER_PORT` | Port | `5055` |
| `BAR_REPLAYER_DATA_ROOT` | Override `data.root` from the config | — |

## Troubleshooting

| Error | Cause |
|---|---|
| `Could not read bar replayer config .../config/config.yaml` | Config file missing |
| `The data directory is unavailable` | `data.root` folder doesn't exist — run step 2 |
| `Dual replay requires both 1H and 5M data` | Only daily/4H parquet present; export M5 and H1 from MT5 |
