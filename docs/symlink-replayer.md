# Use the Replayer from Another Project

Symlink this repo's `src/` into another project to reuse the replayer engine
without copying it.

## 1. Link `src`

From the other project's root:

```bash
REPLAYER=~/Documents/git-no-drive/py-historical-bar-replayer   # your real checkout path
ln -s "$REPLAYER/src" ./src
ls src/bar_replayer   # must list files; "No such file" means the link target is wrong
```

Replace a wrong link with `ln -sfn "$REPLAYER/src" ./src`.

Link the whole folder — `bar_replayer` imports its sibling packages (`smc`,
`ta`, `quant`, `fusion`, `hypothesis`, `order_plan`) by top-level name.

## 2. Run

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m bar_replayer.app
```

## Config and data

No need to link `config/` or `data/`. The engine follows the symlink back to
this repo, so it reads this repo's `config/config.yaml` and resolves `data.root`
against this repo.

To read the other project's parquet files instead, point the data root there:

```bash
BAR_REPLAYER_DATA_ROOT=/path/to/other-project/data \
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m bar_replayer.app
```

The startup line `Bar replayer data root: ...` shows which folder is in use.
