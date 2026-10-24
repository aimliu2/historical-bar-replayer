"""
Parse MT5 CSVs from a chosen input folder into output/{SYMBOL}/ parquet files.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

TF_MAP = {
    "Daily": "1d",
    "H4": "4h",
    "H1": "1h",
    "M15": "15m",
    "M5": "5m",
    "M1": "1m",
}

def parse_mt5_csv(filepath: Path, timezone_offset_hours: int = 0) -> pd.DataFrame:
    raw = pd.read_csv(filepath, sep='\t', nrows=2)
    has_time = '<TIME>' in list(raw.columns)

    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]

    if has_time:
        df['timestamp'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            format='%Y.%m.%d %H:%M:%S'
        )
    else:
        df['timestamp'] = pd.to_datetime(df['date'], format='%Y.%m.%d')

    if timezone_offset_hours != 0:
        df['timestamp'] = df['timestamp'] - pd.Timedelta(hours=timezone_offset_hours)

    df = df.rename(columns={'tickvol': 'tick_vol'})
    keep = ['timestamp', 'open', 'high', 'low', 'close', 'tick_vol', 'spread']
    df = df[keep].copy()
    df['bar_close_bull'] = df['close'] > df['open']

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['tick_vol'] = df['tick_vol'].astype(int)

    return df.sort_values('timestamp').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i', '--input', type=Path, default=ROOT / 'fx-csv-chart',
        help='Input folder containing MT5 CSV files (default: fx-csv-chart beside this script)',
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=ROOT / 'fx-parquet',
        help='Output folder for parquet files (default: fx-parquet beside this script)',
    )
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f'Input folder does not exist or is not a directory: {args.input}')

    pattern = re.compile(
        rf'(?P<symbol>.+)_(?P<tf>{"|".join(TF_MAP)})_.+\.csv'
    )
    inputs = {}
    for csv_path in sorted(args.input.glob('*.csv')):
        match = pattern.fullmatch(csv_path.name)
        if not match:
            print(f'  SKIP  {csv_path.name} — unrecognized filename or timeframe')
            continue
        key = (match['symbol'], match['tf'])
        if key in inputs:
            parser.error(f'Multiple CSV files for {key[0]} {key[1]}: {inputs[key].name}, {csv_path.name}')
        inputs[key] = csv_path

    if not inputs:
        parser.error(f'No supported MT5 CSV files found in: {args.input}')

    for (symbol, tf_raw), csv_path in inputs.items():
        out_dir = args.output / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{symbol}-sorted-{TF_MAP[tf_raw]}.parquet'
        df = parse_mt5_csv(csv_path)
        df.to_parquet(out_path, index=False)
        print(f'  OK    {symbol} {tf_raw:6s} → {out_path}  ({len(df):,} bars)')


if __name__ == "__main__":
    main()
