"""
pandas_reference.py - single-machine reference implementation of every metric, one symbol at a time.

This is the ground truth the Spark pipeline (metrics.py) must reproduce to several decimals.
It is deliberately plain pandas: explicit sort, explicit gap checks, no cleverness.

    python pandas_reference.py BTCUSDT TNSRUSDT VIRTUALUSDT
        -> data/reference/{SYMBOL}_minute.parquet   per-minute metrics + forward returns
        -> data/reference/{SYMBOL}_weekly.csv       per-week aggregates used for the quintile sort
        -> prints three hand-check rows per symbol (raw inputs next to computed outputs)

Definitions (shared with metrics.py):
    ret           close / prev_close - 1, only when the previous row is exactly 1 minute earlier
    order_flow    2 * taker_buy_base / volume - 1           (null when volume == 0)
    amihud        |ret| / quote_volume                      (null when quote_volume == 0)
    avg_trade_sz  volume / num_trades                       (null when num_trades == 0)
    fwd_ret_h     close[t+h] / close[t] - 1, only when the row h positions ahead is exactly h minutes later.
                  Order flow of minute t is known at its close, so fwd_ret_h starts strictly after it.
    week          7-day blocks anchored at STUDY_START (Wed 2026-07-01 00:00 UTC). Weeks 0..12 are complete;
                  the leftover day (Sep 30) is week 13 and is excluded from weekly aggregates.
    week_ret      close of the week's last minute / open of its first minute - 1
    week_amihud   mean minute amihud over minutes that traded
    zero_vol_frac share of the week's minutes with volume == 0 (a second illiquidity measure)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "data" / "csv"
OUT_DIR = ROOT / "data" / "reference"

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
STUDY_MONTHS = ["2026-07", "2026-08", "2026-09"]
N_WEEKS = 13
HORIZONS = [1, 5, 30, 60]

# Clock arithmetic is done on integer microseconds (as Spark will do on longs): exact, no float or unit surprises.
MINUTE_US = 60 * 10**6
WEEK_US = 7 * 24 * 60 * MINUTE_US
STUDY_START_US = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp()) * 10**6


def load_symbol(symbol):
    """Concatenate the symbol's monthly CSVs, convert timestamps, sort, drop duplicate minutes."""
    frames = []
    for month in STUDY_MONTHS:
        path = CSV_DIR / f"{symbol}-1m-{month}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, header=None, names=COLUMNS, dtype={"open_time": "int64"}))
    if not frames:
        raise FileNotFoundError(f"no CSVs for {symbol} in {CSV_DIR}")
    df = pd.concat(frames, ignore_index=True)

    # Binance spot klines are microseconds from 2025 on, milliseconds before: normalise per row.
    df["ts_us"] = np.where(df["open_time"] >= 10**15, df["open_time"], df["open_time"] * 1000)
    df["ts"] = pd.to_datetime(df["ts_us"], unit="us", utc=True)

    n = len(df)
    df = df.sort_values("ts_us").drop_duplicates(subset="ts_us", keep="first").reset_index(drop=True)
    df.attrs["duplicates_dropped"] = n - len(df)
    return df


def minute_metrics(df):
    """Per-minute metrics. Every lag/lead is validated against the clock, never trusted to row order."""
    out = pd.DataFrame({"ts": df["ts"], "ts_us": df["ts_us"], "close": df["close"], "volume": df["volume"],
                        "quote_volume": df["quote_volume"], "num_trades": df["num_trades"]})

    prev_ok = df["ts_us"].diff() == MINUTE_US
    out["ret"] = (df["close"] / df["close"].shift(1) - 1).where(prev_ok)

    volume = df["volume"].where(df["volume"] > 0)
    quote_volume = df["quote_volume"].where(df["quote_volume"] > 0)
    num_trades = df["num_trades"].where(df["num_trades"] > 0)
    out["order_flow"] = 2 * df["taker_buy_base"] / volume - 1
    out["amihud"] = out["ret"].abs() / quote_volume
    out["avg_trade_sz"] = df["volume"] / num_trades

    for h in HORIZONS:
        ahead_ok = (df["ts_us"].shift(-h) - df["ts_us"]) == h * MINUTE_US
        out[f"fwd_ret_{h}"] = (df["close"].shift(-h) / df["close"] - 1).where(ahead_ok)

    out["week"] = (df["ts_us"] - STUDY_START_US) // WEEK_US
    return out


def weekly_metrics(df, minute):
    """Per-week aggregates for the cross-sectional illiquidity sort."""
    m = minute[minute["week"] < N_WEEKS].assign(open=df["open"], zero_vol=minute["volume"] == 0)
    g = m.groupby("week")
    weekly = pd.DataFrame({
        "week_start": g["ts"].min(),
        "minutes": g.size(),
        "open_first": g["open"].first(),
        "close_last": g["close"].last(),
        "quote_volume": g["quote_volume"].sum(),
        "week_amihud": g["amihud"].mean(),          # NaN-skipping: mean over traded minutes only
        "zero_vol_frac": g["zero_vol"].mean(),
        "mean_order_flow": g["order_flow"].mean(),
    })
    weekly["week_ret"] = weekly["close_last"] / weekly["open_first"] - 1
    return weekly.reset_index()


def hand_check_rows(df, minute):
    """Row 1 (first computable return), a mid-sample traded minute, and the first zero-volume minute."""
    rows = [1, len(df) // 2]
    zero = np.flatnonzero(df["volume"].to_numpy() == 0)
    if len(zero):
        rows.append(int(zero[0]))
    raw_cols = ["ts", "open_time", "close", "volume", "quote_volume", "num_trades", "taker_buy_base"]
    calc_cols = ["ret", "order_flow", "amihud", "avg_trade_sz", "fwd_ret_1", "fwd_ret_5"]
    for i in rows:
        print(f"\n  row {i}")
        print(f"    prev close     = {df['close'].iloc[i - 1]!r}")
        for c in raw_cols:
            print(f"    {c:<14} = {df[c].iloc[i]!r}")
        print(f"    close[t+1]     = {df['close'].iloc[i + 1]!r}")
        print(f"    close[t+5]     = {df['close'].iloc[i + 5]!r}")
        for c in calc_cols:
            print(f"    -> {c:<11} = {minute[c].iloc[i]!r}")


def run(symbol):
    df = load_symbol(symbol)
    minute = minute_metrics(df)
    weekly = weekly_metrics(df, minute)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    minute.to_parquet(OUT_DIR / f"{symbol}_minute.parquet", index=False)
    weekly.to_csv(OUT_DIR / f"{symbol}_weekly.csv", index=False)

    steps = df["ts_us"].diff().dropna()
    print(f"\n=== {symbol}")
    print(f"  rows {len(df):,}  duplicates dropped {df.attrs['duplicates_dropped']}  "
          f"non-1-minute steps {(steps != MINUTE_US).sum()}")
    print(f"  first {df['ts'].iloc[0]}  last {df['ts'].iloc[-1]}")
    print(f"  zero-volume minutes {(df['volume'] == 0).mean():.1%}  "
          f"null order_flow {minute['order_flow'].isna().mean():.1%}")
    print("  non-null forward returns: " +
          ", ".join(f"h={h}: {minute[f'fwd_ret_{h}'].notna().sum():,}" for h in HORIZONS))
    print(f"  weeks {len(weekly)} (minutes per week: {sorted(int(m) for m in weekly['minutes'].unique())})")
    hand_check_rows(df, minute)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="+")
    for symbol in parser.parse_args().symbols:
        run(symbol)


if __name__ == "__main__":
    main()
