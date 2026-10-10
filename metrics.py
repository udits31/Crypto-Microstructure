"""
metrics.py - Spark port of pandas_reference.py, run on the full panel (99 symbols x 132,480 minutes).

    python metrics.py
        -> data/parquet/minute/   per-minute metrics + forward returns (partitioned by symbol)
        -> data/parquet/weekly/   per (symbol, week) aggregates
        -> results/validation.csv Spark vs pandas reference, column by column
        -> results/plans/metrics_*.txt  physical plans (the 'Exchange' lines are the shuffles)

Exits non-zero if Spark disagrees with the pandas reference.

Every time-series operation uses Window.partitionBy("symbol").orderBy("ts_us"):
  - partitionBy("symbol") stops returns being computed across coin boundaries (trap 3)
  - orderBy("ts_us") gives lag/lead a defined order; Spark rows have none otherwise (trap 2)
  - each lag/lead is checked against the clock, so a missing minute can never be jumped (trap 4)
"""
import sys

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from common import (HORIZONS, KLINES_PQ, MINUTE_PQ, MINUTE_US, N_WEEKS, REFERENCE_DIR, RESULTS, STUDY_START_US,
                    WEEK_US, WEEKLY_PQ, get_spark, save_plan, spark_path)

MINUTE_COLS = ["ret", "order_flow", "amihud", "avg_trade_sz"] + [f"fwd_ret_{h}" for h in HORIZONS] + ["week"]
WEEKLY_COLS = ["minutes", "open_first", "close_last", "quote_volume", "week_amihud", "zero_vol_frac",
               "mean_order_flow", "week_ret"]


def build_minute(klines):
    """Per-minute metrics. Same definitions as pandas_reference.minute_metrics."""
    w = Window.partitionBy("symbol").orderBy("ts_us")
    ts, close = F.col("ts_us"), F.col("close")

    def positive(c):
        return F.when(F.col(c) > 0, F.col(c))

    df = klines.select(
        "symbol", "ts_us", F.timestamp_micros("ts_us").alias("ts"),
        "open", "close", "volume", "quote_volume", "num_trades", "taker_buy_base",
        F.when(ts - F.lag(ts).over(w) == MINUTE_US, close / F.lag(close).over(w) - 1).alias("ret"),
        *[F.when(F.lead(ts, h).over(w) - ts == h * MINUTE_US, F.lead(close, h).over(w) / close - 1)
          .alias(f"fwd_ret_{h}") for h in HORIZONS],
    )
    return (df
            .withColumn("order_flow", 2 * F.col("taker_buy_base") / positive("volume") - 1)
            .withColumn("amihud", F.abs("ret") / positive("quote_volume"))
            .withColumn("avg_trade_sz", F.col("volume") / positive("num_trades"))
            .withColumn("week", F.floor((ts - STUDY_START_US) / WEEK_US).cast("long"))
            .drop("taker_buy_base"))


def build_weekly(minute):
    """Per (symbol, week) aggregates. Same definitions as pandas_reference.weekly_metrics."""
    return (minute
            .where(F.col("week") < N_WEEKS)
            .groupBy("symbol", "week")
            .agg(F.min("ts").alias("week_start"),
                 F.count("*").alias("minutes"),
                 F.min_by("open", "ts_us").alias("open_first"),
                 F.max_by("close", "ts_us").alias("close_last"),
                 F.sum("quote_volume").alias("quote_volume"),
                 F.avg("amihud").alias("week_amihud"),              # avg skips nulls: traded minutes only
                 F.avg((F.col("volume") == 0).cast("double")).alias("zero_vol_frac"),
                 F.avg("order_flow").alias("mean_order_flow"))
            .withColumn("week_ret", F.col("close_last") / F.col("open_first") - 1))


def compare(name, spark_df, ref_df, key, cols, rtol):
    """Column-by-column comparison: same keys, same null pattern, values equal within rtol."""
    rows = []
    merged = ref_df.merge(spark_df, on=key, how="outer", suffixes=("_ref", "_spark"), indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    for c in cols:
        a = merged[f"{c}_ref"].to_numpy(dtype="float64")
        b = merged[f"{c}_spark"].to_numpy(dtype="float64")
        null_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.abs(a[both] - b[both])
        scale = np.maximum(np.abs(a[both]), np.abs(b[both]))
        rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
        max_rel = float(rel.max()) if rel.size else 0.0
        rows.append({"table": name, "column": c, "rows": len(merged), "unmatched_keys": unmatched,
                     "null_mismatches": null_mismatch, "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                     "max_rel_diff": max_rel,
                     "ok": unmatched == 0 and null_mismatch == 0 and max_rel <= rtol})
    return rows


def validate(spark):
    """Compare Spark output with every pandas reference symbol found in data/reference/."""
    symbols = sorted(p.name.replace("_minute.parquet", "") for p in REFERENCE_DIR.glob("*_minute.parquet"))
    if not symbols:
        print("no pandas reference files found - run pandas_reference.py first")
        return False
    minute = spark.read.parquet(spark_path(MINUTE_PQ))
    weekly = spark.read.parquet(spark_path(WEEKLY_PQ))
    rows = []
    for sym in symbols:
        s_min = (minute.where(F.col("symbol") == sym).select("ts_us", *MINUTE_COLS).toPandas())
        r_min = pd.read_parquet(REFERENCE_DIR / f"{sym}_minute.parquet")[["ts_us", *MINUTE_COLS]]
        rows += [dict(r, symbol=sym) for r in compare("minute", s_min, r_min, "ts_us", MINUTE_COLS, rtol=1e-12)]

        s_wk = weekly.where(F.col("symbol") == sym).select("week", *WEEKLY_COLS).toPandas()
        r_wk = pd.read_csv(REFERENCE_DIR / f"{sym}_weekly.csv")[["week", *WEEKLY_COLS]]
        # weekly sums/means are accumulated in a different order by Spark: allow float rounding only
        rows += [dict(r, symbol=sym) for r in compare("weekly", s_wk, r_wk, "week", WEEKLY_COLS, rtol=1e-9)]

    report = pd.DataFrame(rows)[["symbol", "table", "column", "rows", "unmatched_keys", "null_mismatches",
                                 "max_abs_diff", "max_rel_diff", "ok"]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS / "validation.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(report.to_string(index=False))
    return bool(report["ok"].all())


def main():
    spark = get_spark("metrics")
    klines = spark.read.parquet(spark_path(KLINES_PQ))

    minute = build_minute(klines)
    save_plan(minute, "metrics_minute")
    minute.write.mode("overwrite").partitionBy("symbol").parquet(spark_path(MINUTE_PQ))

    minute = spark.read.parquet(spark_path(MINUTE_PQ))
    weekly = build_weekly(minute)
    save_plan(weekly, "metrics_weekly")
    weekly.write.mode("overwrite").parquet(spark_path(WEEKLY_PQ))

    print(f"minute rows {minute.count():,}  weekly rows {spark.read.parquet(spark_path(WEEKLY_PQ)).count():,}")
    ok = validate(spark)
    print("VALIDATION PASSED: Spark matches the pandas reference" if ok else "VALIDATION FAILED - see results/validation.csv")
    spark.stop()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
