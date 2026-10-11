"""
benchmark.py - CSV vs Parquet read benchmark on the same data, same queries, same Spark settings.

    python benchmark.py
        -> results/benchmark_runs.csv      every timed run
        -> results/benchmark.csv           median per (format, query): wall time, tasks, partitions, bytes scanned
                                           (scanned_mb: CSV file sizes / Parquet column chunks from footers;
                                            spark_input_metric_mb is Spark's own counter, unreliable for Parquet)
        -> results/figures/01_read_benchmark.png
        -> results/plans/benchmark_*.txt   physical plans (PartitionFilters / ReadSchema show the pruning)

Queries:
    count_all       count every row                          - pure scan cost
    flow_by_symbol  mean order flow + total volume per symbol - needs 4 of 12 columns (column pruning)
    one_symbol      count + mean close for BTCUSDT only      - Parquet reads one folder (partition pruning)
"""
import pandas as pd
from pyspark.sql import functions as F

from common import (BLUE, CSV_DIR, FIGURES, KLINES_PQ, ORANGE, RESULTS, TEXT_2, chart_style, dir_bytes,
                    get_spark, kline_schema, measured, save_plan, spark_path)

REPEATS = 5

QUERIES = {
    "count_all": lambda df: df.count(),
    "flow_by_symbol": lambda df: df.groupBy("symbol").agg(
        # try_divide: Spark 4 runs in ANSI mode, where x/0 raises; zero-volume minutes give null instead
        F.avg(2 * F.try_divide(F.col("taker_buy_base"), F.col("volume")) - 1).alias("flow"),
        F.sum("quote_volume").alias("qv")).collect(),
    "one_symbol": lambda df: df.where(F.col("symbol") == "BTCUSDT").agg(
        F.count("*"), F.avg("close")).collect(),
}


# What each query needs, for computing bytes scanned independently of Spark's input metric (which
# under-reports Parquet reads here). count_all on Parquet is answered from file footers alone.
QUERY_COLUMNS = {"count_all": [], "flow_by_symbol": ["taker_buy_base", "volume", "quote_volume"],
                 "one_symbol": ["close"]}
QUERY_SYMBOL = {"one_symbol": "BTCUSDT"}


def scanned_mb(fmt, query):
    """CSV: whole files the scan opens (every byte must be parsed). Parquet: compressed column chunks needed."""
    sym = QUERY_SYMBOL.get(query)
    if fmt == "csv":
        files = CSV_DIR.glob(f"{sym}-1m-*.csv" if sym else "*.csv")
        return sum(f.stat().st_size for f in files) / 1e6
    import pyarrow.parquet as pq
    total = 0
    for f in KLINES_PQ.glob(f"symbol={sym}/*.parquet" if sym else "symbol=*/*.parquet"):
        meta = pq.ParquetFile(f).metadata
        for rg in range(meta.num_row_groups):
            for c in range(meta.num_columns):
                col = meta.row_group(rg).column(c)
                if col.path_in_schema in QUERY_COLUMNS[query]:
                    total += col.total_compressed_size
    return total / 1e6


def read_csv(spark):
    return (spark.read.csv(spark_path(CSV_DIR / "*.csv"), schema=kline_schema(), header=False)
            .withColumn("symbol", F.regexp_extract(F.col("_metadata.file_name"), r"^(.+)-1m-\d{4}-\d{2}\.csv$", 1)))


def read_parquet(spark):
    return spark.read.parquet(spark_path(KLINES_PQ))


def chart(summary):
    plt = chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    queries = list(QUERIES)
    x = range(len(queries))
    for ax, metric, label in [(axes[0], "wall_s", "Median wall time (s)"),
                              (axes[1], "scanned_mb", "Data the query must scan (MB)")]:
        for i, (fmt, color) in enumerate([("csv", ORANGE), ("parquet", BLUE)]):
            vals = [summary.loc[(fmt, q), metric] for q in queries]
            pos = [p + (i - 0.5) * 0.36 for p in x]
            bars = ax.bar(pos, vals, width=0.34, color=color, label=fmt.upper())
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:,.1f}" if v < 100 else f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7, color=TEXT_2, xytext=(0, 2),
                            textcoords="offset points")
        ax.set_xticks(list(x), queries)
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    fig.suptitle("CSV vs Parquet: same 13.1M rows, same queries", x=0.01, y=1.04, ha="left", fontsize=12,
                 fontweight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "01_read_benchmark.png")
    plt.close(fig)


def main():
    spark = get_spark("benchmark")
    spark.range(1000).count()                              # warm up the JVM before timing anything

    readers = {"csv": read_csv, "parquet": read_parquet}
    for fmt, reader in readers.items():
        df = reader(spark)
        save_plan(df.where(F.col("symbol") == "BTCUSDT").select("close"), f"benchmark_one_symbol_{fmt}")
        save_plan(df.groupBy("symbol").agg(F.sum("quote_volume")), f"benchmark_flow_by_symbol_{fmt}")

    runs = []
    for rep in range(REPEATS):
        for query, fn in QUERIES.items():
            for fmt, reader in readers.items():              # interleave formats so OS caching treats both alike
                df = reader(spark)
                partitions = df.rdd.getNumPartitions()
                _, wall, stages = measured(spark, f"{fmt}-{query}-{rep}", lambda: fn(df))
                runs.append({"format": fmt, "query": query, "rep": rep, "wall_s": wall,
                             "scan_partitions": partitions,
                             "tasks": sum(s["numCompleteTasks"] for s in stages),
                             "input_mb": sum(s["inputBytes"] for s in stages) / 1e6,
                             "input_rows": sum(s["inputRecords"] for s in stages)})
                print(f"rep {rep} {query:<15} {fmt:<8} {wall:6.2f}s  tasks {runs[-1]['tasks']:>4}  "
                      f"read {runs[-1]['input_mb']:8.1f} MB")

    runs = pd.DataFrame(runs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS / "benchmark_runs.csv", index=False)
    summary = (runs.groupby(["format", "query"])
               .agg(wall_s=("wall_s", "median"), wall_min_s=("wall_s", "min"),
                    scan_partitions=("scan_partitions", "first"), tasks=("tasks", "median"),
                    input_mb=("input_mb", "median"), input_rows=("input_rows", "median")))
    summary = summary.rename(columns={"input_mb": "spark_input_metric_mb"})
    summary["scanned_mb"] = [scanned_mb(f, q) for f, q in summary.index]
    summary["on_disk_mb"] = [dir_bytes(CSV_DIR if f == "csv" else KLINES_PQ) / 1e6 for f, _ in summary.index]
    summary.round(3).to_csv(RESULTS / "benchmark.csv")
    print(summary.round(2).to_string())
    chart(summary)
    spark.stop()


if __name__ == "__main__":
    main()
