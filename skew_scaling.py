"""
skew_scaling.py - partition skew diagnosis/fix, and the scaling curve across core counts.

    python skew_scaling.py skew
        Weekly aggregation of the minute panel, run twice with AQE off so Spark cannot rebalance for us:
          before: repartition(32, "week")            -> only 13 distinct keys for 32 partitions; hash collisions
                                                        put 2-3 whole weeks in some tasks, most tasks get nothing
          after:  repartition(32, "symbol", "week")  -> 1,287 keys spread across all 32 partitions
        Same output both times (groupBy(symbol, week) is satisfied by either partitioning, so no extra shuffle).
        -> results/skew_tasks.csv, results/skew_summary.csv, results/figures/04_skew_tasks.png

    python skew_scaling.py scale
        Runs the full metrics workload (window functions + weekly aggregation) in a fresh Spark per core count.
        -> results/scaling.csv, results/figures/05_scaling.png
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from common import (BLUE, FIGURES, GRID, KLINES_PQ, MINUTE_PQ, N_WEEKS, ORANGE, RESULTS, TEXT, TEXT_2, chart_style,
                    get_spark, measured, save_plan, spark_path, stage_tasks)

SKEW_PARTITIONS = 32
VARIANTS = [("before", ["week"]), ("after", ["symbol", "week"])]
CORES = [1, 2, 4, 8, 12]
SCALE_REPEATS = 2

# ---------------------------------------------------------------- skew


def weekly_job(minute, keys):
    return (minute.where(F.col("week") < N_WEEKS)
            .repartition(SKEW_PARTITIONS, *keys)
            .groupBy("symbol", "week")
            .agg(F.avg("amihud").alias("week_amihud"),
                 F.percentile_approx("amihud", 0.5, 1000).alias("median_amihud"),
                 F.count("*").alias("minutes")))


def cmd_skew():
    spark = get_spark("skew", aqe=False)
    minute = spark.read.parquet(spark_path(MINUTE_PQ))
    tasks, summary = [], []
    for name, keys in VARIANTS:
        job = weekly_job(minute, keys)
        save_plan(job, f"skew_{name}")
        for rep in range(2):                                   # first run warms caches; keep the second
            _, wall, stages = measured(spark, f"skew-{name}-{rep}",
                                       lambda: job.write.format("noop").mode("overwrite").save())
        reduce_stage = max((s for s in stages if s["shuffleReadRecords"] > 0), key=lambda s: s["executorRunTime"])
        t = pd.DataFrame(stage_tasks(spark, reduce_stage)).sort_values("partition")
        t.insert(0, "variant", name)
        tasks.append(t)
        busy = t[t["shuffle_records"] > 0]
        summary.append({"variant": name, "partition_keys": ", ".join(keys), "job_wall_s": wall,
                        "stage_tasks": len(t), "tasks_with_data": len(busy),
                        "max_task_s": t["duration_ms"].max() / 1000, "median_task_s": t["duration_ms"].median() / 1000,
                        "max_task_records": int(t["shuffle_records"].max()),
                        "mean_task_records": float(t["shuffle_records"].mean()),
                        "max_over_mean_records": float(t["shuffle_records"].max() / t["shuffle_records"].mean())})
        print(f"{name:<7} keys={keys}  wall {wall:6.2f}s  tasks with data {len(busy)}/{len(t)}  "
              f"max task {summary[-1]['max_task_s']:.2f}s  max/mean records {summary[-1]['max_over_mean_records']:.2f}")

    tasks, summary = pd.concat(tasks), pd.DataFrame(summary)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(RESULTS / "skew_tasks.csv", index=False)
    summary.to_csv(RESULTS / "skew_summary.csv", index=False)
    chart_skew(tasks, summary)
    spark.stop()


def chart_skew(tasks, summary):
    plt = chart_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharey="row")
    max_s = tasks["duration_ms"].max() / 1000
    bins = np.linspace(0, max_s * 1.05, 16)
    for col, (name, keys) in enumerate(VARIANTS):
        t = tasks[tasks["variant"] == name]
        s = summary.set_index("variant").loc[name]
        top, bottom = axes[0, col], axes[1, col]
        top.bar(t["partition"], t["shuffle_records"] / 1e6, color=ORANGE if name == "before" else BLUE, width=0.8)
        top.set_title(f"{name}: repartition({SKEW_PARTITIONS}, {', '.join(repr(k) for k in keys)})", loc="left")
        top.set_xlabel("Shuffle partition (task)")
        top.grid(axis="x", visible=False)
        bottom.hist(t["duration_ms"] / 1000, bins=bins, color=ORANGE if name == "before" else BLUE,
                    edgecolor=GRID, linewidth=1)
        bottom.set_xlabel("Task duration (s)")
        bottom.text(0.98, 0.92, f"stage wall {s['job_wall_s']:.1f}s\nlongest task {s['max_task_s']:.1f}s\n"
                    f"{int(s['tasks_with_data'])}/{int(s['stage_tasks'])} tasks have data",
                    transform=bottom.transAxes, ha="right", va="top", fontsize=8, color=TEXT)
    axes[0, 0].set_ylabel("Rows per task (millions)")
    axes[1, 0].set_ylabel("Number of tasks")
    fig.suptitle("Partition skew: weekly aggregation of 13M minute rows, AQE off", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "04_skew_tasks.png")
    plt.close(fig)


# ---------------------------------------------------------------- scaling


def cmd_scale_worker(cores):
    from metrics import build_minute, build_weekly
    spark = get_spark(f"scale-{cores}", cores=cores)
    spark.range(1000).count()                                # JVM warm-up, not timed
    runs = []
    for _ in range(SCALE_REPEATS):
        t0 = time.perf_counter()
        minute = build_minute(spark.read.parquet(spark_path(KLINES_PQ)))
        minute.write.format("noop").mode("overwrite").save()
        build_weekly(minute).write.format("noop").mode("overwrite").save()
        runs.append(time.perf_counter() - t0)
    spark.stop()
    print("RESULT " + json.dumps({"cores": cores, "runs": runs}))


def cmd_scale():
    rows = []
    for cores in CORES:
        proc = subprocess.run([sys.executable, __file__, "scale-worker", "--cores", str(cores)],
                              capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-2000:], proc.stderr[-4000:])
            sys.exit(f"scaling run with {cores} cores failed")
        res = json.loads(line[len("RESULT "):])
        rows.append({"cores": cores, "runtime_s": float(np.median(res["runs"])),
                     "runs_s": " ".join(f"{r:.1f}" for r in res["runs"])})
        print(f"{cores:>2} cores: {rows[-1]['runtime_s']:6.1f}s  (runs {rows[-1]['runs_s']})")

    df = pd.DataFrame(rows)
    df["speedup"] = df["runtime_s"].iloc[0] / df["runtime_s"]
    df["efficiency"] = df["speedup"] / df["cores"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "scaling.csv", index=False)
    print(df.round(2).to_string(index=False))
    chart_scaling(df)


def chart_scaling(df):
    plt = chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(df["cores"], df["runtime_s"], color=BLUE, marker="o", markersize=6)
    for c, r in zip(df["cores"], df["runtime_s"]):
        ax1.annotate(f"{r:.0f}s", (c, r), xytext=(0, 7), textcoords="offset points", ha="center",
                     fontsize=8, color=TEXT_2)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(df["cores"], [str(c) for c in df["cores"]])
    ax1.set_xlabel("Cores (local[n])")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Full metrics workload runtime", loc="left")
    ax1.set_ylim(0, df["runtime_s"].max() * 1.15)

    ax2.plot(df["cores"], df["cores"], color=TEXT_2, linestyle="--", linewidth=1.2, label="ideal (linear)")
    ax2.plot(df["cores"], df["speedup"], color=ORANGE, marker="o", markersize=6, label="measured")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(df["cores"], [str(c) for c in df["cores"]])
    ax2.set_yticks(df["cores"], [str(c) for c in df["cores"]])
    ax2.set_xlabel("Cores (local[n])")
    ax2.set_ylabel("Speedup vs 1 core")
    ax2.set_title("Speedup", loc="left")
    ax2.legend(loc="upper left")
    fig.text(0.01, -0.04, f"One machine, {os.cpu_count()} logical cores; 'executors' are threads in local mode. "
             "Shuffle partitions fixed at 32.", fontsize=8, color=TEXT_2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "05_scaling.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["skew", "scale", "scale-worker"])
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()
    if args.command == "skew":
        cmd_skew()
    elif args.command == "scale":
        cmd_scale()
    else:
        cmd_scale_worker(args.cores)


if __name__ == "__main__":
    main()
