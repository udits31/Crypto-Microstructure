"""
common.py - paths, constants, the Spark session factory, and Spark-UI metric helpers shared by the
Spark scripts (ingest.py parquet, benchmark.py, metrics.py, analysis.py, skew_scaling.py).
"""
import contextlib
import io
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
META = DATA / "meta"
CSV_DIR = DATA / "csv"
REFERENCE_DIR = DATA / "reference"
PARQUET_DIR = DATA / "parquet"
KLINES_PQ = PARQUET_DIR / "klines"      # cleaned raw klines, partitioned by symbol
MINUTE_PQ = PARQUET_DIR / "minute"      # per-minute metrics + forward returns, partitioned by symbol
WEEKLY_PQ = PARQUET_DIR / "weekly"      # per (symbol, week) aggregates
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PLANS = RESULTS / "plans"
SPARK_TMP = Path("D:/spark-tmp")         # shuffle/spill scratch: C: has too little free space

# ---------------------------------------------------------------- study constants (match pandas_reference.py)

HORIZONS = [1, 5, 30, 60]
N_WEEKS = 13                             # complete 7-day weeks from 2026-07-01; Sep 30 is week 13, excluded
MINUTE_US = 60 * 10**6
WEEK_US = 7 * 24 * 60 * MINUTE_US
STUDY_START_US = 1782864000 * 10**6      # 2026-07-01 00:00:00 UTC

# ---------------------------------------------------------------- chart style (validated reference palette)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light -> dark, ordinal-safe


def chart_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
        "font.family": "DejaVu Sans", "savefig.dpi": 160, "savefig.bbox": "tight",
    })
    return plt


# ---------------------------------------------------------------- Spark

def get_spark(app, cores="*", aqe=True, shuffle_partitions=32, driver_memory="4g"):
    from pyspark.sql import SparkSession
    SPARK_TMP.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder
             .appName(app)
             .master(f"local[{cores}]")
             .config("spark.driver.memory", driver_memory)                 # machine has 7.8 GB RAM
             .config("spark.sql.shuffle.partitions", str(shuffle_partitions))  # default 200 is cluster-tuned
             .config("spark.sql.adaptive.enabled", str(aqe).lower())
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.local.dir", str(SPARK_TMP))
             .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={spark_path(SPARK_TMP)}")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def kline_schema():
    """Explicit schema: no inferSchema pass, and timestamps stay exact 64-bit integers."""
    from pyspark.sql.types import DoubleType, LongType, StructField, StructType
    cols = [("open_time", LongType()), ("open", DoubleType()), ("high", DoubleType()), ("low", DoubleType()),
            ("close", DoubleType()), ("volume", DoubleType()), ("close_time", LongType()),
            ("quote_volume", DoubleType()), ("num_trades", LongType()), ("taker_buy_base", DoubleType()),
            ("taker_buy_quote", DoubleType()), ("ignore", LongType())]
    return StructType([StructField(name, typ, True) for name, typ in cols])


def spark_path(path):
    return Path(path).as_posix()


def save_plan(df, name, mode="formatted"):
    """Write df.explain() output to results/plans/<name>.txt (look for 'Exchange' = shuffle)."""
    PLANS.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain(mode=mode)
    (PLANS / f"{name}.txt").write_text(buf.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------- Spark UI metrics (REST API on localhost:4040)

def _rest(spark, path):
    sc = spark.sparkContext
    r = requests.get(f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def measured(spark, label, action):
    """Run action() as its own job group; return (result, wall_seconds, completed stage records)."""
    sc = spark.sparkContext
    sc.setJobGroup(label, label)
    t0 = time.perf_counter()
    result = action()
    wall = time.perf_counter() - t0

    tracker = sc.statusTracker()
    stage_ids = set()
    for job_id in tracker.getJobIdsForGroup(label):
        info = tracker.getJobInfo(job_id)
        if info is not None:
            stage_ids.update(info.stageIds)

    stages = []
    for sid in sorted(stage_ids):
        for _ in range(40):                 # the UI listener is asynchronous: wait until the stage settles
            try:
                attempts = _rest(spark, f"/stages/{sid}")
            except requests.HTTPError:
                attempts = []
            if attempts and all(a["status"] in ("COMPLETE", "SKIPPED", "FAILED") for a in attempts):
                break
            time.sleep(0.25)
        stages.extend(a for a in attempts if a["status"] == "COMPLETE")
    return result, wall, stages


def stage_tasks(spark, stage):
    """Per-task records for one completed stage: duration and records read."""
    tasks = _rest(spark, f"/stages/{stage['stageId']}/{stage['attemptId']}/taskList?length=100000")
    out = []
    for t in tasks:
        m = t.get("taskMetrics", {})
        out.append({
            "partition": t.get("partitionId", t["index"]),
            "duration_ms": t.get("duration", 0),
            "run_ms": m.get("executorRunTime", 0),
            "shuffle_records": m.get("shuffleReadMetrics", {}).get("recordsRead", 0),
            "input_records": m.get("inputMetrics", {}).get("recordsRead", 0),
        })
    return out


def dir_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
