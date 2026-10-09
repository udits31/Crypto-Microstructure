"""
ingest.py - universe selection and parallel download of Binance spot 1-minute klines.

Run from the bda venv, in two phases so the universe can be reviewed before the bulk download:

    python ingest.py universe    # list every symbol ever listed, filter, rank on June 2026 volume
                                 #   -> data/meta/candidates.csv, data/meta/universe.csv
    python ingest.py download    # download Jul-Sep 2026 1m zips for the universe, verify, unzip
                                 #   -> data/zip/, data/csv/, data/meta/manifest.csv
    python ingest.py parquet     # Spark: CSV -> Parquet partitioned by symbol, duplicate minutes dropped
                                 #   -> data/parquet/klines/symbol=*/

Both phases are safe to re-run: verified files already on disk are skipped.
Downloading is network-bound, so it uses a thread pool rather than Spark.
"""
import argparse
import calendar
import csv
import hashlib
import io
import logging
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------- configuration

RANK_MONTH = "2026-06"                              # point-in-time ranking month (before the study)
STUDY_MONTHS = ["2026-07", "2026-08", "2026-09"]
UNIVERSE_SIZE = 100                                 # never below 100: quintiles need 20 coins each
WORKERS = 16                                        # parallel HTTP requests
RETRIES = 4

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
META = DATA / "meta"
ZIP_DIR = DATA / "zip"
CSV_DIR = DATA / "csv"

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
LIST_PREFIX = "data/spot/monthly/klines/"
FILE_URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{interval}/{sym}-{interval}-{month}.zip"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Bases quoted against USDT that are not free-floating crypto: USD stablecoins, fiat,
# and tokens pegged 1:1 to another asset (gold, BTC, ETH). Their "returns" are peg noise.
STABLE_OR_PEGGED = {
    "USDC", "FDUSD", "TUSD", "BUSD", "USDP", "PAX", "DAI", "PYUSD", "USD1", "RLUSD", "USDE",
    "USDS", "USDSB", "SUSD", "BFUSD", "XUSD", "UST",
    "EUR", "EURI", "AEUR", "GBP", "AUD", "BRL", "TRY",
    "PAXG", "XAUT", "WBTC", "WBETH",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# Tokenized stocks/ETFs (Binance "<ticker>B" series, batch-listed from 2026-06-11). They follow equity
# market hours and news, not crypto microstructure. Only those with ranking-month data are listed;
# the rest of the series was listed after June and is already excluded by the point-in-time rule.
TOKENIZED_EQUITIES = {
    "SPCXB", "MUB", "SNDKB", "CRCLB", "TSLAB", "NVDAB", "AMDB", "EWYB", "INTCB", "MSTRB",
    "LITEB", "METAB", "MSFTB", "PLTRB", "QQQB",
}

# Data-driven peg filter: catches stablecoins missing from the list above (e.g. newly listed ones).
# A pair is excluded if every ranking-month daily close lies within this fraction of the month's mean close.
# Needs a few days of data to mean anything: a 1-day series trivially has zero deviation.
PEG_TOLERANCE = 0.01
PEG_MIN_DAYS = 5

# Local ASCII names for non-ASCII Binance symbols, so file paths and Spark partition folders stay safe.
# The original symbol is still used for download URLs. Unlisted non-ASCII symbols get a hex fallback.
SYMBOL_ALIASES = {"币安人生USDT": "BINANCELIFEUSDT"}

UNIT_DIVISOR = {"us": 10**6, "ms": 10**3, "s": 1}

MANIFEST_FIELDS = ["symbol", "binance_symbol", "month", "status", "zip_bytes", "rows", "expected_rows",
                   "missing_minutes", "gaps", "ts_unit", "first_open_time", "last_open_time",
                   "first_minute_ok", "has_header"]

log = logging.getLogger("ingest")

# ---------------------------------------------------------------- HTTP helpers

_local = threading.local()


def session():
    """One requests.Session per thread: reuses TCP/TLS connections without sharing state."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def http_get(url, params=None, stream=False):
    """GET with exponential-backoff retries. Returns the Response, or None on 404."""
    for attempt in range(RETRIES):
        try:
            r = session().get(url, params=params, stream=stream, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning("retry %d/%d for %s (%s), waiting %ds", attempt + 1, RETRIES - 1, url, e, wait)
            time.sleep(wait)


def fetch_checksum(url):
    """Expected SHA-256 from the .CHECKSUM companion ('<hash>  <filename>'), or None if absent."""
    r = http_get(url + ".CHECKSUM")
    return None if r is None else r.text.split()[0].lower()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- kline parsing helpers

def ts_unit(t):
    """Binance switched spot klines from milliseconds to microseconds on 2025-01-01: detect, never assume."""
    if t >= 10**15:
        return "us"
    if t >= 10**12:
        return "ms"
    return "s"


def to_epoch_seconds(t):
    return t // UNIT_DIVISOR[ts_unit(t)]


def utc_date(t):
    return datetime.fromtimestamp(to_epoch_seconds(t), timezone.utc).date().isoformat()


def local_symbol(sym):
    """ASCII-safe name used for files on disk and as the symbol column downstream."""
    if sym in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[sym]
    if sym.isascii():
        return sym
    return "X" + sym[:-4].encode("utf-8").hex().upper() + "USDT"


def read_zip_rows(src):
    """Rows of the single CSV inside a Binance zip (path or file-like). Skips a header row if present."""
    with zipfile.ZipFile(src) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if rows and not rows[0][0].isdigit():
        rows = rows[1:]
    return rows


# ---------------------------------------------------------------- phase 1: universe

def list_symbols():
    """Every symbol directory in the archive, including delisted pairs. Paginated 1000 at a time."""
    symbols, marker = [], ""
    while True:
        r = http_get(LIST_URL, params={"delimiter": "/", "prefix": LIST_PREFIX, "marker": marker})
        root = ET.fromstring(r.content)
        for p in root.findall(f"{S3_NS}CommonPrefixes/{S3_NS}Prefix"):
            symbols.append(p.text[len(LIST_PREFIX):].rstrip("/"))
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return symbols
        marker = root.findtext(f"{S3_NS}NextMarker")


def exclusion_reason(sym, usdt_bases):
    base = sym[:-4]
    if base in STABLE_OR_PEGGED:
        return "stablecoin/pegged"
    if base in TOKENIZED_EQUITIES:
        return "tokenized equity"
    # Leveraged tokens are <listed base> + suffix (BTCUP, ETHBEAR). Requiring the remainder to be a
    # real base of 2+ chars avoids false positives like JUP or SYRUP.
    for suffix in LEVERAGED_SUFFIXES:
        rest = base[:-len(suffix)]
        if base.endswith(suffix) and len(rest) >= 2 and rest in usdt_bases:
            return "leveraged token"
    return ""


def rank_month_stats(sym):
    """Daily klines for the ranking month: total quote volume and first/last trading day."""
    url = FILE_URL.format(sym=sym, interval="1d", month=RANK_MONTH)
    expected = fetch_checksum(url)
    r = http_get(url) if expected else None
    if r is None:
        return None
    if hashlib.sha256(r.content).hexdigest() != expected:
        raise ValueError(f"checksum mismatch: {url}")
    rows = read_zip_rows(io.BytesIO(r.content))
    if not rows:
        return None
    days = [utc_date(int(row[0])) for row in rows]
    closes = [float(row[4]) for row in rows]
    mean_close = sum(closes) / len(closes)
    return {
        "days": len(rows),
        "first_day": min(days),
        "last_day": max(days),
        "quote_volume": round(sum(float(row[7]) for row in rows), 2),
        "max_close_dev": round(max(abs(c / mean_close - 1) for c in closes), 6),
    }


def cmd_universe():
    start = time.time()
    all_symbols = list_symbols()
    usdt = sorted(s for s in all_symbols if s.endswith("USDT") and len(s) > 4)
    usdt_bases = {s[:-4] for s in usdt}
    log.info("archive lists %d symbols, %d USDT pairs", len(all_symbols), len(usdt))

    candidates = {s: {"symbol": s, "local_symbol": local_symbol(s), "reason": exclusion_reason(s, usdt_bases)}
                  for s in usdt}
    to_rank = [s for s in usdt if not candidates[s]["reason"]]
    log.info("excluded %d stablecoin/pegged/tokenized-equity/leveraged pairs; fetching %s daily data for %d",
             len(usdt) - len(to_rank), RANK_MONTH, len(to_rank))

    year, month = map(int, RANK_MONTH.split("-"))
    rank_last_day = f"{RANK_MONTH}-{calendar.monthrange(year, month)[1]:02d}"

    with ThreadPoolExecutor(WORKERS) as pool:
        for sym, stats in zip(to_rank, pool.map(rank_month_stats, to_rank)):
            if stats is None:
                candidates[sym]["reason"] = f"no {RANK_MONTH} data"
                continue
            candidates[sym].update(stats)
            if stats["days"] >= PEG_MIN_DAYS and stats["max_close_dev"] < PEG_TOLERANCE:
                candidates[sym]["reason"] = f"price pegged (closes within {PEG_TOLERANCE:.0%} of mean)"
            elif stats["last_day"] != rank_last_day:
                candidates[sym]["reason"] = f"stopped trading before {rank_last_day}"

    eligible = sorted((c for c in candidates.values() if not c["reason"]),
                      key=lambda c: c["quote_volume"], reverse=True)
    for i, c in enumerate(eligible, 1):
        c["rank"] = i
    universe = eligible[:UNIVERSE_SIZE]
    for c in eligible[UNIVERSE_SIZE:]:
        c["reason"] = f"rank > {UNIVERSE_SIZE}"
    if len(universe) < UNIVERSE_SIZE:
        log.warning("only %d eligible symbols, fewer than %d", len(universe), UNIVERSE_SIZE)

    fields = ["symbol", "local_symbol", "reason", "rank", "quote_volume", "days", "first_day", "last_day",
              "max_close_dev"]
    write_csv(META / "candidates.csv", fields, sorted(candidates.values(), key=lambda c: c["symbol"]))
    write_csv(META / "universe.csv", [f for f in fields if f != "reason"], universe)

    log.info("universe: %d symbols; #1 %s (%.0f USDT), #%d %s (%.0f USDT); %.1fs",
             len(universe), universe[0]["symbol"], universe[0]["quote_volume"],
             len(universe), universe[-1]["symbol"], universe[-1]["quote_volume"], time.time() - start)


# ---------------------------------------------------------------- phase 2: download

def stream_to_file(url, path):
    """Stream a download to <path>.part, then rename, so a crash never leaves a half file under the real name."""
    r = http_get(url, stream=True)
    if r is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(path)
    return True


def extract_csv(zip_path, csv_path):
    """Write the zip's single CSV member to csv_path. Never uses extractall, so member names can't escape."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{zip_path.name}: expected 1 CSV, found {members}")
        tmp = csv_path.with_suffix(".csv.part")
        with zf.open(members[0]) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
    tmp.replace(csv_path)


def inspect_csv(csv_path, month):
    """Row count, timestamp unit, gaps and first-minute check for one monthly file."""
    year, mon = map(int, month.split("-"))
    expected_rows = calendar.monthrange(year, mon)[1] * 1440
    month_start = int(datetime(year, mon, 1, tzinfo=timezone.utc).timestamp())

    rows, gaps, has_header = 0, 0, False
    units, first, last, prev_sec = set(), None, None, None
    with open(csv_path, newline="") as f:
        for rec in csv.reader(f):
            if not rec:
                continue
            if not rec[0].isdigit():
                has_header = True
                continue
            t = int(rec[0])
            sec = to_epoch_seconds(t)
            units.add(ts_unit(t))
            if prev_sec is not None and sec - prev_sec != 60:
                gaps += 1
            if first is None:
                first = t
            last, prev_sec = t, sec
            rows += 1

    return {
        "rows": rows,
        "expected_rows": expected_rows,
        "missing_minutes": expected_rows - rows,
        "gaps": gaps,
        "ts_unit": "|".join(sorted(units)),
        "first_open_time": first,
        "last_open_time": last,
        "first_minute_ok": first is not None and to_epoch_seconds(first) == month_start,
        "has_header": has_header,
    }


def download_one(sym, local, month):
    """Checksum -> download (unless a verified copy exists) -> verify -> unzip -> inspect.
    sym is the Binance symbol (for the URL); local is the ASCII name used on disk."""
    name = f"{local}-1m-{month}"
    url = FILE_URL.format(sym=sym, interval="1m", month=month)
    zip_path = ZIP_DIR / local / f"{name}.zip"
    csv_path = CSV_DIR / f"{name}.csv"
    row = {"symbol": local, "binance_symbol": sym, "month": month}

    expected = fetch_checksum(url)
    if expected is None:
        row["status"] = "missing"
        return row

    fresh = False
    if not (zip_path.exists() and sha256_file(zip_path) == expected):
        for attempt in (1, 2):
            if not stream_to_file(url, zip_path):
                row["status"] = "missing"
                return row
            if sha256_file(zip_path) == expected:
                fresh = True
                break
            log.warning("%s: checksum mismatch (attempt %d)", name, attempt)
        else:
            zip_path.unlink(missing_ok=True)
            row["status"] = "checksum_failed"
            return row

    if fresh or not csv_path.exists():
        extract_csv(zip_path, csv_path)

    row.update(inspect_csv(csv_path, month))
    row["zip_bytes"] = zip_path.stat().st_size
    row["status"] = "ok"
    return row


def cmd_download():
    universe_path = META / "universe.csv"
    if not universe_path.exists():
        sys.exit("data/meta/universe.csv not found - run `python ingest.py universe` first")
    with open(universe_path, newline="", encoding="utf-8") as f:
        symbols = [(r["symbol"], r["local_symbol"]) for r in csv.DictReader(f)]

    tasks = [(s, local, m) for s, local in symbols for m in STUDY_MONTHS]
    log.info("downloading %d files (%d symbols x %d months) with %d workers",
             len(tasks), len(symbols), len(STUDY_MONTHS), WORKERS)

    start, results = time.time(), []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(download_one, *task): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, local, month = futures[fut]
            try:
                row = fut.result()
            except Exception as e:                      # log and continue; re-running retries it
                row = {"symbol": local, "binance_symbol": sym, "month": month, "status": f"error: {e}"}
            results.append(row)
            if row["status"] != "ok":
                log.warning("%s %s: %s", local, month, row["status"])
            if i % 25 == 0 or i == len(tasks):
                log.info("%d/%d done (%.0fs)", i, len(tasks), time.time() - start)

    results.sort(key=lambda r: (r["symbol"], r["month"]))
    write_csv(META / "manifest.csv", MANIFEST_FIELDS, results)

    ok = [r for r in results if r["status"] == "ok"]
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    log.info("status counts: %s", statuses)
    log.info("rows: %d total; zipped: %.0f MB; time units seen: %s",
             sum(r["rows"] for r in ok), sum(r["zip_bytes"] for r in ok) / 1e6,
             sorted({r["ts_unit"] for r in ok}))
    bad_start = [f"{r['symbol']} {r['month']}" for r in ok if not r["first_minute_ok"]]
    if bad_start:
        log.warning("%d files do not start at the first minute of the month: %s", len(bad_start), bad_start)
    log.info("finished in %.1fs", time.time() - start)


# ---------------------------------------------------------------- phase 3: CSV -> Parquet (Spark)

def cmd_parquet():
    """Read all CSVs with an explicit schema, drop duplicate minutes, write Parquet partitioned by symbol."""
    from pyspark.sql import functions as F
    from common import KLINES_PQ, get_spark, kline_schema, save_plan, spark_path

    start = time.time()
    spark = get_spark("ingest-parquet")
    raw = spark.read.csv(spark_path(CSV_DIR / "*.csv"), schema=kline_schema(), header=False)
    df = (raw
          .withColumn("symbol", F.regexp_extract(F.col("_metadata.file_name"), r"^(.+)-1m-\d{4}-\d{2}\.csv$", 1))
          # microseconds from 2025 on, milliseconds before: normalise every row to microseconds
          .withColumn("ts_us", F.when(F.col("open_time") >= 10**15, F.col("open_time"))
                                .otherwise(F.col("open_time") * 1000))
          .drop("ignore"))

    n_raw = df.count()
    clean = df.dropDuplicates(["symbol", "ts_us"])
    out = clean.repartition("symbol").sortWithinPartitions("symbol", "ts_us")
    save_plan(out, "ingest_parquet")
    out.write.mode("overwrite").partitionBy("symbol").parquet(spark_path(KLINES_PQ))

    written = spark.read.parquet(spark_path(KLINES_PQ))
    n_clean = written.count()
    n_symbols = written.select("symbol").distinct().count()
    log.info("csv rows %d -> parquet rows %d (%d duplicates dropped), %d symbols; %.1fs",
             n_raw, n_clean, n_raw - n_clean, n_symbols, time.time() - start)
    spark.stop()


# ---------------------------------------------------------------- main

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["universe", "download", "parquet"])
    args = parser.parse_args()

    META.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # some Binance symbols are non-ASCII
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(META / "ingest.log", encoding="utf-8")],
    )
    {"universe": cmd_universe, "download": cmd_download, "parquet": cmd_parquet}[args.phase]()


if __name__ == "__main__":
    main()
