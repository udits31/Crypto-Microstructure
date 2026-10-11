"""
analysis.py - the two measurements, on the Spark outputs of metrics.py.

    python analysis.py
        -> results/signal_decay.csv, results/signal_decay_by_symbol.csv
        -> results/quintile_weekly_returns.csv, results/quintile_summary.csv
        -> results/quintile_summary_zero_vol.csv   robustness: sort on share of no-trade minutes instead
        -> results/figures/02_signal_decay.png, results/figures/03_quintile_returns.png
        -> results/plans/analysis_*.txt

1. Signal decay. Regress the return over the next h minutes on this minute's order flow, h in 1/5/30/60
   (plus the same-minute return as a baseline). Univariate OLS with intercept, so R^2 = corr^2.
   Reported pooled across coins and per coin (median and quartiles across coins, and by liquidity tier).

2. Illiquidity premium. Each week w, rank coins by week_amihud (ntile(5) inside Window.partitionBy("week")),
   then average each quintile's return in week w+1. Ranking uses only week w; returns only week w+1.
   12 rebalances x ~20 coins per quintile is a small sample: confidence intervals are reported for that reason.

This measures market structure. It is NOT a trading strategy: no fees, spreads, price impact or
delisting losses are modelled, and none of these returns should be read as achievable.
"""
import math

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from common import (BLUE_ORDINAL, FIGURES, GRID, HORIZONS, META, MINUTE_PQ, N_WEEKS, ORANGE, RESULTS, TEXT,
                    TEXT_2, WEEKLY_PQ, chart_style, get_spark, save_plan, spark_path)

TARGETS = [("same minute", 0, "ret")] + [(f"+{h} min", h, f"fwd_ret_{h}") for h in HORIZONS]
TIERS = [("top third by June volume", 1, 33), ("middle third", 34, 66), ("bottom third", 67, 100)]

# two-sided 95% Student-t critical values by degrees of freedom
T_CRIT_95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}
BOOTSTRAP_REPS = 10_000

# ---------------------------------------------------------------- 1. signal decay


def signal_decay(spark):
    minute = spark.read.parquet(spark_path(MINUTE_PQ))
    panel = (minute.where(F.col("order_flow").isNotNull())        # zero-volume minutes have no order flow
             .select("symbol", "order_flow", *[y for _, _, y in TARGETS])
             .cache())                                             # reused by the pooled and per-coin passes

    aggs = []
    for _, h, y in TARGETS:
        aggs += [F.corr("order_flow", y).alias(f"corr_{h}"),
                 F.regr_slope(y, "order_flow").alias(f"beta_{h}"),
                 F.count(y).alias(f"n_{h}")]

    per_symbol_df = panel.groupBy("symbol").agg(*aggs)
    save_plan(per_symbol_df, "analysis_signal_decay")
    pooled = panel.agg(*aggs).first().asDict()
    per_symbol = per_symbol_df.toPandas()
    panel.unpersist()

    universe = pd.read_csv(META / "universe.csv", encoding="utf-8")[["local_symbol", "rank"]]
    per_symbol = per_symbol.merge(universe.rename(columns={"local_symbol": "symbol"}), on="symbol", how="left")

    rows = []
    for label, h, _ in TARGETS:
        r2 = per_symbol[f"corr_{h}"] ** 2
        row = {"horizon": label, "minutes": h,
               "pooled_r2": pooled[f"corr_{h}"] ** 2, "pooled_beta": pooled[f"beta_{h}"], "pooled_n": pooled[f"n_{h}"],
               "median_r2": r2.median(), "r2_q25": r2.quantile(0.25), "r2_q75": r2.quantile(0.75),
               "share_beta_positive": (per_symbol[f"beta_{h}"] > 0).mean(),
               "median_beta": per_symbol[f"beta_{h}"].median()}
        for name, lo, hi in TIERS:
            in_tier = per_symbol["rank"].between(lo, hi)
            row[f"median_r2 {name}"] = r2[in_tier].median()
        rows.append(row)
    decay = pd.DataFrame(rows)

    for _, h, _ in TARGETS:
        per_symbol[f"r2_{h}"] = per_symbol[f"corr_{h}"] ** 2
    per_symbol.sort_values("rank").to_csv(RESULTS / "signal_decay_by_symbol.csv", index=False)
    decay.to_csv(RESULTS / "signal_decay.csv", index=False)
    return decay


def chart_decay(decay):
    plt = chart_style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(decay))
    floor = 1e-6                                       # log axis: show R^2 below 1e-6 at the floor

    ax.fill_between(x, decay["r2_q25"].clip(lower=floor), decay["r2_q75"].clip(lower=floor),
                    color=GRID, alpha=0.9, linewidth=0, label="all coins: middle 50%")
    ax.plot(x, decay["median_r2"].clip(lower=floor), color=TEXT, marker="o", markersize=5,
            label="all coins: median")
    for (name, _, _), color in zip(TIERS, [BLUE_ORDINAL[4], BLUE_ORDINAL[2], BLUE_ORDINAL[0]]):
        y = decay[f"median_r2 {name}"].clip(lower=floor)
        ax.plot(x, y, color=color, marker="o", markersize=4, linewidth=1.6, label=f"median, {name}")

    ax.set_yscale("log")
    ax.set_xticks(x, decay["horizon"])
    ax.set_xlabel("Return horizon after the order-flow minute")
    ax.set_ylabel("R² of return on order flow (log scale)")
    ax.set_title("How fast order flow stops explaining returns", loc="left", pad=22)
    ax.text(0, 1.025,"Per-coin OLS R², 99 coins, Jul–Sep 2026 1-minute bars. Same-minute is the contemporaneous baseline.",
            transform=ax.transAxes, fontsize=8, color=TEXT_2)
    ax.legend(loc="upper right")
    fig.savefig(FIGURES / "02_signal_decay.png")
    plt.close(fig)


# ---------------------------------------------------------------- 2. illiquidity premium


def quintile_portfolios(weekly, sort_col):
    """Rank in week w on sort_col (1 = most liquid, 5 = least), equal-weighted return in week w+1."""
    w = Window.partitionBy("week").orderBy(F.col(sort_col), F.col("symbol"))     # symbol breaks ties deterministically
    ranked = (weekly.where((F.col("week") < N_WEEKS - 1) & F.col(sort_col).isNotNull())
              .withColumn("quintile", F.ntile(5).over(w)))
    next_week = weekly.select("symbol", (F.col("week") - 1).alias("week"), F.col("week_ret").alias("next_ret"))
    return (ranked.join(next_week, ["symbol", "week"], "inner")
            .groupBy("week", "quintile")
            .agg(F.avg("next_ret").alias("ret"), F.count("*").alias("n_coins"),
                 F.avg(sort_col).alias("avg_sort_value"))
            .orderBy("week", "quintile"))


def summarize(port):
    wide = port.pivot(index="week", columns="quintile", values="ret").sort_index()
    wide.columns = [f"Q{q}" for q in wide.columns]
    wide["Q5-Q1"] = wide["Q5"] - wide["Q1"]
    rng = np.random.default_rng(42)
    rows = []
    for col in wide.columns:
        r = wide[col].dropna().to_numpy()
        n, mean, sd = len(r), r.mean(), r.std(ddof=1)
        se = sd / math.sqrt(n)
        t_crit = T_CRIT_95[n - 1]
        boot = rng.choice(r, size=(BOOTSTRAP_REPS, n), replace=True).mean(axis=1)
        rows.append({"portfolio": col, "weeks": n, "mean_weekly_ret": mean, "sd_weekly_ret": sd,
                     "t_stat": mean / se, "ci95_lo_t": mean - t_crit * se, "ci95_hi_t": mean + t_crit * se,
                     "ci95_lo_boot": np.percentile(boot, 2.5), "ci95_hi_boot": np.percentile(boot, 97.5),
                     "share_weeks_positive": (r > 0).mean(), "cumulative_ret": np.prod(1 + r) - 1})
    return wide, pd.DataFrame(rows)


def chart_quintiles(wide, summary, week_starts):
    plt = chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

    # left: cumulative compounded return of each quintile, starting at 0 before the first return week
    weeks = list(wide.index)
    dates = [week_starts[w + 1] for w in weeks]
    x = [week_starts[weeks[0]]] + dates
    for q, color in zip(range(1, 6), BLUE_ORDINAL):
        cum = np.concatenate([[0.0], np.cumprod(1 + wide[f"Q{q}"].to_numpy()) - 1])
        tag = {1: "Q1 most liquid", 5: "Q5 least liquid"}.get(q, f"Q{q}")
        ax1.plot(x, cum * 100, color=color, linewidth=2, label=tag)
    ax1.axhline(0, color=TEXT_2, linewidth=0.8)
    ax1.set_ylabel("Cumulative return (%)")
    ax1.set_title("Next-week return of Amihud quintiles", loc="left")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(loc="upper left")

    # right: weekly long-short spread with its mean and 95% CI - shows how noisy 12 weeks are
    ls = wide["Q5-Q1"].to_numpy() * 100
    row = summary.set_index("portfolio").loc["Q5-Q1"]
    ax2.bar(range(1, len(ls) + 1), ls, color=[ORANGE if v >= 0 else BLUE_ORDINAL[3] for v in ls], width=0.7)
    ax2.axhspan(row["ci95_lo_t"] * 100, row["ci95_hi_t"] * 100, color=GRID, alpha=0.8, zorder=0,
                label="95% CI of the mean (t)")
    ax2.axhline(row["mean_weekly_ret"] * 100, color=TEXT, linewidth=1.5,
                label=f"mean {row['mean_weekly_ret'] * 100:+.2f}% / week")
    ax2.axhline(0, color=TEXT_2, linewidth=0.8)
    ax2.set_xlabel("Rebalance week")
    ax2.set_ylabel("Q5 − Q1 return (%)")
    ax2.set_title("Illiquid minus liquid, week by week", loc="left")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)

    fig.text(0.01, -0.06,"Measurement of market structure, not a trading strategy: no fees, spreads, price impact "
             "or delisting losses. 12 rebalances x ~20 coins per quintile is a small sample.",
             fontsize=8, color=TEXT_2)
    fig.savefig(FIGURES / "03_quintile_returns.png")
    plt.close(fig)


def illiquidity_premium(spark):
    weekly = spark.read.parquet(spark_path(WEEKLY_PQ))
    week_starts = (weekly.groupBy("week").agg(F.min("week_start").alias("start")).toPandas()
                   .set_index("week")["start"].dt.tz_localize(None).to_dict())

    outputs = {}
    for sort_col, suffix in [("week_amihud", ""), ("zero_vol_frac", "_zero_vol")]:
        port_df = quintile_portfolios(weekly, sort_col)
        if not suffix:
            save_plan(port_df, "analysis_quintiles")
        port = port_df.toPandas()
        wide, summary = summarize(port)
        port.to_csv(RESULTS / f"quintile_weekly_returns{suffix}.csv", index=False)
        summary.to_csv(RESULTS / f"quintile_summary{suffix}.csv", index=False)
        outputs[sort_col] = (port, wide, summary)

    port, wide, summary = outputs["week_amihud"]
    chart_quintiles(wide, summary, week_starts)
    return outputs


# ---------------------------------------------------------------- main


def pct(v):
    return f"{v * 100:+.2f}%"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    spark = get_spark("analysis")

    decay = signal_decay(spark)
    chart_decay(decay)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3g}".format):
        print("\n=== signal decay: R^2 of return on order flow")
        print(decay[["horizon", "pooled_r2", "median_r2", "r2_q25", "r2_q75", "median_beta",
                     "share_beta_positive", "pooled_n"]].to_string(index=False))

    outputs = illiquidity_premium(spark)
    for sort_col, (port, _, summary) in outputs.items():
        print(f"\n=== quintiles sorted on {sort_col} (Q1 = most liquid); coins per quintile per week: "
              f"{sorted(port['n_coins'].unique().tolist())}")
        show = summary.copy()
        for c in ["mean_weekly_ret", "sd_weekly_ret", "ci95_lo_t", "ci95_hi_t", "ci95_lo_boot", "ci95_hi_boot",
                  "cumulative_ret"]:
            show[c] = show[c].map(pct)
        show["t_stat"] = show["t_stat"].map("{:+.2f}".format)
        print(show.to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
