#!/usr/bin/env python3
"""Turn a results folder into a chart + a markdown summary table.

    python scripts/plot.py results/20261007-101500            # main ladder
    python scripts/plot.py results/20261007-101500 --ghz 3.2 --cores 4
    python scripts/plot.py results/sweep-20261007-1030 --sweep  # tile sweep

Writes <dir>/chart.png and <dir>/summary.md (sweep: sweep.png).
--ghz / --cores add a FLOPs/core/cycle column (use your sustained all-core clock).
"""
import argparse
import csv
import sys
from pathlib import Path

# Article numbers (Intel i7-6700, 4C/8T Skylake @ 3.4 GHz, clang 14), in ms.
ARTICLE_MS = {1: 4481, 2: 1621, 3: 1512, 4: 89, 5: 70, 6: 16, 7: 8}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"   # your machine
REFERENCE = "#8f8d87"  # article reference (neutral, hollow)
SERIES_2 = "#eb6834"   # second line in the sweep chart


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def step_of(label):
    head = (label or "").strip().split(" ", 1)[0]
    return int(head) if head.isdigit() else None


def summary_table(rows, ghz, cores):
    first_ms = float(rows[0]["median_ms"])
    numpy_rows = [r for r in rows if r["kernel"] == "numpy"]
    numpy_ms = float(numpy_rows[-1]["median_ms"]) if numpy_rows else None
    hdr = ["Step", "Kernel", "Threads", "Median ms", "Stdev ms", "GFLOP/s",
           "vs prev", "vs first row", "% of numpy", "Article ms"]
    if ghz:
        hdr.insert(6, "FLOPs/core/cycle")
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    prev = None
    for r in rows:
        ms = float(r["median_ms"])
        gf = float(r["gflops"])
        step = step_of(r["label"])
        kern = r["kernel"] + (" t=" + r["tile"] if r.get("tile") else "")
        cells = [
            r["label"],
            kern,
            r["threads"],
            "%.1f" % ms,
            "%.1f" % float(r["stdev_ms"]),
            "%.1f" % gf,
            ("%.1fx" % (prev / ms)) if prev else "-",
            "%.0fx" % (first_ms / ms),
            (("%.0f%%" if 100 * numpy_ms / ms >= 10 else "%.1f%%") % (100 * numpy_ms / ms)) if numpy_ms else "-",
            str(ARTICLE_MS.get(step, "-")) if r["n"] == "1024" else "-",
        ]
        if ghz:
            used = int(r["threads"]) if r["threads"].lstrip("-").isdigit() and int(r["threads"]) > 0 else cores
            used = min(used, cores)
            cells.insert(6, "%.2f" % (gf / (used * ghz)))
        lines.append("| " + " | ".join(cells) + " |")
        prev = ms
    meta = rows[0]
    head = ["# MMM-on-CPU results", "",
            "- CPU: %s" % meta["cpu"],
            "- Compiler: %s" % meta["compiler"],
            "- Flags used: %s" % "  |  ".join(dict.fromkeys(r["flags"] for r in rows if r["flags"])),
            "- N = %s (2*N^3 = %.2f GFLOP per multiply)" % (meta["n"], 2 * int(meta["n"]) ** 3 / 1e9),
            "- CPU affinity mask: %s" % ", ".join(dict.fromkeys(r.get("affinity") or "?" for r in rows)),
            "- Timestamp (UTC): %s" % meta["timestamp_utc"], ""]
    bad = [r["label"] for r in rows if r["status"] != "ok"]
    if bad:
        head.append("**WARNING: wrong results for: %s**\n" % ", ".join(bad))
    return "\n".join(head + lines) + "\n"


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def ladder_chart(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = [r["label"] for r in rows]
    ms = [float(r["median_ms"]) for r in rows]
    art = [ARTICLE_MS.get(step_of(r["label"])) if r["n"] == "1024" else None for r in rows]
    y = list(range(len(rows)))[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    style(ax)
    ax.set_xscale("log")

    ay = [yy for yy, a in zip(y, art) if a]
    av = [a for a in art if a]
    if av:
        ax.scatter(av, ay, s=70, facecolors="none", edgecolors=REFERENCE, linewidths=1.6,
                   label="Article (i7-6700, 2022)", zorder=2)
    ax.scatter(ms, y, s=70, color=SERIES_1, edgecolors="#fcfcfb", linewidths=1.5,
               label="This machine", zorder=3)
    for v, yy in zip(ms, y):
        txt = ("%.0f ms" % v) if v >= 10 else ("%.1f ms" % v)
        ax.annotate(txt, (v, yy), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    lo = min(ms + av) / 2
    hi = max(ms + av) * 2
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v)))
    ax.set_xlabel("Median time per %sx%s multiply, ms (log scale, lower is better)" % (rows[0]["n"], rows[0]["n"]),
                  color=INK_2, fontsize=9)
    ax.set_title("Matrix multiply: optimisation ladder", loc="left", color=INK,
                 fontsize=12, fontweight="bold", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=8.5, labelcolor=INK_2, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("wrote", out)


def sweep_chart(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    series = {}
    for r in rows:
        key = "%s (%s thr)" % (r["kernel"], r["threads"])
        series.setdefault(key, []).append((int(r["tile"]), float(r["median_ms"])))
    colors = [SERIES_1, SERIES_2]
    for i, (name, pts) in enumerate(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
        best = min(pts, key=lambda p: p[1])
        ax.annotate("best: tile %d, %.1f ms" % best, best, xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(r["tile"]) for r in rows}))
    ax.set_xticklabels([str(t) for t in sorted({int(r["tile"]) for r in rows})])
    ax.set_xlabel("Inner tile size (rows of B per pass)", color=INK_2, fontsize=9)
    ax.set_ylabel("Median ms (lower is better)", color=INK_2, fontsize=9)
    ax.set_title("Tile-size grid search", loc="left", color=INK, fontsize=12,
                 fontweight="bold", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--ghz", type=float, default=None)
    p.add_argument("--cores", type=int, default=1)
    a = p.parse_args()
    d = Path(a.dir)
    csv_path = d / ("sweep.csv" if a.sweep else "results.csv")
    if not csv_path.exists():
        sys.exit("no %s" % csv_path)
    rows = load(csv_path)
    if a.sweep:
        sweep_chart(rows, d / "sweep.png")
        return
    md = summary_table(rows, a.ghz, a.cores)
    (d / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    try:
        ladder_chart(rows, d / "chart.png")
    except ImportError:
        print("matplotlib not installed - skipped chart (pip install matplotlib)")


if __name__ == "__main__":
    main()
