"""Original paper bar style plus strict macro aggregation."""
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from . import ROOT
OUT_DIR = ROOT / "outputs/figures"

def bars(summary: dict[str, dict], target: str, filename: str,
         regions: list[str], labels: list[str], colors: list[str], title: str,
         statistic: str = "mean") -> None:
    if statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported statistic: {statistic}")
    values = [summary[r][f"{statistic}_nmape_percent"] for r in regions]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    bars_ = ax.bar(np.arange(len(regions)), values, color=colors, width=0.58)
    ax.set_xticks(np.arange(len(regions)), labels)
    ax.set_ylabel(f"{statistic.capitalize()} per-MOF normalized MAE (%)")
    ax.set_title(title)
    ax.grid(axis="y", color="#dfe3e8", linewidth=0.6)
    ax.set_axisbelow(True)
    ymax = max(values) if values else 1.0
    ax.set_ylim(0, ymax * 1.2)
    for bar, value in zip(bars_, values):
        ax.annotate(f"{value:.2f}%", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9, color="#273746", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    fig.savefig((OUT_DIR / filename).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


REGIONS = {
    "ed": ["uniform", "gt_0.01_e_per_angstrom3", "gt_0.05_e_per_angstrom3", "gt_0.1_e_per_angstrom3"],
    "esp": ["uniform", "bottom_3pct", "top_3pct", "bottom_5pct", "top_5pct"],
}


def plot_all(metrics_dir, output_dir):
    """Require the complete fixed test set before drawing a paper comparison."""
    import json
    global OUT_DIR
    OUT_DIR = Path(output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/reproduce.json").read_text())
    expected = set(json.loads((ROOT / cfg["split"]).read_text())["test"])
    summaries = {}
    for target, regions in REGIONS.items():
        rows = {region: {} for region in regions}
        for shard in (0, 1):
            path = Path(metrics_dir) / f"{target}_shard{shard}_per_material.csv"
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    region, material = row["region"], row["material"]
                    if region not in rows:
                        continue
                    if material in rows[region]:
                        raise ValueError(f"Duplicate material-region: {material}, {region}")
                    value = float(row["normalized_mae_percent"])
                    if not math.isfinite(value) or int(row["points"]) != cfg["probes"]:
                        raise ValueError(f"Invalid score or sample count: {material}, {region}")
                    rows[region][material] = value
        summary = {}
        for region, by_material in rows.items():
            if set(by_material) != expected:
                raise ValueError(f"{target}/{region} requires exactly the original {len(expected)} test MOFs")
            scores = np.asarray(list(by_material.values()), dtype=float)
            summary[region] = {"target": target, "region": region, "mofs": len(scores),
                               "sampled_points": len(scores) * cfg["probes"],
                               "mean_nmape_percent": float(scores.mean()),
                               "median_nmape_percent": float(np.median(scores)),
                               "q25_nmape_percent": float(np.quantile(scores, .25)),
                               "q75_nmape_percent": float(np.quantile(scores, .75))}
        summaries[target] = summary
    bars(summaries["esp"], "esp", "esp_error_by_potential_region_mean_per_mof.png",
         REGIONS["esp"], ["All grid points", "Lowest 3%", "Highest 3%", "Lowest 5%", "Highest 5%"],
         ["#e5ad63", "#82b4ce", "#d46767", "#4e8ebd", "#b83f4b"],
         "ESP performance by potential region")
    bars(summaries["ed"], "ed", "ed_error_by_density_region_mean_per_mof.png",
         REGIONS["ed"], ["All grid points", "ED > 0.01", "ED > 0.05", "ED > 0.10"],
         ["#8eb5ce", "#609bc2", "#3782af", "#1b5c83"],
         "ED performance by density region")
    all_rows = [row for summary in summaries.values() for row in summary.values()]
    with (OUT_DIR / "macro_mean_per_mof_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote two figures (PNG + PDF) and macro summary to {OUT_DIR}")
    for target, summary in summaries.items():
        print(target.upper(), ", ".join(f"{region}: {row['mean_nmape_percent']:.2f}%" for region, row in summary.items()))
    return 0
