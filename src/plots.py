# Run plots (SIMULATION_DESIGN.md §8.3). Usage: python plots.py runs/<run_name>
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GENES = ("agg", "alt", "risk", "spec")
BINS = 10
LAST_GENS = 5


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    gens = _read_csv(run_dir / "generations.csv")
    agents = _read_csv(run_dir / "agents.csv")
    plots = (_gene_means, _gene_heatmaps, _behaviour_rates, _survival, _gene_vs_behaviour, _inequality, _specialization)
    return [plot(gens, agents, out) for plot in plots]


# ---- 1-7 --------------------------------------------------------------------

def _gene_means(gens, agents, out) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = _col(gens, "gen")
    for g in GENES:
        mean, std = _col(gens, f"mean_{g}"), _col(gens, f"std_{g}")
        ax.plot(x, mean, label=g)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set(xlabel="generation", ylabel="gene value", ylim=(0, 1), title="Gene means ± std")
    ax.legend()
    return _save(fig, out / "1_gene_means.png")


def _gene_heatmaps(gens, agents, out) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    generations = np.unique(_col(agents, "gen"))
    for ax, g in zip(axes.flat, GENES):
        image = ax.imshow(_histograms(agents, g, generations), aspect="auto", origin="lower", cmap="viridis",
                          extent=(0, 1, generations[0] - 0.5, generations[-1] + 0.5))
        ax.set(title=g, xlabel="gene value", ylabel="generation")
        fig.colorbar(image, ax=ax, label="fraction")
    fig.suptitle("Gene distributions per generation")
    return _save(fig, out / "2_gene_heatmaps.png")


def _behaviour_rates(gens, agents, out) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = _col(gens, "gen")
    for key in ("raid_rate", "contribution_rate", "hunt_participation", "gift_rate", "accept_rate"):
        ax.plot(x, _col(gens, key), marker=".", label=key)
    ax.set(xlabel="generation", ylabel="rate", ylim=(0, 1), title="Behaviour rates")
    ax.legend()
    return _save(fig, out / "3_behaviour_rates.png")


def _survival(gens, agents, out) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = _col(gens, "gen")
    bottom = np.zeros(len(x))
    for cause in ("starvation", "injury", "fight", "hunt"):
        deaths = _col(gens, f"deaths_{cause}")
        ax.bar(x, deaths, bottom=bottom, label=f"died: {cause}")
        bottom += deaths
    ax.plot(x, _col(gens, "survivors"), color="black", marker="o", label="survivors")
    ax.set(xlabel="generation", ylabel="agents", title="Survivors and deaths by cause")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return _save(fig, out / "4_survival.png")


def _gene_vs_behaviour(gens, agents, out) -> Path:
    last = _col(agents, "gen").max() - LAST_GENS + 1
    recent = [r for r in agents if r["gen"] >= last]
    pairs = (("agg", "raid_rate"), ("alt", "gift_rate"), ("alt", "contrib_rate"), ("risk", "far_rate"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (gene, rate) in zip(axes.flat, pairs):
        ax.scatter(_col(recent, gene), _col(recent, rate), s=12, alpha=0.5)
        ax.set(xlabel=gene, ylabel=rate, xlim=(0, 1), title=f"{gene} vs {rate}")
    fig.suptitle(f"Gene vs personal behaviour (last {LAST_GENS} generations)")
    return _save(fig, out / "5_gene_vs_behaviour.png")


def _inequality(gens, agents, out) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = _col(gens, "gen")
    ax.plot(x, _col(gens, "gini_wealth"), color="tab:red", marker=".", label="Gini (survivors)")
    ax.set(xlabel="generation", ylabel="Gini", ylim=(0, 1), title="Wealth inequality")
    twin = ax.twinx()
    twin.plot(x, _col(gens, "mean_wealth_end"), color="tab:blue", marker=".", label="mean wealth")
    twin.set_ylabel("mean wealth")
    fig.legend(loc="upper right")
    return _save(fig, out / "6_inequality.png")


def _specialization(gens, agents, out) -> Path:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=(2, 1))
    generations = np.unique(_col(agents, "gen"))
    top.imshow(_histograms(agents, "spec", generations).T, aspect="auto", origin="lower", cmap="viridis",
               extent=(generations[0] - 0.5, generations[-1] + 0.5, 0, 1))
    top.set(ylabel="spec (0 woodcutter, 1 forager)", title="Specialization over time (bimodal = division of labour)")
    bottom.plot(_col(gens, "gen"), _col(gens, "trade_rate"), marker=".")
    bottom.set(xlabel="generation", ylabel="trade rate")
    return _save(fig, out / "7_specialization.png")


# ---- helpers ----------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _to_float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _to_float(value: str) -> float:
    if value in ("True", "False"):
        return float(value == "True")
    return float(value) if value else np.nan


def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def _histograms(agents, gene: str, generations) -> np.ndarray:
    """(generations x BINS) matrix with the fraction of agents in each gene-value bin."""
    edges = np.linspace(0, 1, BINS + 1)
    rows = []
    for gen in generations:
        values = [r[gene] for r in agents if r["gen"] == gen]
        counts, _ = np.histogram(values, bins=edges)
        rows.append(counts / max(len(values), 1))
    return np.array(rows)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in plot_run(sys.argv[1]):
        print(p)
