"""
phi_sweep.py
============
Example experiment script: sweep --phi-max (the rate at which an infected
host generates an isolated inter-community transmission event; see
phi_of()/attempt_inter_community_transmission() in community_sirs.py) across
several values, run multiple stochastic realizations at each value, and plot
how endemic outcomes respond.

This is meant as a template, not a one-off script. The same
"build args -> override one field -> run_sweep -> collect summary rows ->
plot" pattern below works for sweeping ANY other community_sirs.py argument
(--kbar, --alpha-max, --rewiring-prob, --K, --rho, ...): copy this file and
change which attribute of `args` gets overwritten inside the loop, and
which CLI flag drives the values.

Usage:
    python3 phi_sweep.py --out results/phi_sweep.png
    python3 phi_sweep.py --phi-values 0 0.02 0.05 0.1 0.2 --realizations 12
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

import community_sirs as cs


def run_for_phi(base_args: argparse.Namespace, phi_max: float,
                realizations: int, seed: int) -> list[dict]:
    """Run several realizations at one --phi-max value; return summary rows.

    Copying base_args and only overwriting the fields that change per
    scenario keeps every other model parameter (N, K, kbar, beta/alpha/gamma
    bounds, t_max, burn_in_time, ...) identical across the sweep, so any
    difference in the outcomes is attributable to phi_max.
    """
    args = argparse.Namespace(**vars(base_args))
    args.phi_max = phi_max
    args.realizations = realizations
    args.seed = seed
    cs.validate_args(args)
    rows, _samples = cs.run_sweep(args)
    return rows


def summarize(rows: list[dict], key: str) -> tuple[float, float]:
    """Mean and standard deviation of one summary field across realizations."""
    vals = np.array([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the sweep-script CLI. These are sweep controls, not passed
    straight through to community_sirs.py's own parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phi-values", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.04, 0.08, 0.16],
                   help="--phi-max values to sweep")
    p.add_argument("--realizations", type=int, default=8,
                   help="stochastic realizations per phi-max value")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--N", type=int, default=90)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--t-max", type=float, default=400.0)
    p.add_argument("--burn-in-time", type=float, default=100.0)
    p.add_argument("--out", default="results/phi_sweep.png",
                   help="output PNG path")
    return p


def main():
    args_cli = build_arg_parser().parse_args()

    # Base simulation arguments: start from community_sirs.py's own
    # defaults (built by its build_arg_parser, exactly like the CLI would),
    # then override only the handful of fields this sweep script exposes.
    base_args = cs.build_arg_parser().parse_args([])
    base_args.N = args_cli.N
    base_args.K = args_cli.K
    base_args.t_max = args_cli.t_max
    base_args.burn_in_time = args_cli.burn_in_time
    base_args.samples_out = None  # skip per-sample rows; only summary rows needed
    base_args.quiet = True

    phi_values = sorted(args_cli.phi_values)
    metrics = ["final_prevalence", "final_infected_virulence",
               "final_intercommunity_transmissions", "cumulative_deaths"]
    results = {m: {"mean": [], "std": []} for m in metrics}

    for phi_max in phi_values:
        rows = run_for_phi(base_args, phi_max, args_cli.realizations, args_cli.seed)
        print(f"phi_max={phi_max:.3f}: "
              f"mean final prevalence={np.nanmean([r['final_prevalence'] for r in rows]):.3f}, "
              f"mean cross-community transmissions="
              f"{np.nanmean([r['final_intercommunity_transmissions'] for r in rows]):.2f}")
        for m in metrics:
            mean, std = summarize(rows, m)
            results[m]["mean"].append(mean)
            results[m]["std"].append(std)

    # --- Plotting ---
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "final_prevalence": "Final prevalence (I/N)",
        "final_infected_virulence": "Final mean infected virulence",
        "final_intercommunity_transmissions": "Cumulative cross-community transmissions",
        "cumulative_deaths": "Cumulative deaths (birth/death turnover events)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, m in zip(axes.ravel(), metrics):
        mean = np.array(results[m]["mean"])
        std = np.array(results[m]["std"])
        ax.errorbar(phi_values, mean, yerr=std, marker="o", capsize=3, color="#1f78b4")
        ax.set_xlabel("--phi-max (inter-community contact rate)")
        ax.set_ylabel(titles[m])
        ax.set_title(titles[m])
        ax.grid(alpha=0.25)

    fig.suptitle("Community SIRS: sweeping the inter-community contact rate (phi_max)",
                fontsize=13)
    fig.tight_layout()

    out_path = args_cli.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
