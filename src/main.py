# Command line (SIMULATION_DESIGN.md §10.3). From the repo root:
#   python src/main.py run --config configs/baseline.yaml --seed 0
#   python src/main.py run --config configs/baseline.yaml --brain mock --generations 2 --days 3   # smoke test
#   python src/main.py plot runs/baseline_s0
#   python src/main.py calibrate --config configs/baseline.yaml --out runs/calibration
import argparse
import sys
import time
from pathlib import Path

import yaml

from calibrate import calibrate
from config import Config
from plots import plot_run
from simulation import run


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="layalab", description="Laya Village evolutionary simulation")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="run a simulation from a config file")
    run_cmd.add_argument("--config", required=True, type=Path)
    run_cmd.add_argument("--seed", type=int, help="overrides `seed`")
    run_cmd.add_argument("--generations", type=int, help="overrides `population.G`")
    run_cmd.add_argument("--days", type=int, help="overrides `population.D`")
    run_cmd.add_argument("--brain", choices=("laya", "mock"), help="overrides `brain.kind`")
    run_cmd.add_argument("--out", type=Path, help="run directory (default: <log.out_dir>/<run_name>_s<seed>)")
    run_cmd.add_argument("--no-plots", action="store_true")

    plot_cmd = commands.add_parser("plot", help="(re)draw the plots of a finished run")
    plot_cmd.add_argument("run_dir", type=Path)

    cal_cmd = commands.add_parser("calibrate", help="E0: check that each decision responds to its gene")
    cal_cmd.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    cal_cmd.add_argument("--model", help="overrides `brain.model`")
    cal_cmd.add_argument("--brain", choices=("laya", "mock"), help="overrides `brain.kind`")
    cal_cmd.add_argument("--out", type=Path, default=Path("runs/calibration"))

    args = parser.parse_args()
    if args.command == "plot":
        _print_plots(plot_run(args.run_dir))
        return
    if args.command == "calibrate":
        _calibrate(args)
        return
    _run(args)


def _run(args) -> None:
    cfg = Config.from_dict(_with_overrides(yaml.safe_load(args.config.read_text()) or {}, args))
    run_dir = args.out or Path(cfg.log.out_dir) / f"{cfg.run_name}_s{cfg.seed}"
    print(f"run {cfg.run_name}: seed {cfg.seed}, brain {cfg.brain.kind}, "
          f"G={cfg.population.G} D={cfg.population.D} N={cfg.population.N} -> {run_dir}")

    start = time.perf_counter()
    result = run(cfg, run_dir)
    print(f"done in {time.perf_counter() - start:.0f}s -> {result.run_dir}")
    if not args.no_plots:
        _print_plots(plot_run(result.run_dir))


def _calibrate(args) -> None:
    cfg = Config.from_dict(_with_overrides(yaml.safe_load(args.config.read_text()) or {}, args))
    engine = {"laya": cfg.brain.model, "policy": f"policy engine ({cfg.brain.policy_weights})"}
    print(f"calibrating {engine.get(cfg.brain.kind, 'mock brain')} "
          f"(persona_mode={cfg.prompts.persona_mode}) -> {args.out / 'calibration.json'}")
    calibrate(cfg, args.out)


def _with_overrides(data: dict, args) -> dict:
    overrides = {
        ("seed",): getattr(args, "seed", None),
        ("population", "G"): getattr(args, "generations", None),
        ("population", "D"): getattr(args, "days", None),
        ("brain", "kind"): args.brain,
        ("brain", "model"): getattr(args, "model", None),
    }
    for path, value in overrides.items():
        if value is None:
            continue
        section = data
        for key in path[:-1]:
            section = section.setdefault(key, {})
        section[path[-1]] = value
    return data


def _print_plots(paths: list[Path]) -> None:
    print("plots:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
