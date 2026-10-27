# Command line (SIMULATION_DESIGN.md §10.3). From the repo root:
#   python src/main.py run --config configs/baseline.yaml --seed 0
#   python src/main.py run --config configs/baseline.yaml --brain mock --generations 2 --days 3   # smoke test
#   python src/main.py plot runs/baseline_s0
#   python src/main.py calibrate --config configs/baseline.yaml --out runs/calibration
#   python src/main.py run --config configs/policy.yaml --live      # watch the village while it runs
#   python src/main.py watch runs/policy_s0                          # replay (or follow) a run in a window
#   python src/main.py render runs/policy_s0 --gens 1,last --days 1-6   # village video (needs ffmpeg)
import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import yaml

from calibrate import calibrate
from config import Config
from genes import Gene
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
    run_cmd.add_argument("--live", action="store_true", help="watch the village in a window while it runs")
    _viewer_args(run_cmd)

    plot_cmd = commands.add_parser("plot", help="(re)draw the plots of a finished run")
    plot_cmd.add_argument("run_dir", type=Path)

    watch_cmd = commands.add_parser("watch", help="village viewer: replays a run, or follows it while it runs")
    watch_cmd.add_argument("run_dir", type=Path)
    _viewer_args(watch_cmd)

    render_cmd = commands.add_parser("render", help="render the village animation of a run to an MP4 (needs ffmpeg)")
    render_cmd.add_argument("run_dir", type=Path)
    render_cmd.add_argument("--out", type=Path, help="video file (default: <run_dir>/village.mp4)")
    render_cmd.add_argument("--gens", default="1,last", help="generations to show, 1-based: e.g. 1,5-7,last or all")
    render_cmd.add_argument("--days", default="1-6", help="days of each shown generation: e.g. 1-20 or all")
    render_cmd.add_argument("--fps", type=int, default=30)
    _viewer_args(render_cmd, size="1920x1080", speed=1.5)

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
    if args.command == "watch":
        from viewer.app import run_window
        run_window(args.run_dir, size=args.size, ssaa=args.ssaa, speed=args.speed, color_by=args.color_by)
        return
    if args.command == "render":
        _render(args)
        return
    _run(args)


def _viewer_args(cmd, size=None, speed=1.0) -> None:
    cmd.add_argument("--speed", type=float, default=speed, help=f"animation speed (default {speed:g})")
    cmd.add_argument("--color-by", choices=[g.value for g in Gene], default="aggression",
                     help="gene shown as the blobs' body colour")
    cmd.add_argument("--size", type=_size, default=size,
                     help="WIDTHxHEIGHT: video size, or initial window size (resizable; default fits the screen)")
    cmd.add_argument("--ssaa", type=int, default=2, help="supersampling factor for anti-aliasing (1 = faster)")


def _size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


def _render(args) -> None:
    from viewer.app import parse_selection, render_video
    from viewer.replay import generations_logged, load_config
    population = load_config(args.run_dir).population
    gens = parse_selection(args.gens, generations_logged(args.run_dir) or population.G)
    days = parse_selection(args.days, population.D)
    out = args.out or args.run_dir / "village.mp4"
    print(f"rendering {args.run_dir}: generations {args.gens}, days {args.days} -> {out}")
    start = time.perf_counter()
    path, seconds = render_video(args.run_dir, out, gens=None if gens is None else {g - 1 for g in gens}, days=days,
                                 fps=args.fps, size=args.size, ssaa=args.ssaa, speed=args.speed,
                                 color_by=args.color_by)
    print(f"done in {time.perf_counter() - start:.0f}s: {seconds:.0f}s of video -> {path}")


def _run(args) -> None:
    cfg = Config.from_dict(_with_overrides(yaml.safe_load(args.config.read_text()) or {}, args))
    run_dir = args.out or Path(cfg.log.out_dir) / f"{cfg.run_name}_s{cfg.seed}"
    print(f"run {cfg.run_name}: seed {cfg.seed}, brain {cfg.brain.kind}, "
          f"G={cfg.population.G} D={cfg.population.D} N={cfg.population.N} -> {run_dir}")

    start = time.perf_counter()
    if args.live:
        _run_live(cfg, run_dir, args)
    else:
        run(cfg, run_dir)
    print(f"done in {time.perf_counter() - start:.0f}s -> {run_dir}")
    if not args.no_plots:
        _print_plots(plot_run(run_dir))


def _run_live(cfg: Config, run_dir: Path, args) -> None:
    """The simulation runs in a child process; this one shows its events.jsonl in the viewer as it grows."""
    from viewer.app import run_window
    started = time.time()
    sim = multiprocessing.get_context("spawn").Process(target=run, args=(cfg, run_dir), name="simulation")
    sim.start()
    try:
        run_window(run_dir, cfg=cfg, live=True, not_before=started - 1, sim=sim, size=args.size, ssaa=args.ssaa,
                   speed=args.speed, color_by=args.color_by)
        if sim.is_alive():
            print("viewer closed; the simulation goes on (Ctrl-C to stop it)")
        sim.join()
    except KeyboardInterrupt:
        sim.terminate()
        sim.join()
        raise
    if sim.exitcode:
        raise SystemExit(f"the simulation failed (exit code {sim.exitcode})")


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
