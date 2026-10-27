# The player feeds log items to the scene one timeline at a time. It shows them in a window (replay, or live while
# a simulation writes the log) or pipes the frames to ffmpeg to make a video.
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import pygame

from viewer.canvas import Canvas
from viewer.dashboard import Dashboard, Status, draw_help, draw_legend, draw_tooltip
from viewer.layout import H, MAP_W, W
from viewer.replay import DayLog, EventReader, GenEnd, GenStart, generations_logged
from viewer.scene import Scene
from viewer.style import GENES, INK, PANEL, shade
from viewer.timeline import Label

SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0)
MAX_RENDER_W = 3840  # cap on the supersampled frame width, so big screens stay smooth
LETTERBOX = shade(PANEL, 0.88)


class Player:
    def __init__(self, reader: EventReader, color_by="aggression", speed=1.0, want_day=None, realtime=True):
        self.reader = reader
        self.scene = Scene(reader.cfg, color_by)
        self.dashboard = Dashboard()
        self.speed = speed
        self.want_day = want_day or (lambda gen, day: True)
        self.realtime = realtime
        self.paused = False
        self.waiting = False
        self.done = False
        self.skip: str | None = None  # gen | live: consume log items without animating them

    def update(self, dt: float) -> None:
        sc = self.scene
        sc.clock += dt
        if self.paused:
            return
        if sc.timeline is not None:
            sc.t += dt * self.speed
            sc.timeline.advance(sc.t, sc)
            if sc.t >= sc.timeline.duration:
                sc.end_timeline()
        if sc.timeline is None:
            self._next(time.perf_counter() + 0.02 if self.realtime else math.inf)

    def command(self, what: str) -> None:
        """day: finish the current animation; gen / live: also skip to the next generation / the newest day."""
        if self.scene.timeline is not None:
            self.scene.end_timeline()
        self.skip = None if what == "day" else what

    def _next(self, deadline: float) -> None:
        sc = self.scene
        while time.perf_counter() < deadline:
            item = self.reader.next_item()
            sc.means = self.reader.gene_means
            if item is None:
                self.done, self.waiting = self.reader.finished, not self.reader.finished
                if self.skip == "live" or self.done:
                    self.skip = None
                return
            self.waiting = False
            if isinstance(item, GenStart):
                animate = self.skip != "live"
                if self.skip == "gen":
                    self.skip = None
                sc.start_generation(item, animate=animate)
                if animate:
                    return
            elif isinstance(item, DayLog):
                if self.skip is None and self.want_day(item.gen, item.day):
                    sc.play_day(item)
                    return
                sc.skip_day(item)
            elif isinstance(item, GenEnd) and self.skip is None:
                sc.end_generation(item)
                return

    def draw(self, cv: Canvas, dt: float, mode: str, run: str, mouse=None, show_help=False) -> None:
        sc = self.scene
        hover = sc.agent_at(mouse) if mouse and mouse[0] < MAP_W else None
        sc.draw(cv, hover)
        if sc.gen is None:
            Label("Waiting for the simulation to write its first day...", (MAP_W / 2, H / 2), 0, math.inf,
                  INK, 22).draw(cv, 1.0, sc)
        draw_legend(cv, sc)
        status = Status(run, mode, self.speed, self.paused, self.waiting, self.skip is not None)
        self.dashboard.draw(cv, sc, status, dt)
        if hover is not None:
            draw_tooltip(cv, sc, hover, mouse)
        if show_help:
            draw_help(cv)


def run_window(run_dir, *, cfg=None, live=None, not_before=0.0, sim=None, size=None, ssaa=2, speed=1.0,
               color_by="aggression") -> None:
    """Interactive viewer in a resizable window (F: full screen). Follows the log while the run is unfinished;
    `sim` is the simulation process, if any."""
    run_dir = Path(run_dir)
    pygame.init()
    window = pygame.Window(f"Laya Village - {run_dir.name}   (H: help, F: full screen)", size or _window_size(),
                           resizable=True, allow_high_dpi=True)
    window.minimum_size = (480, 270)
    reader = EventReader(run_dir, cfg=cfg, follow=True, not_before=not_before)
    if live is None:
        live = generations_logged(run_dir) < reader.cfg.population.G
    player = Player(reader, color_by, speed)
    clock, mouse, show_help, fullscreen, view = pygame.time.Clock(), None, False, False, None
    try:
        while True:
            dt = clock.tick(60) / 1000
            screen = window.get_surface()
            if view is None or view.pixels != screen.get_size():
                view = _Viewport(screen.get_size(), ssaa)
            dpi = screen.get_width() / max(1, window.size[0])  # 2 on Retina: events come in points
            for ev in pygame.event.get():
                key = ev.key if ev.type == pygame.KEYDOWN else None
                if ev.type in (pygame.QUIT, pygame.WINDOWCLOSE) or key == pygame.K_q or \
                        (key == pygame.K_ESCAPE and not fullscreen):
                    return
                if ev.type == pygame.MOUSEMOTION:
                    mouse = view.to_virtual((ev.pos[0] * dpi, ev.pos[1] * dpi))
                elif key in (pygame.K_f, pygame.K_F11, pygame.K_ESCAPE):
                    fullscreen = not fullscreen
                    if fullscreen:
                        window.set_fullscreen(desktop=True)
                    else:
                        window.set_windowed()
                elif key is not None:
                    show_help = _key(player, key, show_help)
            if sim is not None and reader.follow and not sim.is_alive():
                reader.follow = False
            player.update(min(dt, 0.1))
            player.draw(view.canvas, dt, "live" if live else "replay", run_dir.name, mouse, show_help)
            view.present(screen)
            window.flip()
    finally:
        pygame.quit()


class _Viewport:
    """The 16:9 frame, letterboxed in a window surface of any size, rendered supersampled and smoothed down."""

    def __init__(self, pixels: tuple[int, int], ssaa: float):
        self.pixels = pixels
        self.scale = max(1e-3, min(pixels[0] / W, pixels[1] / H))
        self.size = (max(1, round(W * self.scale)), max(1, round(H * self.scale)))
        self.offset = ((pixels[0] - self.size[0]) // 2, (pixels[1] - self.size[1]) // 2)
        k = min(self.scale * ssaa, max(self.scale, MAX_RENDER_W / W))
        self.canvas = Canvas(pygame.Surface((round(W * k), round(H * k))), k)

    def to_virtual(self, pos) -> tuple[float, float]:
        return (pos[0] - self.offset[0]) / self.scale, (pos[1] - self.offset[1]) / self.scale

    def present(self, screen: pygame.Surface) -> None:
        frame = self.canvas.surface
        if self.offset != (0, 0):
            screen.fill(LETTERBOX)
        screen.blit(frame if frame.get_size() == self.size else pygame.transform.smoothscale(frame, self.size),
                    self.offset)


def render_video(run_dir, out, *, gens=None, days=None, fps=30, size=(1920, 1080), ssaa=2, speed=1.0,
                 color_by="aggression", progress=print) -> tuple[Path, float]:
    """Renders the selected generations (0-based set) and days (set) to an H.264 video; returns (path, seconds)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; install it first (macOS: brew install ffmpeg)")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    run_dir, out = Path(run_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reader = EventReader(run_dir, want_gen=(lambda g: g in gens) if gens is not None else None)
    player = Player(reader, color_by, speed, (lambda gen, day: day in days) if days is not None else None,
                    realtime=False)
    w, h = size
    frame = pygame.Surface((round(W * w / W * ssaa), round(H * w / W * ssaa)))
    cv = Canvas(frame, w / W * ssaa)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames, tail, started = 0, 0, time.perf_counter()
    try:
        while tail < 2 * fps:
            player.update(1 / fps)
            tail += player.done
            player.draw(cv, 1 / fps, "video", run_dir.name)
            image = pygame.transform.smoothscale(frame, (w, h)) if frame.get_size() != (w, h) else frame
            proc.stdin.write(pygame.image.tobytes(image, "RGB"))
            frames += 1
            if frames % (fps * 20) == 0 and player.scene.gen is not None:
                progress(f"  {frames / fps:4.0f}s of video  (generation {player.scene.gen + 1}, day "
                         f"{player.scene.day})  {time.perf_counter() - started:.0f}s")
    finally:
        proc.stdin.close()
        proc.wait()
        pygame.quit()
    if proc.returncode:
        raise SystemExit(f"ffmpeg failed with exit code {proc.returncode}")
    return out, frames / fps


def parse_selection(spec: str, last: int) -> set[int] | None:
    """'1,4-6,last' -> {1, 4, 5, 6, last}; 'all' -> None."""
    if spec.strip().lower() == "all":
        return None
    picked = set()
    for part in spec.split(","):
        part = part.strip().lower().replace("last", str(last))
        lo, _, hi = part.partition("-")
        picked.update(range(int(lo), int(hi or lo) + 1))
    return picked


def _key(player: Player, key: int, show_help: bool) -> bool:
    if key == pygame.K_SPACE:
        player.paused = not player.paused
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP):
        player.speed = next((s for s in SPEEDS if s > player.speed), SPEEDS[-1])
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
        player.speed = next((s for s in reversed(SPEEDS) if s < player.speed), SPEEDS[0])
    elif key in (pygame.K_n, pygame.K_RIGHT):
        player.command("day")
    elif key == pygame.K_g:
        player.command("gen")
    elif key == pygame.K_l:
        player.command("live")
    elif key == pygame.K_h:
        return not show_help
    elif pygame.K_1 <= key <= pygame.K_4:
        player.scene.set_color_by(GENES[key - pygame.K_1])
    return show_help


def _window_size() -> tuple[int, int]:
    """The largest 16:9 window that fits on the desktop with room for the menu bar and dock."""
    info = pygame.display.Info()
    s = min(0.95 * info.current_w / W, 0.88 * info.current_h / H)
    return round(W * s), round(H * s)
