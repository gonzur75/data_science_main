import sys
import math
from enum import IntEnum

import readchar
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Action3DEnum(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    FORWARD = 4
    BACKWARD = 5


class World3D:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.start = (0, 0, 0)
        self.state = self.start  # (x, y, z)
        self.goal = (size - 1, size - 1, size - 1)
        self.obstacles = self._build_obstacles()
        self.base_beacons = self._build_beacons()
        self.beacons = set(self.base_beacons)
        self.done = False
        self.reward = 0
        self.camera_yaw = -35.0
        self.camera_pitch = 25.0
        self.camera_distance = float(size * 3.3)
        self.camera_zoom = 1.4
        self.console = Console()

    def _build_obstacles(self) -> set[tuple[int, int, int]]:
        obstacles: set[tuple[int, int, int]] = set()
        n = self.size
        mid = n // 2

        # Centralna ściana z kilkoma bramami.
        for y in range(n):
            for z in range(n):
                obstacles.add((mid, y, z))

        # Druga ściana tworzy skręt i „korytarze” na różnych wysokościach.
        for x in range(1, n - 1):
            for y in range(n):
                if (x + y) % 2 == 0:
                    obstacles.add((x, y, mid))

        # Pojedyncze filary dla efektu 3D.
        pillars = {
            (1, n - 2, 1),
            (n - 2, 1, n - 2),
            (n - 3, n - 3, 1),
        }
        obstacles |= {p for p in pillars if all(0 <= v < n for v in p)}

        # Gwarantowany korytarz od startu do celu.
        safe_path = set()
        safe_path |= {(x, 0, 0) for x in range(n)}
        safe_path |= {(n - 1, y, 0) for y in range(n)}
        safe_path |= {(n - 1, n - 1, z) for z in range(n)}
        safe_path |= {
            (mid, mid, mid),
            (mid, 1, mid),
            (mid, n - 2, mid),
        }

        obstacles -= safe_path
        obstacles.discard(self.start)
        obstacles.discard(self.goal)
        return obstacles

    def _build_beacons(self) -> set[tuple[int, int, int]]:
        n = self.size
        points = {
            (n // 2, 0, n // 2),
            (0, n // 2, n // 2),
            (n // 2, n // 2, n - 1),
            (n - 2, n - 2, 1),
        }
        return {
            p
            for p in points
            if p != self.start and p != self.goal and p not in self.obstacles
        }

    def is_valid_state(self, state: tuple[int, int, int]) -> bool:
        x, y, z = state
        if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size):
            return False
        if state in self.obstacles:
            return False
        return True

    def _token_and_style(self, pos: tuple[int, int, int]) -> tuple[str, str]:
        if pos == self.state == self.goal:
            return "AK", "bold black on green"
        if pos == self.state:
            return "A", "bold white on blue"
        if pos == self.goal:
            return "K", "bold black on yellow"
        if pos in self.obstacles:
            return "O", "bold white on red"
        if pos in self.beacons:
            return "B", "bold cyan"
        return ".", "grey50"

    def _rotate_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        yaw = math.radians(self.camera_yaw)
        pitch = math.radians(self.camera_pitch)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)

        x1 = cos_yaw * x - sin_yaw * z
        z1 = sin_yaw * x + cos_yaw * z
        y1 = y

        y2 = cos_pitch * y1 - sin_pitch * z1
        z2 = sin_pitch * y1 + cos_pitch * z1
        return x1, y2, z2

    def _project_point(
        self, point: tuple[float, float, float], width: int, height: int
    ) -> tuple[int, int, float] | None:
        x, y, z = point
        center = (self.size - 1) / 2.0
        x -= center
        y -= center
        z -= center

        rx, ry, rz = self._rotate_point(x, y, z)
        depth = rz + self.camera_distance
        if depth <= 0.2:
            return None

        focal = min(width, height) * self.camera_zoom
        sx = int(width / 2 + (rx * focal) / depth * 2.0)
        sy = int(height / 2 - (ry * focal) / depth * 1.1)
        return sx, sy, depth

    @staticmethod
    def _put_pixel(
        chars: list[list[str]],
        styles: list[list[str | None]],
        depth_map: list[list[float]],
        x: int,
        y: int,
        depth: float,
        char: str,
        style: str,
    ) -> None:
        height = len(chars)
        width = len(chars[0])
        if 0 <= x < width and 0 <= y < height and depth < depth_map[y][x]:
            depth_map[y][x] = depth
            chars[y][x] = char
            styles[y][x] = style

    def _draw_line_3d(
        self,
        chars: list[list[str]],
        styles: list[list[str | None]],
        depth_map: list[list[float]],
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        width: int,
        height: int,
        *,
        char: str,
        style: str,
        samples: int = 24,
    ) -> None:
        for step in range(samples + 1):
            t = step / samples
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            )
            projected = self._project_point(point, width, height)
            if projected is None:
                continue
            sx, sy, depth = projected
            self._put_pixel(chars, styles, depth_map, sx, sy, depth, char, style)

    def _world_text_3d(self) -> Text:
        term_width = self.console.size.width
        term_height = self.console.size.height
        width = max(44, min(term_width - 8, 130))
        height = max(16, min(term_height - 17, 34))

        chars = [[" " for _ in range(width)] for _ in range(height)]
        styles: list[list[str | None]] = [[None for _ in range(width)] for _ in range(height)]
        depth_map = [[float("inf") for _ in range(width)] for _ in range(height)]

        cube_min = -0.5
        cube_max = self.size - 0.5
        corners = [
            (cube_min, cube_min, cube_min),
            (cube_max, cube_min, cube_min),
            (cube_max, cube_max, cube_min),
            (cube_min, cube_max, cube_min),
            (cube_min, cube_min, cube_max),
            (cube_max, cube_min, cube_max),
            (cube_max, cube_max, cube_max),
            (cube_min, cube_max, cube_max),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            self._draw_line_3d(
                chars,
                styles,
                depth_map,
                corners[a],
                corners[b],
                width,
                height,
                char=".",
                style="grey27",
                samples=32,
            )

        token_to_char = {
            ".": ".",
            "O": "#",
            "K": "$",
            "A": "@",
            "AK": "*",
            "B": "+",
        }

        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    pos = (x, y, z)
                    token, style = self._token_and_style(pos)
                    projected = self._project_point((float(x), float(y), float(z)), width, height)
                    if projected is None:
                        continue
                    sx, sy, depth = projected
                    char = token_to_char[token]

                    radius = 1 if token in {"A", "K", "AK", "O", "B"} else 0
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            attenuation = (abs(dx) + abs(dy)) * 0.01
                            self._put_pixel(
                                chars,
                                styles,
                                depth_map,
                                sx + dx,
                                sy + dy,
                                depth + attenuation,
                                char,
                                style,
                            )

        axis_origin = (0.0, 0.0, 0.0)
        axis_len = max(1.0, self.size - 1.0)
        self._draw_line_3d(
            chars,
            styles,
            depth_map,
            axis_origin,
            (axis_len, 0.0, 0.0),
            width,
            height,
            char="x",
            style="bold cyan",
        )
        self._draw_line_3d(
            chars,
            styles,
            depth_map,
            axis_origin,
            (0.0, axis_len, 0.0),
            width,
            height,
            char="y",
            style="bold green",
        )
        self._draw_line_3d(
            chars,
            styles,
            depth_map,
            axis_origin,
            (0.0, 0.0, axis_len),
            width,
            height,
            char="z",
            style="bold magenta",
        )

        text = Text()
        for row in range(height):
            for col in range(width):
                style = styles[row][col]
                if style is None:
                    text.append(chars[row][col])
                else:
                    text.append(chars[row][col], style=style)
            if row < height - 1:
                text.append("\n")

        text.append("\n")
        text.append(
            "Legenda: @ agent, $ cel, # przeszkoda, + beacon | Kamera: j/l yaw, i/k pitch, +/- zoom",
            style="italic cyan",
        )
        return text

    def _build_view(self, status: str) -> Group:
        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold cyan", justify="right")
        info.add_column()
        info.add_row("rozmiar", f"{self.size}x{self.size}x{self.size}")
        info.add_row("pozycja", f"{self.state}")
        info.add_row("cel", f"{self.goal}")
        info.add_row("przeszkody", f"{len(self.obstacles)}")
        info.add_row("beacony", f"{len(self.beacons)}")
        info.add_row("nagroda", f"{self.reward}")
        info.add_row(
            "epizod",
            "[green]zakończony[/green]" if self.done else "[yellow]w toku[/yellow]",
        )
        info.add_row("sterowanie", "WASD + Q/E | kamera: IJKL +/- | reset: r | wyjście: x")
        info.add_row("kamera", f"yaw={self.camera_yaw:.0f} pitch={self.camera_pitch:.0f} zoom={self.camera_zoom:.2f}")

        return Group(
            Panel(self._world_text_3d(), title="Rzut 3D", border_style="magenta"),
            Panel(info, title="Status", border_style="cyan"),
            Panel(status, border_style="green" if self.done else "white"),
        )

    def render(self) -> None:
        pass  # rendering handled by Live

    def step(self, action: Action3DEnum) -> tuple[tuple[int, int, int], float, bool, str]:
        if self.done:
            return self.state, self.reward, self.done, "finished"

        deltas: dict[Action3DEnum, tuple[int, int, int]] = {
            Action3DEnum.LEFT: (-1, 0, 0),
            Action3DEnum.RIGHT: (1, 0, 0),
            Action3DEnum.UP: (0, 1, 0),
            Action3DEnum.DOWN: (0, -1, 0),
            Action3DEnum.FORWARD: (0, 0, 1),
            Action3DEnum.BACKWARD: (0, 0, -1),
        }
        dx, dy, dz = deltas[action]
        candidate = (self.state[0] + dx, self.state[1] + dy, self.state[2] + dz)
        bounded = (
            max(0, min(self.size - 1, candidate[0])),
            max(0, min(self.size - 1, candidate[1])),
            max(0, min(self.size - 1, candidate[2])),
        )

        if bounded != candidate:
            self.reward = -0.35
            return self.state, self.reward, self.done, "out_of_bounds"
        if bounded in self.obstacles:
            self.reward = -0.6
            return self.state, self.reward, self.done, "obstacle"

        self.state = bounded
        if self.state == self.goal:
            self.done = True
            self.reward = 15
            return self.state, self.reward, self.done, "goal"
        if self.state in self.beacons:
            self.beacons.remove(self.state)
            self.reward = 1.0
            return self.state, self.reward, self.done, "beacon"

        self.reward = -0.1
        return self.state, self.reward, self.done, "moved"

    def reset(self) -> None:
        self.state = self.start
        self.done = False
        self.reward = 0
        self.beacons = set(self.base_beacons)

    def rotate_camera(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0) -> None:
        self.camera_yaw = (self.camera_yaw + yaw_delta) % 360
        self.camera_pitch = max(-75.0, min(75.0, self.camera_pitch + pitch_delta))

    def zoom_camera(self, delta: float) -> None:
        self.camera_zoom = max(0.5, min(3.0, self.camera_zoom + delta))

    def _normalize_command(self, key: str) -> str:
        if key in {readchar.key.LEFT, "a", "A"}:
            return "left"
        if key in {readchar.key.RIGHT, "d", "D"}:
            return "right"
        if key in {readchar.key.UP, "w", "W"}:
            return "up"
        if key in {readchar.key.DOWN, "s", "S"}:
            return "down"
        if key in {"q", "Q"}:
            return "backward"
        if key in {"e", "E"}:
            return "forward"
        if key in {"r", "R"}:
            return "reset"
        if key in {"j", "J"}:
            return "cam_left"
        if key in {"l", "L"}:
            return "cam_right"
        if key in {"i", "I"}:
            return "cam_up"
        if key in {"k", "K"}:
            return "cam_down"
        if key in {"+", "="}:
            return "zoom_in"
        if key in {"-", "_"}:
            return "zoom_out"
        if key in {"x", "X", readchar.key.CTRL_C}:
            return "quit"
        return "unknown"

    def _read_command(self) -> str:
        if not sys.stdin.isatty():
            typed = input("Ruch [w/s/a/d/q/e], kamera [i/j/k/l +/-], reset [r], quit [x]: ").strip()
            return self._normalize_command(typed[:1]) if typed else "unknown"
        return self._normalize_command(readchar.readkey())


def main():
    world = World3D()
    status = (
        "Ruch: strzałki/WASD + Q/E | Kamera: IJKL +/− | reset [r] | wyjście [x]."
    )
    move_actions = {
        "left": Action3DEnum.LEFT,
        "right": Action3DEnum.RIGHT,
        "up": Action3DEnum.UP,
        "down": Action3DEnum.DOWN,
        "forward": Action3DEnum.FORWARD,
        "backward": Action3DEnum.BACKWARD,
    }
    event_to_status = {
        "moved": "Ruch wykonany.",
        "beacon": "Zebrano beacon (+1.0).",
        "goal": "Brawo, dotarłeś do celu! [r] reset | [x] wyjście",
        "out_of_bounds": "Ściana świata: nie możesz wyjść poza granice.",
        "obstacle": "Kolizja z przeszkodą: ten blok jest zamknięty.",
        "finished": "Epizod zakończony. Użyj [r], aby zacząć od nowa.",
    }

    with Live(
        world._build_view(status),
        console=world.console,
        auto_refresh=False,
        transient=False,
    ) as live:
        while True:
            command = world._read_command()

            if command == "quit":
                status = "Koniec symulacji."
                live.update(world._build_view(status), refresh=True)
                break
            if command == "reset":
                world.reset()
                status = "Świat zresetowany."
            elif command == "cam_left":
                world.rotate_camera(yaw_delta=-8.0)
                status = "Kamera: yaw w lewo"
            elif command == "cam_right":
                world.rotate_camera(yaw_delta=8.0)
                status = "Kamera: yaw w prawo"
            elif command == "cam_up":
                world.rotate_camera(pitch_delta=6.0)
                status = "Kamera: pitch w górę"
            elif command == "cam_down":
                world.rotate_camera(pitch_delta=-6.0)
                status = "Kamera: pitch w dół"
            elif command == "zoom_in":
                world.zoom_camera(delta=0.12)
                status = "Kamera: zoom +"
            elif command == "zoom_out":
                world.zoom_camera(delta=-0.12)
                status = "Kamera: zoom -"
            elif command in move_actions:
                _, _, _, event = world.step(move_actions[command])
                status = event_to_status[event]
            else:
                status = "Nieznany klawisz. Użyj strzałek/WASD, Q/E, IJKL, +/-."

            live.update(world._build_view(status), refresh=True)


if __name__ == "__main__":
    main()
