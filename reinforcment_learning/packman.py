from __future__ import annotations

from collections import deque
from enum import IntEnum
import random
import sys

import numpy as np
import readchar
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class PackmanAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


DEFAULT_LAYOUT = (
    "####################",
    "#P...#...#...#.....#",
    "#.###.#.#.#.#.###..#",
    "#...#.#.#.#.#...#..#",
    "###.#.#.#.#.###.#.##",
    "#...#...#...#...#..#",
    "#.#####.###.#####..#",
    "#.....#..G..#....o.#",
    "#.###.#.###.#.###..#",
    "#..o..#.....#...#G.#",
    "####################",
)

ACTION_TO_DELTA: dict[PackmanAction, tuple[int, int]] = {
    PackmanAction.UP: (0, -1),
    PackmanAction.DOWN: (0, 1),
    PackmanAction.LEFT: (-1, 0),
    PackmanAction.RIGHT: (1, 0),
    PackmanAction.STAY: (0, 0),
}


class PackmanEnv:
    def __init__(
        self,
        layout: tuple[str, ...] = DEFAULT_LAYOUT,
        *,
        seed: int = 42,
        max_steps: int = 500,
    ) -> None:
        self.layout = layout
        self.height = len(layout)
        self.width = len(layout[0])
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.console = Console()

        self.walls: set[tuple[int, int]] = set()
        self.initial_pellets: set[tuple[int, int]] = set()
        self.initial_power_pellets: set[tuple[int, int]] = set()
        self.start: tuple[int, int] | None = None
        self.ghost_spawns: list[tuple[int, int]] = []
        self._parse_layout()

        if self.start is None:
            raise ValueError("Layout must contain one start position 'P'.")
        if not self.ghost_spawns:
            raise ValueError("Layout must contain at least one ghost spawn 'G'.")
        self._ensure_playable_map()

        self.player_pos: tuple[int, int] = self.start
        self.ghost_positions: list[tuple[int, int]] = list(self.ghost_spawns)
        self.pellets: set[tuple[int, int]] = set(self.initial_pellets)
        self.power_pellets: set[tuple[int, int]] = set(self.initial_power_pellets)
        self.power_timer: int = 0
        self.score: int = 0
        self.step_count: int = 0
        self.done: bool = False
        self.last_event: str = "new_episode"
        self.total_reward: float = 0.0

    @property
    def action_size(self) -> int:
        return len(PackmanAction)

    def _parse_layout(self) -> None:
        for y, row in enumerate(self.layout):
            if len(row) != self.width:
                raise ValueError("All rows in layout must have equal width.")
            for x, char in enumerate(row):
                pos = (x, y)
                if char == "#":
                    self.walls.add(pos)
                elif char == ".":
                    self.initial_pellets.add(pos)
                elif char == "o":
                    self.initial_power_pellets.add(pos)
                elif char == "P":
                    self.start = pos
                elif char == "G":
                    self.ghost_spawns.append(pos)

    def sample_action(self) -> PackmanAction:
        return PackmanAction(self.rng.randrange(self.action_size))

    def _reachable_from_start(self) -> set[tuple[int, int]]:
        if self.start is None:
            return set()
        if self.start in self.walls:
            return set()

        q: deque[tuple[int, int]] = deque([self.start])
        seen: set[tuple[int, int]] = {self.start}
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nxt = (nx, ny)
                if nxt in seen:
                    continue
                if 0 <= nx < self.width and 0 <= ny < self.height and nxt not in self.walls:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    def _carve_corridor(self, source: tuple[int, int], target: tuple[int, int]) -> None:
        x, y = source
        tx, ty = target
        horizontal_first = self.rng.random() < 0.5

        def step_towards(cx: int, cy: int, nx: int, ny: int) -> tuple[int, int]:
            if nx > cx:
                cx += 1
            elif nx < cx:
                cx -= 1
            elif ny > cy:
                cy += 1
            elif ny < cy:
                cy -= 1
            self.walls.discard((cx, cy))
            return cx, cy

        self.walls.discard((x, y))
        self.walls.discard((tx, ty))
        while (x, y) != (tx, ty):
            if horizontal_first and x != tx:
                x, y = step_towards(x, y, tx, y)
            elif y != ty:
                x, y = step_towards(x, y, x, ty)
            elif x != tx:
                x, y = step_towards(x, y, tx, y)

    def _ensure_playable_map(self) -> None:
        if self.start is None:
            raise ValueError("Missing player start 'P' in layout.")
        start_pos = self.start

        required = (
            set(self.initial_pellets)
            | set(self.initial_power_pellets)
            | set(self.ghost_spawns)
            | {start_pos}
        )
        reachable = self._reachable_from_start()
        unreachable = required - reachable

        while unreachable:
            target = min(
                unreachable, key=lambda p: abs(p[0] - start_pos[0]) + abs(p[1] - start_pos[1])
            )
            if not reachable:
                self.walls.discard(target)
                self.walls.discard(start_pos)
                reachable = self._reachable_from_start()
                unreachable = required - reachable
                continue

            anchor = min(
                reachable,
                key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]),
            )
            self._carve_corridor(anchor, target)
            reachable = self._reachable_from_start()
            unreachable = required - reachable

    def _to_observation(self) -> np.ndarray:
        obs = np.zeros((self.height, self.width, 6), dtype=np.float32)
        for x, y in self.walls:
            obs[y, x, 0] = 1.0
        for x, y in self.pellets:
            obs[y, x, 1] = 1.0
        for x, y in self.power_pellets:
            obs[y, x, 2] = 1.0
        px, py = self.player_pos
        obs[py, px, 3] = 1.0
        for gx, gy in self.ghost_positions:
            obs[gy, gx, 4] = 1.0
            if self.power_timer > 0:
                obs[gy, gx, 5] = 1.0
        return obs

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        self.player_pos = self.start  # type: ignore[assignment]
        self.ghost_positions = list(self.ghost_spawns)
        self.pellets = set(self.initial_pellets)
        self.power_pellets = set(self.initial_power_pellets)
        self.power_timer = 0
        self.score = 0
        self.step_count = 0
        self.done = False
        self.last_event = "new_episode"
        self.total_reward = 0.0
        return self._to_observation()

    def _is_in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_walkable(self, pos: tuple[int, int]) -> bool:
        return self._is_in_bounds(pos) and pos not in self.walls

    def _move(self, pos: tuple[int, int], action: PackmanAction) -> tuple[int, int]:
        dx, dy = ACTION_TO_DELTA[action]
        candidate = (pos[0] + dx, pos[1] + dy)
        if self._is_walkable(candidate):
            return candidate
        return pos

    def _ghost_next_position(self, ghost_pos: tuple[int, int]) -> tuple[int, int]:
        actions = (
            PackmanAction.UP,
            PackmanAction.DOWN,
            PackmanAction.LEFT,
            PackmanAction.RIGHT,
            PackmanAction.STAY,
        )
        candidates = [self._move(ghost_pos, action) for action in actions]
        candidates = list(dict.fromkeys(candidates))

        if self.rng.random() < 0.15:
            return self.rng.choice(candidates)

        px, py = self.player_pos
        scored: list[tuple[int, tuple[int, int]]] = []
        for cx, cy in candidates:
            dist = abs(cx - px) + abs(cy - py)
            scored.append((dist, (cx, cy)))

        if self.power_timer > 0:
            best_score = max(score for score, _ in scored)
            best = [pos for score, pos in scored if score == best_score]
        else:
            best_score = min(score for score, _ in scored)
            best = [pos for score, pos in scored if score == best_score]
        return self.rng.choice(best)

    def _resolve_collisions(self) -> tuple[float, str, bool]:
        reward_delta = 0.0
        if self.player_pos not in self.ghost_positions:
            return reward_delta, "none", False

        if self.power_timer > 0:
            reward_delta += 8.0
            self.score += 200
            refreshed: list[tuple[int, int]] = []
            for ghost in self.ghost_positions:
                if ghost == self.player_pos:
                    refreshed.append(self.rng.choice(self.ghost_spawns))
                else:
                    refreshed.append(ghost)
            self.ghost_positions = refreshed
            return reward_delta, "eat_ghost", False

        reward_delta -= 20.0
        return reward_delta, "hit_ghost", True

    def step(
        self, action: PackmanAction | int
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if isinstance(action, int):
            action = PackmanAction(action)
        if self.done:
            raise ValueError("Episode ended. Call reset() before step().")

        self.step_count += 1
        reward = -0.02
        events: list[str] = []

        old_pos = self.player_pos
        self.player_pos = self._move(self.player_pos, action)
        if self.player_pos == old_pos and action != PackmanAction.STAY:
            reward -= 0.12
            events.append("wall")

        if self.player_pos in self.pellets:
            self.pellets.remove(self.player_pos)
            reward += 1.0
            self.score += 10
            events.append("pellet")

        if self.player_pos in self.power_pellets:
            self.power_pellets.remove(self.player_pos)
            self.power_timer = 20
            reward += 4.0
            self.score += 50
            events.append("power")

        bonus, collision_event, terminal = self._resolve_collisions()
        reward += bonus
        if collision_event != "none":
            events.append(collision_event)
        if terminal:
            self.done = True

        if not self.done:
            self.ghost_positions = [
                self._ghost_next_position(ghost) for ghost in self.ghost_positions
            ]
            bonus, collision_event, terminal = self._resolve_collisions()
            reward += bonus
            if collision_event != "none":
                events.append(collision_event)
            if terminal:
                self.done = True

        if self.power_timer > 0:
            self.power_timer -= 1

        if not self.done and not self.pellets and not self.power_pellets:
            self.done = True
            reward += 25.0
            self.score += 500
            events.append("win")

        if not self.done and self.step_count >= self.max_steps:
            self.done = True
            events.append("timeout")

        self.total_reward += reward
        if events:
            self.last_event = ",".join(events)
        else:
            self.last_event = "move"

        info: dict[str, object] = {
            "score": self.score,
            "steps": self.step_count,
            "power_timer": self.power_timer,
            "pellets_left": len(self.pellets),
            "power_pellets_left": len(self.power_pellets),
            "event": self.last_event,
            "position": self.player_pos,
            "done_reason": self.last_event if self.done else "",
        }
        return self._to_observation(), reward, self.done, info

    def _world_text(self) -> Text:
        world = Text()
        ghost_set = set(self.ghost_positions)
        frightened = self.power_timer > 0

        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos in self.walls:
                    world.append("##", style="bold blue")
                    continue

                if pos == self.player_pos and pos in ghost_set:
                    if frightened:
                        world.append("X ", style="bold black on green")
                    else:
                        world.append("! ", style="bold white on red")
                    continue

                if pos == self.player_pos:
                    world.append("C ", style="bold yellow")
                    continue

                if pos in ghost_set:
                    style = "bold cyan" if frightened else "bold red"
                    world.append("G ", style=style)
                    continue

                if pos in self.power_pellets:
                    world.append("* ", style="bold magenta")
                elif pos in self.pellets:
                    world.append(". ", style="white")
                else:
                    world.append("  ")
            if y < self.height - 1:
                world.append("\n")
        return world

    def _build_view(self, status: str) -> Group:
        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold cyan", justify="right")
        info.add_column()
        info.add_row("size", f"{self.width}x{self.height}")
        info.add_row("position", str(self.player_pos))
        info.add_row("score", str(self.score))
        info.add_row("reward_sum", f"{self.total_reward:.2f}")
        info.add_row("steps", f"{self.step_count}/{self.max_steps}")
        info.add_row("pellets", str(len(self.pellets)))
        info.add_row("power", str(len(self.power_pellets)))
        info.add_row("power_timer", str(self.power_timer))
        info.add_row("event", self.last_event)
        info.add_row("episode", "done" if self.done else "running")
        info.add_row("controls", "WASD/Arrows, stay: space, reset: r, quit: x")

        return Group(
            Panel(self._world_text(), title="Packman RL World", border_style="magenta"),
            Panel(info, title="Status", border_style="cyan"),
            Panel(status, border_style="green" if self.done else "white"),
        )

    def render(self, status: str = "preview") -> None:
        self.console.print(self._build_view(status))

    @staticmethod
    def _normalize_command(key: str) -> str:
        if key in {readchar.key.LEFT, "a", "A"}:
            return "left"
        if key in {readchar.key.RIGHT, "d", "D"}:
            return "right"
        if key in {readchar.key.UP, "w", "W"}:
            return "up"
        if key in {readchar.key.DOWN, "s", "S"}:
            return "down"
        if key in {" "}:
            return "stay"
        if key in {"r", "R"}:
            return "reset"
        if key in {"x", "X", readchar.key.CTRL_C}:
            return "quit"
        return "unknown"

    def _read_command(self) -> str:
        if not sys.stdin.isatty():
            typed = input("move [w/s/a/d/space], reset [r], quit [x]: ").strip()
            return self._normalize_command(typed[:1]) if typed else "unknown"
        return self._normalize_command(readchar.readkey())

    def run_interactive(self) -> None:
        status = "Use WASD or arrows. Space = stay. r = reset. x = quit."
        command_to_action: dict[str, PackmanAction] = {
            "up": PackmanAction.UP,
            "down": PackmanAction.DOWN,
            "left": PackmanAction.LEFT,
            "right": PackmanAction.RIGHT,
            "stay": PackmanAction.STAY,
        }

        with Live(
            self._build_view(status),
            console=self.console,
            auto_refresh=False,
            transient=False,
        ) as live:
            while True:
                command = self._read_command()
                if command == "quit":
                    status = "Simulation ended."
                    live.update(self._build_view(status), refresh=True)
                    break
                if command == "reset":
                    self.reset()
                    status = "Environment reset."
                    live.update(self._build_view(status), refresh=True)
                    continue

                if command in command_to_action:
                    if self.done:
                        status = (
                            f"episode done ({self.last_event}) | press r to reset or x to quit"
                        )
                    else:
                        try:
                            _, reward, _, info = self.step(command_to_action[command])
                            status = f"reward={reward:+.2f} | event={info['event']}"
                        except ValueError:
                            status = (
                                f"episode done ({self.last_event}) | press r to reset or x to quit"
                            )
                else:
                    status = "Unknown key. Use WASD/arrows, space, r, x."

                if self.done and "episode done" not in status:
                    status = f"{status} | episode done ({self.last_event}) | press r"
                live.update(self._build_view(status), refresh=True)


def run_random_episode(env: PackmanEnv, seed: int | None = None) -> float:
    env.reset(seed=seed)
    done = False
    total = 0.0
    while not done:
        _, reward, done, _ = env.step(env.sample_action())
        total += reward
    return total


if __name__ == "__main__":
    PackmanEnv().run_interactive()
