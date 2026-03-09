# from enum import IntEnum
#
# import numpy as np
# from rich.console import Console
# from rich.text import Text
#
# class ActionEnum(IntEnum):
#     LEFT = 0
#     RIGHT = 1
#
#
# class SimpleWorld:
#     def __init__(self, size: int = 15) -> None:
#         self.state: int = 0
#         self.size = size
#         self.endstep = self.size - 1
#         self.done: bool = False
#         self.action: ActionEnum = ActionEnum.LEFT
#         self.console: Console = Console()
#         self._deltas = np.array([1, -1], dtype=np.int8)
#
#
#     def render(self) -> None:
#         world = Text()
#         for idx in range(self.size):
#             if idx == self.state == self.endstep:
#                 world.append(" AK ", style="bold black on green")
#             elif idx == self.state:
#                 world.append(" A  ", style="bold white on blue")
#             elif idx == self.endstep:
#                 world.append(" K  ", style="bold black on yellow")
#             else:
#                 world.append(" .  ", style="dim")
#
#         self.console.print(world)
#         self.console.print(f"pozycja X: {self.state} | koniec: {self.endstep}")
#
#
#     def step(self, action) -> None:
#         if  self.done:
#             raise ValueError("Episode is done. Please reset the environment.")
#
#         try:
#             delta = int(self._deltas[int(action)])
#         except (IndexError, ValueError) as exc:
#             raise ValueError(f"Invalid action: {action}") from exc
#
#         self.state = int(np.clip(self.state + delta, 0, self.endstep))
#         self.done = self.state == self.endstep
#
#         # Pawel's version:
#         # if action == ActionEnum.LEFT:
#         #     self.state = max(0, self.state - 1)
#         # elif action == ActionEnum.RIGHT:
#         #     self.state = min(self.size - 1, self.state + 1)
#         #
#         # if self.state == self.endstep:
#         #     self.done = True
#
#
#         # My version:
#         # if action == ActionEnum.LEFT:
#         #     self.state = max(0, self.state - 1)
#         # elif action == ActionEnum.RIGHT:
#         #     self.state = min(self.size - 1, self.state + 1)
#         # else:
#         #     raise ValueError("Invalid action. Use ActionEnum.LEFT or ActionEnum.RIGHT.")
#         #
#         # self.done = self.state == self.endstep
#
#     def reset(self) -> None:
#         self.state = 0
#         self.done = False
#
# def play_interactive(env: SimpleWorld) -> None:
#     env.reset()
#     while not env.done:
#         action_input = input("Enter action (0 for LEFT, 1 for RIGHT): ")
#         try:
#             action = ActionEnum(int(action_input))
#         except ValueError:
#             print("Invalid input. Please enter 0 for LEFT or 1 for RIGHT.")
#             continue
#         env.step(action)
#         env.render()
#     print("You win!")
#
#
#
# simple_world: SimpleWorld = SimpleWorld(size=15)
# play_interactive(simple_world)
#


from enum import IntEnum
import sys

import numpy as np
import readchar
from pyparsing import actions
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

class ActionEnum(IntEnum):
    LEFT = -1
    RIGHT = 1


class SimpleWorld:
    def __init__(self, size: int = 15) -> None:
        self.state: int = 0
        self.size: int = size
        self.endstep: int = size - 1
        self.done: bool = False
        self.console = Console()

    def _world_text(self) -> Text:
        cells = np.full(self.size, ".", dtype="<U2")
        cells[self.endstep] = "K"
        cells[self.state] = "A"
        if self.state == self.endstep:
            cells[self.endstep] = "AK"

        world = Text()
        for token in cells:
            if token == "AK":
                world.append(" AK ", style="bold black on green")
            elif token == "A":
                world.append(" A  ", style="bold white on blue")
            elif token == "K":
                world.append(" K  ", style="bold black on yellow")
            else:
                world.append(" .  ", style="dim")
        return world

    def _build_view(self, status: str) -> Group:
        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold cyan", justify="right")
        info.add_column()
        info.add_row("pozycja", f"{self.state} / {self.endstep}")
        info.add_row(
            "epizod", "[green]zakonczony[/green]" if self.done else "[yellow]w toku[/yellow]"
        )
        info.add_row("sterowanie", "←/→ lub a/d | reset: r | wyjscie: q")

        return Group(
            Panel(self._world_text(), title="SimpleWorld", border_style="magenta"),
            Panel(info, title="Status", border_style="cyan"),
            Panel(status, border_style="green" if self.done else "white"),
        )

    def render(self) -> None:
        self.console.print(self._build_view("Podglad swiata."))

    def step(self, action: ActionEnum) -> None:
        if self.done:
            raise ValueError("Episode has ended. Please reset the environment.")

        self.state = int(np.clip(self.state + int(action), 0, self.endstep))
        self.done = self.state == self.endstep

    def reset(self) -> None:
        self.state = 0
        self.done = False

    @staticmethod
    def _normalize_command(key: str) -> str:
        if key in {readchar.key.LEFT, "a", "A"}:
            return "left"
        if key in {readchar.key.RIGHT, "d", "D"}:
            return "right"
        if key in {"r", "R"}:
            return "reset"
        if key in {"q", "Q", readchar.key.CTRL_C}:
            return "quit"
        return "unknown"

    def _read_command(self) -> str:
        if not sys.stdin.isatty():
            typed = input("Ruch [a/d], reset [r], quit [q]: ").strip()
            return self._normalize_command(typed[:1]) if typed else "unknown"
        return self._normalize_command(readchar.readkey())

    def run_interactive(self) -> None:
        status = "Sterowanie: strzalki lub [a]/[d], reset [r], wyjscie [q]."

        with Live(
            self._build_view(status),
            console=self.console,
            auto_refresh=False,
            transient=False,
        ) as live:
            while True:
                command = self._read_command()

                if command == "quit":
                    status = "Koniec symulacji."
                    live.update(self._build_view(status), refresh=True)
                    break
                if command == "reset":
                    self.reset()
                    status = "Swiat zresetowany."
                elif command == "left":
                    if self.done:
                        status = "Epizod zakonczony. Uzyj [r], aby zaczac od nowa."
                    else:
                        self.step(ActionEnum.LEFT)
                        status = "Ruch: LEFT"
                elif command == "right":
                    if self.done:
                        status = "Epizod zakonczony. Uzyj [r], aby zaczac od nowa."
                    else:
                        self.step(ActionEnum.RIGHT)
                        status = "Ruch: RIGHT"
                else:
                    status = "Nieznany klawisz. Uzyj strzalek, [a]/[d], [r], [q]."

                if self.done:
                    status = "Brawo, dotarles do konca! [r] reset | [q] wyjscie"

                live.update(self._build_view(status), refresh=True)

class QLearningAgent:
    def __init__(
            self,
            world: SimpleWorld,
            alpha: float = 0.1,
            gamma: float = 0.9,
            epsilon: float  = 0.1,
            seed: int = 42,
    ) -> None:
        self.world = world
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)
        self.actions = np.array([ActionEnum.LEFT, ActionEnum.RIGHT], dtype=np.int8)
        self.q_table = np.zeros((world.size, np.size(self.actions)), dtype=np.float64)

    def _action_to_idx(self, action: ActionEnum) -> int:
        matches = np.nonzero(self.actions == int(action))[0]
        if matches.size == 0:
            raise ValueError(f"Unknown action: {action}")
        return int(matches[0])

    def _idx_to_action(self, idx: int) -> ActionEnum:
        return ActionEnum(int(self.actions[idx]))

    def choose_action(self):
        state = self.world.state
        if self.rng.random() < self.epsilon:
            random_idx = self.rng.integers(0,len(self.actions))
            return self._idx_to_action(random_idx)
        best_idx = int(np.argmax(self.q_table[state]))

        return self._idx_to_action(best_idx)

    def learn(
        self,
        state: int,
        action: ActionEnum,
        reward: float,
        next_state: int,
        done: bool = False,
    ) -> None:
        action_idx = self._action_to_idx(action)
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action_idx]
        self.q_table[state, action_idx] += self.alpha * td_error

def train_agent(agent: QLearningAgent, episodes: int = 1000) -> None:
    print(f"Training Q-Learning agent for {episodes} episodes...")

    for episode in range(episodes):
        agent.world.reset()

        for step in range(100):
            state = agent.world.state
            action = agent.choose_action()
            agent.world.step(action)
            # reward = 1.0 if agent.world.done else 0.0
            next_state = agent.world.state

            if next_state == 0:
                reward = -1.0
            elif agent.world.done:
                reward = 1.0
            else:
                reward = -0.01

            agent.learn(state, action, reward, next_state)

        if (episode + 1) % (episodes // 10) == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

    print("Training completed.")
    print(agent.q_table)
    print("\n Running a greedy test:")
    agent.world.reset()
    agent.world.render()

    while not agent.world.done:
        action = agent.choose_action()
        agent.world.step(action)
        agent.world.render()

    print("\n FINISH")

            # done = agent.world.done
            #
            # agent.learn(state, action, reward, next_state, done)
            # if done:
            #     break



if __name__ == "__main__":
    world_ = SimpleWorld(size=15)
    agent_ = QLearningAgent(world = world_)

    train_agent(agent_)