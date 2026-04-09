"""Grid world environment for the autonomous vacuum cleaner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import EnvironmentConfig


DIRT_REWARD = {0: 0, 1: 20, 2: 35, 3: 50}
ACTIONS = ["up", "down", "left", "right", "clean", "recharge"]
MOVE_ACTIONS = {"up", "down", "left", "right"}
MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class StepResult:
    observation: Dict
    reward: float
    done: bool
    info: Dict


class VacuumEnvironment:
    """A partially observable 2D cleaning environment."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.height = config.height
        self.width = config.width
        self.charger_pos = (0, 0)
        self.agent_pos = (0, 0)
        self.battery = config.battery_capacity
        self.steps = 0
        self.total_initial_dirt = 0
        self.dirt = np.zeros((self.height, self.width), dtype=np.int8)
        self.obstacles = np.zeros((self.height, self.width), dtype=bool)
        self.cleaned_total = 0

    def reset(self, seed: int | None = None) -> Dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.battery = self.config.battery_capacity
        self.cleaned_total = 0
        self.obstacles.fill(False)
        self.dirt.fill(0)

        self.charger_pos = self._random_empty_cell()
        self.agent_pos = self.charger_pos

        obstacle_count = int(self.width * self.height * self.config.obstacle_ratio)
        dirt_count = int(self.width * self.height * self.config.dirt_ratio)

        self._place_random_obstacles(obstacle_count)
        self._place_random_dirt(dirt_count)
        self.total_initial_dirt = int(self.dirt.sum())
        return self.get_observation()

    def _random_empty_cell(self) -> Tuple[int, int]:
        return (
            int(self.rng.integers(0, self.height)),
            int(self.rng.integers(0, self.width)),
        )

    def _place_random_obstacles(self, count: int) -> None:
        placed = 0
        while placed < count:
            cell = self._random_empty_cell()
            if cell == self.charger_pos:
                continue
            if self.obstacles[cell]:
                continue
            self.obstacles[cell] = True
            placed += 1

    def _place_random_dirt(self, count: int) -> None:
        placed = 0
        while placed < count:
            cell = self._random_empty_cell()
            if self.obstacles[cell] or cell == self.charger_pos:
                continue
            if self.dirt[cell] != 0:
                continue
            self.dirt[cell] = int(self.rng.integers(1, 4))
            placed += 1

    def get_valid_actions(self) -> List[str]:
        valid = ["clean"]
        for action, delta in MOVES.items():
            nx, ny = self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]
            if 0 <= nx < self.height and 0 <= ny < self.width and not self.obstacles[nx, ny]:
                valid.append(action)
        if self.agent_pos == self.charger_pos:
            valid.append("recharge")
        return valid

    def get_observation(self) -> Dict:
        x, y = self.agent_pos
        radius = self.config.visibility_radius
        xmin, xmax = max(0, x - radius), min(self.height, x + radius + 1)
        ymin, ymax = max(0, y - radius), min(self.width, y + radius + 1)

        visible_dirt = self.dirt[xmin:xmax, ymin:ymax]
        visible_obstacles = self.obstacles[xmin:xmax, ymin:ymax]
        charger_distance = abs(x - self.charger_pos[0]) + abs(y - self.charger_pos[1])
        total_dirt_remaining = int(self.dirt.sum())

        return {
            "position": self.agent_pos,
            "current_dirt": int(self.dirt[x, y]),
            "battery": self.battery,
            "visible_dirt": visible_dirt.copy(),
            "visible_obstacles": visible_obstacles.copy(),
            "charger_position": self.charger_pos,
            "distance_to_charger": charger_distance,
            "dirt_remaining": total_dirt_remaining,
            "valid_actions": self.get_valid_actions(),
        }

    def get_state_key(self, battery_buckets: int = 5) -> Tuple[int, int, int, int, int, int]:
        obs = self.get_observation()
        battery_bucket = min(
            battery_buckets - 1,
            int(obs["battery"] / max(1, self.config.battery_capacity / battery_buckets)),
        )
        visible_dirt_level = int(obs["visible_dirt"].sum())
        visible_bucket = min(4, visible_dirt_level // 2)
        charger_bucket = min(3, obs["distance_to_charger"] // 3)
        return (
            obs["position"][0],
            obs["position"][1],
            obs["current_dirt"],
            battery_bucket,
            visible_bucket,
            charger_bucket,
        )

    def step(self, action: str) -> StepResult:
        reward = 0.0
        done = False
        info = {"cleaned": 0, "action": action}
        self.steps += 1

        if action in MOVE_ACTIONS:
            reward += self._move(action)
        elif action == "clean":
            reward += self._clean()
        elif action == "recharge":
            reward += self._recharge()
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.config.dynamic_dirt:
            self._regenerate_dirt()

        # Add an early warning penalty so the agent learns to avoid
        # drifting into critically low battery states.
        if 0 < self.battery <= max(self.config.battery_move_cost * 3, 10):
            reward -= 8


        if self.battery <= 0:
            reward -= 100
            done = True
            info["battery_depleted"] = True

        if int(self.dirt.sum()) == 0:
            reward += 50
            done = True
            info["all_clean"] = True

        if self.steps >= self.config.max_steps:
            done = True
            info["max_steps"] = True

        return StepResult(self.get_observation(), reward, done, info)

    def _move(self, action: str) -> float:
        dx, dy = MOVES[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        self.battery -= self.config.battery_move_cost
        if not (0 <= nx < self.height and 0 <= ny < self.width):
            return -10
        if self.obstacles[nx, ny]:
            return -10
        self.agent_pos = (nx, ny)
        return -1

    def _clean(self) -> float:
        x, y = self.agent_pos
        self.battery -= self.config.battery_clean_cost
        dirt_level = int(self.dirt[x, y])
        if dirt_level == 0:
            return -2
        reward = DIRT_REWARD[dirt_level]
        self.cleaned_total += dirt_level
        self.dirt[x, y] = 0
        return reward

    def _recharge(self) -> float:
        self.battery -= self.config.battery_idle_cost
        if self.agent_pos != self.charger_pos:
            return -4
        charge_gain = self.config.battery_capacity - self.battery
        self.battery = self.config.battery_capacity
        return 5 if charge_gain > 0 else -1

    def _regenerate_dirt(self) -> None:
        if self.rng.random() > self.config.dirt_regen_prob:
            return
        candidates = np.argwhere((~self.obstacles) & (self.dirt == 0))
        if len(candidates) == 0:
            return
        idx = int(self.rng.integers(0, len(candidates)))
        x, y = candidates[idx]
        if (int(x), int(y)) == self.charger_pos:
            return
        self.dirt[int(x), int(y)] = int(self.rng.integers(1, 4))

    def render_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.height, self.width), dtype=np.int8)
        matrix[self.obstacles] = -1
        matrix[self.dirt > 0] = self.dirt[self.dirt > 0]
        cx, cy = self.charger_pos
        matrix[cx, cy] = 4
        ax, ay = self.agent_pos
        matrix[ax, ay] = 5
        return matrix

    def cleaning_efficiency(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.cleaned_total / self.steps
