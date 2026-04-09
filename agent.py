"""Agent implementations for RL, fuzzy control, and hybrid control."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import FuzzyConfig, TrainingConfig
from environment import ACTIONS, MOVE_ACTIONS, MOVES, VacuumEnvironment
from fuzzy_controller import FuzzyController


class BaseVacuumAgent:
    name = "base"

    def select_action(self, env: VacuumEnvironment, training: bool = True) -> str:
        raise NotImplementedError

    def observe_transition(
        self,
        state_key,
        action: str,
        reward: float,
        next_state_key,
        done: bool,
    ) -> None:
        return None

    def on_episode_end(self) -> None:
        return None


class QLearningVacuumAgent(BaseVacuumAgent):
    name = "pure_rl"

    def __init__(self, training_config: TrainingConfig):
        self.lr = training_config.learning_rate
        self.gamma = training_config.discount_factor
        self.epsilon = training_config.epsilon_start
        self.epsilon_end = training_config.epsilon_end
        self.epsilon_decay = training_config.epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float32))

    def select_action(self, env: VacuumEnvironment, training: bool = True) -> str:
        state = env.get_state_key()
        valid_actions = env.get_valid_actions()
        if training and np.random.random() < self.epsilon:
            return str(np.random.choice(valid_actions))
        return self._best_valid_action(state, valid_actions)

    def _best_valid_action(self, state, valid_actions: List[str]) -> str:
        q_values = self.q_table[state]
        best_score = float("-inf")
        best_action = valid_actions[0]
        for action in valid_actions:
            score = float(q_values[ACTIONS.index(action)])
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def observe_transition(self, state_key, action: str, reward: float, next_state_key, done: bool) -> None:
        action_idx = ACTIONS.index(action)
        current_q = self.q_table[state_key][action_idx]
        next_max = 0.0 if done else float(np.max(self.q_table[next_state_key]))
        updated = current_q + self.lr * (reward + self.gamma * next_max - current_q)
        self.q_table[state_key][action_idx] = updated

    def on_episode_end(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(dict(self.q_table), handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float32), data)


class FuzzyRuleBasedAgent(BaseVacuumAgent):
    name = "fuzzy_only"

    def __init__(self):
        self.controller = FuzzyController()

    def select_action(self, env: VacuumEnvironment, training: bool = True) -> str:
        obs = env.get_observation()
        mode = self.controller.preferred_mode(
            obs["current_dirt"], obs["battery"], obs["distance_to_charger"]
        )
        valid_actions = obs["valid_actions"]

        if mode == "recharge" and "recharge" in valid_actions:
            return "recharge"
        if mode == "clean" and obs["current_dirt"] > 0:
            return "clean"
        if mode == "move":
            return self._move_towards_goal(env, prefer_charger=obs["battery"] < 35)

        if obs["current_dirt"] > 0:
            return "clean"
        if obs["battery"] < 30:
            return self._move_towards_goal(env, prefer_charger=True)
        return self._move_towards_goal(env, prefer_charger=False)

    def _move_towards_goal(self, env: VacuumEnvironment, prefer_charger: bool) -> str:
        obs = env.get_observation()
        valid_actions = [a for a in obs["valid_actions"] if a in MOVE_ACTIONS]
        if not valid_actions:
            return "clean"

        target = obs["charger_position"] if prefer_charger else self._nearest_visible_dirt(env)
        if target is None:
            return valid_actions[np.random.randint(len(valid_actions))]

        best_action = valid_actions[0]
        best_distance = float("inf")
        for action in valid_actions:
            dx, dy = MOVES[action]
            nx, ny = obs["position"][0] + dx, obs["position"][1] + dy
            distance = abs(nx - target[0]) + abs(ny - target[1])
            if distance < best_distance:
                best_distance = distance
                best_action = action
        return best_action

    def _nearest_visible_dirt(self, env: VacuumEnvironment):
        obs = env.get_observation()
        vx = obs["visible_dirt"]
        pos = obs["position"]
        radius = env.config.visibility_radius
        best = None
        best_distance = float("inf")
        for i in range(vx.shape[0]):
            for j in range(vx.shape[1]):
                if vx[i, j] <= 0:
                    continue
                wx = pos[0] - radius + i
                wy = pos[1] - radius + j
                if 0 <= wx < env.height and 0 <= wy < env.width:
                    distance = abs(wx - pos[0]) + abs(wy - pos[1])
                    if distance < best_distance:
                        best = (wx, wy)
                        best_distance = distance
        return best


class HybridVacuumAgent(QLearningVacuumAgent):
    name = "hybrid_rl_fuzzy"

    def __init__(self, training_config: TrainingConfig, fuzzy_config: FuzzyConfig):
        super().__init__(training_config)
        self.controller = FuzzyController()
        self.fuzzy_config = fuzzy_config

    def select_action(self, env: VacuumEnvironment, training: bool = True) -> str:
        state = env.get_state_key()
        obs = env.get_observation()
        valid_actions = obs["valid_actions"]

        if training and np.random.random() < self.epsilon:
            scores = self._fuzzy_action_scores(env)
            weights = np.array([scores[action] for action in valid_actions], dtype=np.float64)
            weights = weights / weights.sum() if weights.sum() > 0 else None
            return str(np.random.choice(valid_actions, p=weights))

        q_values = self.q_table[state].copy()
        fuzzy_scores = self._fuzzy_action_scores(env)
        for action, score in fuzzy_scores.items():
            q_values[ACTIONS.index(action)] += self.fuzzy_config.q_bias_scale * score

        best_score = float("-inf")
        best_action = valid_actions[0]
        for action in valid_actions:
            value = float(q_values[ACTIONS.index(action)])
            if value > best_score:
                best_score = value
                best_action = action
        return best_action

    def shape_reward(self, env: VacuumEnvironment, action: str, reward: float) -> float:
        scores = self.controller.infer(
            env.get_observation()["current_dirt"],
            env.get_observation()["battery"],
            env.get_observation()["distance_to_charger"],
        )
        if action == "clean":
            return reward + self.fuzzy_config.reward_shaping_scale * scores["clean"]
        if action in MOVE_ACTIONS:
            return reward + self.fuzzy_config.reward_shaping_scale * scores["move"] * 0.5
        if action == "recharge":
            return reward + self.fuzzy_config.reward_shaping_scale * scores["recharge"]
        return reward

    def _fuzzy_action_scores(self, env: VacuumEnvironment) -> Dict[str, float]:
        obs = env.get_observation()
        mode_scores = self.controller.infer(
            obs["current_dirt"], obs["battery"], obs["distance_to_charger"]
        )
        scores = {action: 0.05 for action in ACTIONS}
        scores["clean"] += mode_scores["clean"]
        scores["recharge"] += mode_scores["recharge"]

        if obs["battery"] < 30:
            charger_moves = self._moves_towards_target(obs["position"], obs["charger_position"], obs["valid_actions"])
            for action in charger_moves:
                scores[action] += mode_scores["move"]
        else:
            dirt_target = self._visible_dirt_target(env)
            if dirt_target is not None:
                dirt_moves = self._moves_towards_target(obs["position"], dirt_target, obs["valid_actions"])
                for action in dirt_moves:
                    scores[action] += mode_scores["move"]

        total = sum(scores[action] for action in obs["valid_actions"])
        if total <= 0:
            return {action: 1.0 for action in obs["valid_actions"]}
        return {action: scores[action] / total for action in obs["valid_actions"]}

    def _visible_dirt_target(self, env: VacuumEnvironment):
        return FuzzyRuleBasedAgent()._nearest_visible_dirt(env)

    def _moves_towards_target(self, position, target, valid_actions: List[str]) -> List[str]:
        current_distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
        actions = []
        for action in valid_actions:
            if action not in MOVE_ACTIONS:
                continue
            dx, dy = MOVES[action]
            nx, ny = position[0] + dx, position[1] + dy
            distance = abs(nx - target[0]) + abs(ny - target[1])
            if distance < current_distance:
                actions.append(action)
        return actions or [action for action in valid_actions if action in MOVE_ACTIONS]
