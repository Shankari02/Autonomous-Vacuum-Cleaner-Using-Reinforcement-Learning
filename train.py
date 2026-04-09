"""Training, evaluation, and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from agent import FuzzyRuleBasedAgent, HybridVacuumAgent, QLearningVacuumAgent
from config import ProjectConfig
from environment import VacuumEnvironment
from visualize import plot_training_metrics, visualize_episode


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("vacuum_rl")


def run_episode(env: VacuumEnvironment, agent, training: bool = True, capture: bool = False):
    observation = env.reset()
    done = False
    total_reward = 0.0
    frames = [env.render_matrix()] if capture else []
    frame_stats = [{"step": 0, "battery": observation["battery"], "reward": 0.0}] if capture else []

    while not done:
        state_key = env.get_state_key()
        action = agent.select_action(env, training=training)
        result = env.step(action)
        reward = result.reward
        if hasattr(agent, "shape_reward") and training:
            reward = agent.shape_reward(env, action, reward)
        next_state_key = env.get_state_key()
        if training:
            agent.observe_transition(state_key, action, reward, next_state_key, result.done)
        total_reward += reward
        done = result.done
        if capture:
            frames.append(env.render_matrix())
            frame_stats.append(
                {"step": env.steps, "battery": env.battery, "reward": total_reward}
            )

    if training:
        agent.on_episode_end()

    metrics = {
        "reward": total_reward,
        "steps": env.steps,
        "efficiency": env.cleaning_efficiency(),
        "dirt_remaining": int(env.dirt.sum()),
    }
    return metrics, frames, frame_stats


def train_agent(agent, env: VacuumEnvironment, episodes: int, logger: logging.Logger, log_interval: int):
    history = {"rewards": [], "steps": [], "efficiency": []}
    for episode in range(1, episodes + 1):
        metrics, _, _ = run_episode(env, agent, training=True, capture=False)
        history["rewards"].append(metrics["reward"])
        history["steps"].append(metrics["steps"])
        history["efficiency"].append(metrics["efficiency"])
        if episode % log_interval == 0 or episode == 1 or episode == episodes:
            logger.info(
                "%s episode %d/%d | reward=%.2f steps=%d efficiency=%.3f epsilon=%s",
                agent.name,
                episode,
                episodes,
                metrics["reward"],
                metrics["steps"],
                metrics["efficiency"],
                f"{getattr(agent, 'epsilon', 'n/a'):.3f}" if hasattr(agent, "epsilon") else "n/a",
            )
    return history


def evaluate_agent(agent, env: VacuumEnvironment, episodes: int) -> Dict[str, float]:
    rewards: List[float] = []
    steps: List[int] = []
    efficiencies: List[float] = []
    dirt_remaining: List[int] = []
    for _ in range(episodes):
        metrics, _, _ = run_episode(env, agent, training=False, capture=False)
        rewards.append(metrics["reward"])
        steps.append(metrics["steps"])
        efficiencies.append(metrics["efficiency"])
        dirt_remaining.append(metrics["dirt_remaining"])

    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "avg_efficiency": mean(efficiencies),
        "avg_dirt_remaining": mean(dirt_remaining),
    }


def compare_agents(config: ProjectConfig, logger: logging.Logger, output_dir: Path):
    env = VacuumEnvironment(config.environment)
    rl_agent = QLearningVacuumAgent(config.training)
    hybrid_agent = HybridVacuumAgent(config.training, config.fuzzy)
    fuzzy_agent = FuzzyRuleBasedAgent()

    logger.info("Training pure RL agent")
    rl_history = train_agent(
        rl_agent, env, config.training.episodes, logger, config.training.log_interval
    )
    plot_training_metrics(rl_history, output_dir / "pure_rl")
    rl_agent.save(output_dir / "pure_rl" / "q_table.pkl")

    logger.info("Training hybrid RL + fuzzy agent")
    hybrid_history = train_agent(
        hybrid_agent, env, config.training.episodes, logger, config.training.log_interval
    )
    plot_training_metrics(hybrid_history, output_dir / "hybrid_rl_fuzzy")
    hybrid_agent.save(output_dir / "hybrid_rl_fuzzy" / "q_table.pkl")

    logger.info("Evaluating all agents")
    comparison = {
        "pure_rl": evaluate_agent(rl_agent, env, config.training.evaluation_episodes),
        "fuzzy_only": evaluate_agent(fuzzy_agent, env, config.training.evaluation_episodes),
        "hybrid_rl_fuzzy": evaluate_agent(
            hybrid_agent, env, config.training.evaluation_episodes
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    return rl_agent, hybrid_agent, fuzzy_agent, comparison


def demo_agent(agent_name: str, config: ProjectConfig, output_dir: Path, save_animation: bool = False):
    env = VacuumEnvironment(config.environment)
    agents = {
        "pure_rl": QLearningVacuumAgent(config.training),
        "fuzzy_only": FuzzyRuleBasedAgent(),
        "hybrid_rl_fuzzy": HybridVacuumAgent(config.training, config.fuzzy),
    }
    agent = agents[agent_name]

    model_path = output_dir / agent_name / "q_table.pkl"
    if hasattr(agent, "load") and model_path.exists():
        agent.load(model_path)

    metrics, frames, frame_stats = run_episode(env, agent, training=False, capture=True)
    animation_path = None
    if save_animation:
        animation_path = output_dir / f"{agent_name}_demo.gif"
    visualize_episode(
        frames,
        stats=frame_stats,
        config=config.visualization,
        output_path=animation_path,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous vacuum cleaner RL project")
    parser.add_argument("--episodes", type=int, default=None, help="Training episodes")
    parser.add_argument(
        "--mode",
        choices=["train", "demo"],
        default="train",
        help="Run training/evaluation or demo visualization",
    )
    parser.add_argument(
        "--agent",
        choices=["pure_rl", "fuzzy_only", "hybrid_rl_fuzzy"],
        default="hybrid_rl_fuzzy",
        help="Agent to visualize in demo mode",
    )
    parser.add_argument("--width", type=int, default=None, help="Grid width")
    parser.add_argument("--height", type=int, default=None, help="Grid height")
    parser.add_argument("--dynamic-dirt", action="store_true", help="Enable dirt regeneration")
    parser.add_argument("--save-gif", action="store_true", help="Save demo animation as a GIF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = build_logger()
    config = ProjectConfig()

    if args.episodes is not None:
        config.training.episodes = args.episodes
    if args.width is not None:
        config.environment.width = args.width
    if args.height is not None:
        config.environment.height = args.height
    if args.dynamic_dirt:
        config.environment.dynamic_dirt = True

    output_dir = config.training.save_dir

    if args.mode == "train":
        _, _, _, comparison = compare_agents(config, logger, output_dir)
        logger.info("Comparison summary: %s", json.dumps(comparison, indent=2))
    else:
        metrics = demo_agent(args.agent, config, output_dir, save_animation=args.save_gif)
        logger.info("Demo metrics: %s", metrics)


if __name__ == "__main__":
    main()
