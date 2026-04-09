"""Project configuration for the autonomous vacuum cleaner simulation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvironmentConfig:
    width: int = 10
    height: int = 10
    obstacle_ratio: float = 0.12
    dirt_ratio: float = 0.28
    max_steps: int = 250
    battery_capacity: int = 100
    battery_move_cost: int = 2
    battery_clean_cost: int = 4
    battery_idle_cost: int = 1
    visibility_radius: int = 1
    dynamic_dirt: bool = False
    dirt_regen_prob: float = 0.02
    seed: int = 42


@dataclass
class TrainingConfig:
    episodes: int = 300
    learning_rate: float = 0.15
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.992
    evaluation_episodes: int = 20
    battery_buckets: int = 5
    save_dir: Path = field(default_factory=lambda: Path("artifacts"))
    log_interval: int = 25


@dataclass
class VisualizationConfig:
    cell_size: int = 48
    interval_ms: int = 250
    dpi: int = 110
    save_gif: bool = False
    gif_fps: int = 4


@dataclass
class FuzzyConfig:
    reward_shaping_scale: float = 2.5
    q_bias_scale: float = 1.4


@dataclass
class ProjectConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
