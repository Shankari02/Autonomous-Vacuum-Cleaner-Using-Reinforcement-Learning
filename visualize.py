"""Visualization utilities for the vacuum cleaner project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap

from config import VisualizationConfig


def plot_training_metrics(history: Dict[str, List[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["rewards"], color="#1f77b4")
    axes[0].set_title("Reward vs Episodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    axes[1].plot(history["steps"], color="#ff7f0e")
    axes[1].set_title("Steps vs Episodes")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    axes[2].plot(history["efficiency"], color="#2ca02c")
    axes[2].set_title("Cleaning Efficiency")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Dirt Cleaned / Step")

    fig.tight_layout()
    fig.savefig(output_dir / "training_metrics.png", bbox_inches="tight")
    plt.close(fig)


def visualize_episode(
    frames: List[np.ndarray],
    stats: Optional[List[Dict]] = None,
    config: Optional[VisualizationConfig] = None,
    output_path: Optional[Path] = None,
) -> None:
    config = config or VisualizationConfig()
    cmap = ListedColormap(
        [
            "#f3f0e8",  # empty
            "#d6e685",  # low dirt
            "#8cc665",  # medium dirt
            "#44a340",  # high dirt
            "#ffc857",  # charger
            "#2d728f",  # agent
            "#555b6e",  # obstacle
        ]
    )

    def remap(frame: np.ndarray) -> np.ndarray:
        mapped = frame.copy()
        mapped[mapped == -1] = 6
        return mapped

    fig, ax = plt.subplots(figsize=(7, 7), dpi=config.dpi)
    image = ax.imshow(remap(frames[0]), cmap=cmap, vmin=0, vmax=6)
    ax.set_title("Autonomous Vacuum Cleaner")
    ax.set_xticks(np.arange(-0.5, frames[0].shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frames[0].shape[0], 1), minor=True)
    ax.grid(which="minor", color="#3d3d3d", linewidth=0.6)
    status_text = ax.text(
        0.02,
        -0.08,
        "",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
    )

    def update(frame_idx: int):
        image.set_array(remap(frames[frame_idx]))
        if stats:
            item = stats[frame_idx]
            status_text.set_text(
                f"Step {item['step']} | Battery {item['battery']} | Reward {item['reward']:.1f}"
            )
        return image, status_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=config.interval_ms, blit=False, repeat=False
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".gif":
            ani.save(output_path, writer="pillow", fps=config.gif_fps)
        else:
            ani.save(output_path)
        plt.close(fig)
        return

    plt.show()
