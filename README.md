# Autonomous Vacuum Cleaner Agent using Reinforcement Learning and Fuzzy Logic

This project simulates an intelligent vacuum cleaner in a 2D grid world. The agent operates under battery constraints, partial observability, and movement cost. It compares three strategies:

- Pure Q-learning
- Fuzzy-only rule-based control
- Hybrid Q-learning with fuzzy reward shaping and action biasing

## Features

- Configurable grid world with dirt levels, obstacles, and a charging station
- Partial observability with a 1-cell visibility radius
- Q-learning with epsilon-greedy exploration
- Fuzzy inference system with triangular and trapezoidal membership functions
- Comparison pipeline for pure RL, fuzzy-only, and hybrid agents
- Real-time matplotlib visualization
- Metric plots and saved Q-table models
- Optional GIF export for demo episodes

## Project Structure

- `config.py`: Shared configuration dataclasses
- `environment.py`: Grid world simulation and reward logic
- `agent.py`: Pure RL, fuzzy-only, and hybrid agent implementations
- `fuzzy_controller.py`: Fuzzy logic membership functions and rules
- `train.py`: Training, evaluation, CLI, and model persistence
- `visualize.py`: Metric plotting and episode animation
- `demo.py`: Shortcut entrypoint for running the visualization flow

## Environment and Reward Design

The environment defaults to a `10x10` grid and randomly initializes obstacles and dirt. Dirt intensity is encoded as:

- Low dirt: reward `+10`
- Medium dirt: reward `+20`
- High dirt: reward `+30`

Additional rewards and penalties:

- Movement cost: `-1`
- Invalid move or obstacle hit: `-10`
- Empty clean attempt: `-2`
- Invalid recharge: `-4`
- Battery depletion: `-100`
- Fully cleaned map: `+50`

## Fuzzy Logic Integration

The fuzzy controller uses:

- Inputs: dirt level, battery level, distance to charger
- Output modes: clean, move, recharge
- Membership functions: triangular and trapezoidal
- Rules: 10 expert rules covering tradeoffs between cleaning urgency and energy safety

The hybrid agent uses fuzzy logic in two ways:

- It biases exploration and greedy action selection toward fuzzy-preferred behaviors
- It shapes rewards so decisions aligned with fuzzy priorities receive a small bonus

## Setup

Create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pillow
```

## Usage

Train and compare all agents:

```bash
python3 train.py --episodes 300
```

Run a demo visualization for a trained agent:

```bash
python3 demo.py --mode demo --agent hybrid_rl_fuzzy
```

Save the demo animation as a GIF:

```bash
python3 demo.py --mode demo --agent hybrid_rl_fuzzy --save-gif
```

Try a different grid size:

```bash
python3 train.py --episodes 200 --width 12 --height 12 --dynamic-dirt
```

## Outputs

Training stores artifacts in `artifacts/`:

- `comparison.json`: evaluation summary for all agents
- `pure_rl/q_table.pkl`: saved pure RL Q-table
- `hybrid_rl_fuzzy/q_table.pkl`: saved hybrid Q-table
- `pure_rl/training_metrics.png`: pure RL plots
- `hybrid_rl_fuzzy/training_metrics.png`: hybrid RL plots
- `*_demo.gif`: optional animation export

## GPU Note

This implementation uses tabular Q-learning, so it does not require a GPU. If you later want to use remote GPU access, the natural upgrade path is to replace the Q-table with a neural function approximator such as a DQN while keeping the same environment and fuzzy controller interface.

## Expected Results

In typical runs, the hybrid agent should learn safer behavior around battery usage and often outperform the pure RL baseline in cleaning efficiency and stability, while the fuzzy-only agent provides a hand-crafted baseline without learning.
