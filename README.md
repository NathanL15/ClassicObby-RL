# 🧠 RL on Roblox: Classic Obby

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Roblox](https://img.shields.io/badge/roblox-studio-red.svg)](https://create.roblox.com/)

Train a **Deep Q-Network (DQN) reinforcement learning agent** to navigate a **Roblox Classic Obby** environment. The agent learns to jump, move, and avoid hazards through trial and error, communicating in real-time between Roblox Studio and a Python server.

## ✨ Features

- **Real-time RL Training**: Agent interacts with Roblox game world via HTTP API
- **Advanced Observations**: Includes kinematics, radial rays, edge probes, hazard detection
- **Smart Reward Shaping**: Distance-based progress, leap bonuses, milestone rewards, hazard avoidance
- **Automatic Checkpointing**: Saves best and latest models during training
- **Elite Replay Buffer**: Retains high-performing episode trajectories to prevent forgetting good strategies
- **Adaptive Exploration**: Epsilon decay adjusts based on performance
- **Hazard Awareness**: Detects and avoids lethal blocks with safe respawn logic

## 🏗️ Architecture

```
Roblox Client (AgentClient.client.lua)
    ↓ RemoteFunction RLStep
Roblox Server (RLServer.lua)
    ↓ HTTP POST
Python Flask Server (rl_server.py)
    ↓ DQN Training Loop
PyTorch Q-Network
```

## 📋 Prerequisites

- **Roblox Studio**: For running the obby environment
- **Python 3.8+**: For the RL server
- **Roblox Game Setup**: Classic Obby with checkpoints (CP_0, CP_1, CP_2, ...) and hazard-tagged parts

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/NathanL15/ClassicObby-RL.git
cd ClassicObby-RL
```

### 2. Set up the Python RL Server

```bash
# Navigate to server directory
cd server

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the DQN server
python rl_server.py
```

### 3. Set up Roblox Environment

1. Open `place/ClassicObby.rbxl` in Roblox Studio
2. Ensure checkpoints are named `CP_0`, `CP_1`, `CP_2`, etc.
3. Tag hazard parts with `CollectionService` tag "Hazard"
4. Add `RemoteFunction` named "RLStep" in `ReplicatedStorage`
5. Insert `roblox/RLServer.lua` as a ServerScript in `ServerScriptService`
6. Insert `roblox/AgentClient.client.lua` as a LocalScript in `StarterPlayerScripts`

### 4. Start Training

1. Run the Python server (from step 2)
2. Play the Roblox game in Studio
3. Watch the agent learn to navigate the obby!

## ⚙️ Configuration

### Agent Parameters (AgentClient.client.lua)

- `STEP_DT`: Internal update interval (0.05s)
- `ACTION_DECISION_DT`: Decision frequency (0.10s)
- `HAZARD_NEAR_RADIUS`: Distance to trigger hazard avoidance (15 studs)
- `CHECK_RADIUS`: Checkpoint reach distance (6 studs)

### Server Parameters (rl_server.py)

- `N_ACT`: Number of actions (7: idle, forward, left, right, jump, forward+jump, backward)
- `gamma`: Discount factor (0.99)
- `eps_min`: Minimum exploration rate (0.05)
- `buf`: Replay buffer size (50,000)
- `elite_buf`: Elite buffer size (5,000)

## 🎮 Actions

| ID | Action | Description |
|----|--------|-------------|
| 0  | Idle   | No movement |
| 1  | Forward| Move forward |
| 2  | Left   | Strafe left |
| 3  | Right  | Strafe right |
| 4  | Jump   | Jump in place |
| 5  | Forward + Jump | Jump while moving forward |
| 6  | Backward | Move backward |

## 📊 Observations

The agent receives 24-dimensional observations:

- **Kinematics**: dx, dy, dz, vx, vy, vz (position/velocity to target)
- **Environment**: down, forward (ray distances)
- **Orientation**: angle (cosine to target), grounded, speed, tJump
- **Radial Rays**: r0-r7 (8 directions for obstacle sensing)
- **Edge Probes**: dropF, dropR, dropL (gap detection ahead/sides)
- **Hazards**: hazardDist (normalized distance to nearest hazard), lastDeathType

## 🏆 Reward Structure

- **Progress**: +1.5 * distance improvement (capped at -2 regress)
- **Leap Bonus**: +1 for jumps >2 studs closer
- **Milestone**: +2 every 1-stud improvement on best distance
- **Checkpoint**: +10 for reaching next CP
- **Penalties**: -10 death, -8 stuck, -0.02 hazard approach
- **Base**: -0.005 per step

## 📁 Project Structure

```
ClassicObby-RL/
├── place/
│   └── ClassicObby.rbxl          # Roblox place file
├── roblox/
│   ├── AgentClient.client.lua    # Client-side agent logic
│   └── RLServer.lua              # Server-side HTTP bridge
├── server/
│   ├── requirements.txt          # Python dependencies
│   └── rl_server.py             # DQN training server
├── checkpoints/                  # Auto-saved models (created on run)
│   ├── best.pt                   # Best performing model
│   └── last.pt                   # Most recent model
└── README.md
```

