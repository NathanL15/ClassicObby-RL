# RL on Roblox Classic Obby

This project connects a Roblox obby (LocalScript client) to a Python DQN server.

## Quick start

### 1) Python server
```bash
cd server
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python rl_server.py
