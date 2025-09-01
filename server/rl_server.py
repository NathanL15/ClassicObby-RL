# rl_server.py
# pip install flask torch numpy
from flask import Flask, request, jsonify
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random
from collections import deque

app = Flask(__name__)

OBS_KEYS = ["dx","dy","dz","vx","vy","vz","down","forward"]
N_OBS, N_ACT = len(OBS_KEYS), 6

device = torch.device("cpu")

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_OBS, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_ACT)
        )
    def forward(self, x): return self.net(x)

q = QNet().to(device)
q_tgt = QNet().to(device)
q_tgt.load_state_dict(q.state_dict())
opt = optim.Adam(q.parameters(), lr=1e-3)

gamma = 0.99
eps = 1.0
eps_min = 0.05
eps_decay = 0.9995
buf = deque(maxlen=50000)
bsz = 64
update_every = 4
tgt_sync_every = 1000
step_count = 0

last_obs = None
last_action = None

def to_vec(obs):
    return np.array([float(obs[k]) for k in OBS_KEYS], dtype=np.float32)

def select_action(s):
    global eps
    if random.random() < eps:
        return random.randrange(N_ACT)
    with torch.no_grad():
        qv = q(torch.from_numpy(s))
        return int(torch.argmax(qv).item())

def train_step():
    if len(buf) < bsz: return
    batch = random.sample(buf, bsz)
    s = torch.tensor([b[0] for b in batch], dtype=torch.float32)
    a = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
    r = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
    sp = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    d = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

    q_sa = q(s).gather(1, a)
    with torch.no_grad():
        q_next = q_tgt(sp).max(1, keepdim=True)[0]
        target = r + gamma * (1 - d) * q_next
    loss = nn.functional.mse_loss(q_sa, target)
    opt.zero_grad(); loss.backward(); opt.step()

@app.route("/step", methods=["POST"])
def step():
    global last_obs, last_action, eps, step_count
    data = request.get_json(force=True)
    obs = to_vec(data["obs"])
    reward = float(data.get("reward", 0.0))
    done = bool(data.get("done", False))

    if last_obs is not None and last_action is not None:
        buf.append((last_obs, last_action, reward, obs, 1.0 if done else 0.0))
        if step_count % update_every == 0:
            train_step()
        if step_count % tgt_sync_every == 0:
            q_tgt.load_state_dict(q.state_dict())
        eps = max(eps_min, eps * eps_decay)

    action = select_action(obs)

    last_obs = None if done else obs
    last_action = None if done else action
    step_count += 1

    return jsonify({"action": action})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
