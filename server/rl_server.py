# rl_server.py
# pip install flask torch numpy
from flask import Flask, request, jsonify
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random
from collections import deque

app = Flask(__name__)

OBS_KEYS = [
    "dx","dy","dz","vx","vy","vz","down","forward","angle","grounded","speed","tJump",
    "r0","r1","r2","r3","r4","r5","r6","r7",
    "dropF","dropR","dropL"
    ,"hazardDist","lastDeathType"
]
N_OBS, N_ACT = len(OBS_KEYS), 7  # added backward action (6)

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
elite_buf = deque(maxlen=5000)  # stores transitions from top episodes
bsz = 64
update_every = 4
tgt_sync_every = 1000
step_count = 0
current_ep_return = 0.0
best_return = -1e9
current_episode_transitions = []  # holds transitions (s,a,r,sp,d) for current episode

import os, time

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_checkpoint(name: str, is_best=False):
    path = os.path.join(SAVE_DIR, name)
    torch.save({
        'q': q.state_dict(),
        'q_tgt': q_tgt.state_dict(),
        'eps': eps,
        'step_count': step_count,
        'best_return': best_return,
        'obs_keys': OBS_KEYS,
        'timestamp': time.time(),
        'is_best': is_best,
    }, path)


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
    # Need enough base samples
    base_len = len(buf)
    if base_len < bsz:
        return
    # Determine how many elite samples to include (up to 50%)
    elite_take = 0
    if len(elite_buf) > 0:
        elite_take = min(len(elite_buf), bsz // 2)
    base_take = bsz - elite_take
    batch = []
    if elite_take > 0:
        batch.extend(random.sample(elite_buf, elite_take))
    batch.extend(random.sample(buf, base_take))
    # Shuffle combined batch
    random.shuffle(batch)
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
    global last_obs, last_action, eps, step_count, current_ep_return, best_return
    data = request.get_json(force=True)
    obs = to_vec(data["obs"])
    reward = float(data.get("reward", 0.0))
    done = bool(data.get("done", False))
    current_ep_return += reward

    if last_obs is not None and last_action is not None:
        transition = (last_obs, last_action, reward, obs, 1.0 if done else 0.0)
        buf.append(transition)
        current_episode_transitions.append(transition)
        if step_count % update_every == 0:
            train_step()
        if step_count % tgt_sync_every == 0:
            q_tgt.load_state_dict(q.state_dict())
        eps = max(eps_min, eps * eps_decay)

    action = select_action(obs)

    if done:
        # episode finished: save latest checkpoint
        save_checkpoint("last.pt", is_best=False)
        # Elite criteria: improve best return or reach at least 95% of best (if best established)
        is_new_best = current_ep_return > best_return
        meets_threshold = (best_return > -1e8) and (current_ep_return >= 0.95 * best_return)
        if is_new_best or meets_threshold:
            # Copy transitions to elite buffer
            for tr in current_episode_transitions:
                elite_buf.append(tr)
        if is_new_best:
            best_return = current_ep_return
            save_checkpoint("best.pt", is_best=True)
            # Strongly reduce exploration after new best
            eps = max(eps_min, eps * 0.7)
        elif meets_threshold:
            # Mild reduction to preserve exploiting good policy
            eps = max(eps_min, eps * 0.9)
        # reset episode accumulator
        current_ep_return = 0.0
        current_episode_transitions.clear()
        last_obs = None
        last_action = None
    else:
        last_obs = obs
        last_action = action
    step_count += 1

    return jsonify({"action": action})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
