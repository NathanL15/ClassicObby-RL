# rl_server.py
# pip install flask torch numpy
from flask import Flask, request, jsonify
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, math
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
    """Dueling network architecture for Double DQN."""
    def __init__(self):
        super().__init__()
        hidden = 256
        self.feature = nn.Sequential(
            nn.Linear(N_OBS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, N_ACT)
        )
    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.val(f)                # (B,1)
        a = self.adv(f)                # (B,A)
        q = v + a - a.mean(1, keepdim=True)
        return q.squeeze(0) if q.size(0) == 1 else q

q = QNet().to(device)
q_tgt = QNet().to(device)
q_tgt.load_state_dict(q.state_dict())
opt = optim.AdamW(q.parameters(), lr=1e-3, weight_decay=1e-4)

gamma = 0.99
eps = 1.0              # start high exploration (will decay)
eps_min = 0.05
eps_decay = 0.999      # slightly faster decay; will apply adaptive bumps on improvements
buf = deque(maxlen=100000)
elite_buf = deque(maxlen=5000)  # preserved for now (will remove/replace with PER in later phase)
bsz = 128
update_every = 2
step_count = 0
current_ep_return = 0.0
best_return = -1e9
current_episode_transitions = []  # holds transitions (s,a,r,sp,d)

# Soft target update factor
tau = 0.005

# Running observation normalization -------------------------------------------------
class RunningNorm:
    def __init__(self, size: int, eps: float = 1e-5, warmup: int = 100):
        self.size = size
        self.eps = eps
        self.warmup = warmup
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.M2 = np.zeros(size, dtype=np.float64)
    def update(self, x: np.ndarray):
        # x shape (size,)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
    def variance(self):
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return self.M2 / (self.count - 1)
    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < self.warmup:
            return x  # do not distort early exploration
        var = self.variance()
        return (x - self.mean) / np.sqrt(var + self.eps)
    def state_dict(self):
        return { 'count': self.count, 'mean': self.mean, 'M2': self.M2 }
    def load_state_dict(self, state):
        self.count = state.get('count', self.count)
        self.mean = state.get('mean', self.mean)
        self.M2 = state.get('M2', self.M2)

obs_norm = RunningNorm(N_OBS)

# Loss & utility
criterion = nn.SmoothL1Loss()
max_grad_norm = 10.0

import os, time
from threading import Lock

# Optional: enable for debugging gradient issues (set to True if still errors)
ENABLE_ANOMALY_DETECT = False
if ENABLE_ANOMALY_DETECT:
    torch.autograd.set_detect_anomaly(True)

train_lock = Lock()

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load best model if exists
best_path = os.path.join(SAVE_DIR, "best.pt")
if os.path.exists(best_path):
    # PyTorch >=2.6 defaults weights_only=True which can fail for older pickled checkpoints.
    try:
        ckpt = torch.load(best_path, map_location=device)
    except Exception as e_safe:
        print(f"Safe load failed ({e_safe}); retrying with weights_only=False (local file assumed trusted).")
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except Exception as e_full:
            print(f"Fallback load also failed: {e_full}. Starting fresh (delete {best_path} if corrupted).")
            ckpt = None
    if ckpt is None:
        print("Proceeding without loading checkpoint.")
    else:
    # Backward compatibility: old checkpoints had a flat 'net.*' key layout.
        try:
            q.load_state_dict(ckpt['q'], strict=False)
            q_tgt.load_state_dict(ckpt['q_tgt'], strict=False)
        except Exception as e:
            print(f"Model state load mismatch (expected new dueling architecture). Continuing with fresh weights. Details: {e}")
        if 'opt' in ckpt:
            try:
                opt.load_state_dict(ckpt['opt'])
            except Exception as e:
                print(f"Opt state load failed: {e}")
        eps = ckpt.get('eps', eps)
        step_count = ckpt.get('step_count', 0)
        best_return = ckpt.get('best_return', best_return)
        if 'obs_norm' in ckpt:
            try:
                obs_norm.load_state_dict(ckpt['obs_norm'])
            except Exception as e:
                print(f"Obs norm load failed: {e}")
        print(f"Loaded best (compat) from {best_path} (eps={eps:.3f}, steps={step_count}, best_return={best_return:.1f}, norm_count={obs_norm.count})")
else:
    print("No saved model found, starting fresh")

def save_checkpoint(name: str, is_best=False):
    path = os.path.join(SAVE_DIR, name)
    torch.save({
        'q': q.state_dict(),
        'q_tgt': q_tgt.state_dict(),
        'opt': opt.state_dict(),
        'eps': eps,
        'step_count': step_count,
        'best_return': best_return,
        'obs_keys': OBS_KEYS,
        'timestamp': time.time(),
        'is_best': is_best,
        'obs_norm': obs_norm.state_dict(),
        'version': 'phase1'
    }, path)


last_obs = None
last_action = None

def to_vec(obs):
    return np.array([float(obs[k]) for k in OBS_KEYS], dtype=np.float32)

def preprocess_state(raw: np.ndarray) -> np.ndarray:
    obs_norm.update(raw)
    return obs_norm.normalize(raw).astype(np.float32)

def select_action(s):
    global eps
    if random.random() < eps:
        return random.randrange(N_ACT)
    with torch.no_grad():
        qv = q(torch.from_numpy(s).to(device))
        if qv.dim() > 1:
            qv = qv[0]
        return int(torch.argmax(qv).item())

def train_step():
    # Ensure only one backward/optimizer step at a time (Flask may be threaded)
    if not train_lock.acquire(blocking=False):
        return  # skip if another thread is training; reduces race risk
    try:
        # Need enough base samples
        if len(buf) < bsz:
            return
        elite_take = 0
        if len(elite_buf) > 0:
            elite_take = min(len(elite_buf), bsz // 4)  # reduce elite influence
        base_take = bsz - elite_take
        batch = []
        if elite_take > 0:
            batch.extend(random.sample(elite_buf, elite_take))
        batch.extend(random.sample(buf, base_take))
        random.shuffle(batch)

        s = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        sp = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

        # Double DQN target
        q_s = q(s)
        q_sa = q_s.gather(1, a)
        with torch.no_grad():
            next_online = q(sp)
            next_actions = next_online.argmax(1, keepdim=True)
            next_target = q_tgt(sp).gather(1, next_actions)
            target = r + gamma * (1 - d) * next_target

        loss = criterion(q_sa, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
        opt.step()

        # Soft target update (no gradients tracked)
        with torch.no_grad():
            for p_tgt, p in zip(q_tgt.parameters(), q.parameters()):
                p_tgt.mul_(1 - tau).add_(p, alpha=tau)
    finally:
        train_lock.release()

@app.route("/step", methods=["POST"])
def step():
    global last_obs, last_action, eps, step_count, current_ep_return, best_return
    data = request.get_json(force=True)
    obs_raw = to_vec(data["obs"])
    obs = preprocess_state(obs_raw)
    reward = float(data.get("reward", 0.0))
    # Optional reward clipping (robustness)
    reward = float(np.clip(reward, -5.0, 5.0))
    done = bool(data.get("done", False))
    current_ep_return += reward

    if last_obs is not None and last_action is not None:
        transition = (last_obs, last_action, reward, obs, 1.0 if done else 0.0)
        buf.append(transition)
        current_episode_transitions.append(transition)
        if step_count % update_every == 0:
            train_step()
        # epsilon decay
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
            # Exploration bump-down on genuine improvement
            eps = max(eps_min, eps * 0.5)
        elif meets_threshold:
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
