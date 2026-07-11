import sys
from pathlib import Path
import numpy as np

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lib.smallgrid.grid_world import GridWorld
from lib.smallgrid.utils import state_to_xy, xy_to_state

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
NUM_EPISODES  = 1000
GAMMA         = 1.0
EPSILON       = 0.55          # exploration rate (annealed over time)
MAX_STEPS     = 1500           # safety cap per episode to prevent infinite loops
FIRST_VISIT   = False          # True → first-visit MC; False → every-visit MC
ALPHA         = 0.1

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
ENV_SIZE      = (5, 5)
START_STATE   = (0, 0)
TARGET_STATE  = (3, 3)
FORB_STATES   = [(2, 1), (3, 2), (1, 3)]

env = GridWorld(
    env_size=ENV_SIZE,
    start_state=START_STATE,
    target_state=TARGET_STATE,
    forbidden_states=FORB_STATES,
)

nS = env.num_states
nA = len(env.action_space)
rng = np.random.default_rng(seed=42)


# ─────────────────────────────────────────────
# Helper: sample a valid random start state
# ─────────────────────────────────────────────

def sample_start_state() -> tuple:
    """Return a random (x, y) that is not the target or a forbidden state."""
    while True:
        candidate = state_to_xy(rng.integers(0, nS), env)
        if candidate != env.target_state:
            return candidate

# ─────────────────────────────────────────────
# Q-table and visit counts
# ─────────────────────────────────────────────
Q     = np.zeros((nS, nA))
count = np.zeros((nS, nA))


# ─────────────────────────────────────────────
# Epsilon-greedy action selection using Q
# ─────────────────────────────────────────────
def select_action(s_idx: int, epsilon: float) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, nA))          # explore
    return int(np.argmax(Q[s_idx]))              # exploit


# ─────────────────────────────────────────────
# TD Control — main loop
# ─────────────────────────────────────────────
episodes_completed = 0

for e in range(NUM_EPISODES):

    # Anneal epsilon: starts at EPSILON, decays toward 0.05
    eps = max(0.05, EPSILON * (1 - e / NUM_EPISODES))

    # Sample a random valid start and reset
    start = sample_start_state()
    env.reset(new_start_state=start)
    current_state = env.agent_state

    # ── Generate one episode ──────────────────
    ep = []          # list of (state_xy, action_idx, reward)

    # update online after each step
    for _ in range(MAX_STEPS):
        s_idx  = xy_to_state(current_state, env)
        a_idx  = select_action(s_idx, eps)
        action = env.action_space[a_idx]

        next_state, reward, done, _ = env.step(action)
        ns_idx = xy_to_state(next_state, env)
        na_idx = select_action(ns_idx, eps)
        next_action = env.action_space[na_idx]

        td_target = reward + GAMMA * Q[ns_idx, na_idx]
        td_error = Q[s_idx, a_idx] - td_target

        Q[s_idx, a_idx] -= ALPHA * td_error

        current_state = next_state

        if done:
            break

    # episode did not reach terminal → skip update
    # (can happen if MAX_STEPS is too small)
    if not done:
        continue

    episodes_completed += 1

print(f"Training complete — {episodes_completed}/{NUM_EPISODES} episodes reached terminal.")

# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────
print("\nLearned Q-table (max Q per state, reshaped to grid):")
print(np.max(Q, axis=1).reshape(ENV_SIZE[1], ENV_SIZE[0]))

print("\nGreedy action indices per state (reshaped to grid):")
print(np.argmax(Q, axis=1).reshape(ENV_SIZE[1], ENV_SIZE[0]))

action_symbols = {(0,1):'↓', (1,0):'→', (0,-1):'↑', (-1,0):'←', (0,0):'·'}
print("\nGreedy policy (arrows):")
for row in range(ENV_SIZE[1]):
    line = ""
    for col in range(ENV_SIZE[0]):
        s_idx  = xy_to_state((col, row), env)
        best_a = int(np.argmax(Q[s_idx]))
        sym    = action_symbols[env.action_space[best_a]]
        line  += f"  {sym}"
    print(line)

# ─────────────────────────────────────────────
# Get the greedy policy from learned Q table
# ─────────────────────────────────────────────
policy = (Q == Q.max(axis=1, keepdims=True)).astype(int)


# ─────────────────────────────────────────────
# Render final policy
# ─────────────────────────────────────────────
env.reset()
env.render()
env.add_policy(policy)
env.render(animation_interval=1)
input("Press Enter to exit...")




