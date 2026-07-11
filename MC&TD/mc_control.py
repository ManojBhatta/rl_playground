import sys
from pathlib import Path
import numpy as np
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from lib.smallgrid.grid_world import GridWorld
from lib.smallgrid.utils import state_to_xy, xy_to_state

rng  = np.random.default_rng()

# model free prediction
# meaning evaluating a policy

num_episodes = 10000
gamma = 1.0
epsilon = 0.5
env_size = (5, 5)
episodes = []
target_state = (3, 3)
start_state = (0, 0)
forb_states = [(2, 1), (3, 2), (1, 3)]
env = GridWorld(
    env_size=env_size,
    start_state=start_state,
    target_state=target_state,
    forbidden_states=forb_states,
)
ss = env.reset()
env.render()
nS = env.num_states
nA = len(env.action_space)
policy = np.ones((nS, nA)) / nA
Q = np.zeros((nS, nA))

count = np.zeros((nS, nA))
for e in range(num_episodes):
    # an episode will be list of state, action, reward 
    ep = []
    # choose a random start that is not the target or a forbidden state
    new_state = state_to_xy(np.random.randint(nS), env)
    while new_state == env.target_state or new_state in env.forbidden_states:
        new_state = state_to_xy(np.random.randint(nS), env)
    env.reset(new_state)

    current_state = env.agent_state

    while True:
        # epsilon-greedy action selection
        s_idx = xy_to_state(current_state, env)
        if rng.random() < epsilon:
            # explore: random action
            a = rng.integers(0, nA)
        else:
            # exploit: greedy action
            a = np.argmax(Q[s_idx])
        action = env.action_space[a]

        # execute action
        next_state, reward, done, info = env.step(action)

        # add to episode: record the state (xy), action and reward
        ep.append((current_state, action, reward))
        current_state = next_state
        if done:
            G = 0.0
            # update the action-value function using every-visit MC
            for s, a, r in reversed(ep):
                G = r + gamma * G
                s_idx = xy_to_state(s, env)
                a_idx = env.action_space.index(a)
                count[s_idx, a_idx] += 1
                Q[s_idx, a_idx] += (G - Q[s_idx, a_idx]) / count[s_idx, a_idx]

                # update the policy to be greedy w.r.t Q (break ties by argmax)
                best_a = np.argmax(Q[s_idx])
                new_pi = np.zeros(nA)
                new_pi[best_a] = 1.0
                policy[s_idx] = new_pi
            break
    if e % 500 == 0:
        env.add_policy(policy)
        env.render()

print('Episodes completed')
sss = env.reset()
env.add_policy(policy)
print(policy)
env.render(animation_interval=5)
input()