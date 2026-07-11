import sys
from pathlib import Path
import numpy as np
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from lib.smallgrid.grid_world import GridWorld
from lib.smallgrid.utils import state_to_xy, xy_to_state

def generate_episodes(policy, env:GridWorld, num_episodes:int):
    episodes = []
    nS = env.num_states
    nA = len(env.action_space)
    rng  = np.random.default_rng()
    for e in range(num_episodes):
        # an episode will be list of state, action, reward 
        ep = []
        # choose a random start that is not the target or a forbidden state
        new_state = state_to_xy(np.random.randint(nS), env)
        while new_state == env.target_state or new_state in env.forbidden_states:
            new_state = state_to_xy(np.random.randint(nS), env)
        env.reset(new_state)
        current_state = env.agent_state
        while not env._is_done(current_state):
            # get an action from policy
            action_probs = policy[xy_to_state(current_state, env)]
            a = rng.choice(nA, p=action_probs)
            action = env.action_space[a]

            # execute action
            next_state, reward, done, info = env.step(action)

            # add to episode: record the state (xy), action and reward
            ep.append((current_state, action, reward))
            current_state = next_state

        episodes.append(ep) 
    return episodes

def mc_predict(policy, env:GridWorld, num_episodes:int) -> np.ndarray:
    episodes = generate_episodes(policy, env, num_episodes)
    nS = env.num_states
    nA = len(env.action_space)
    # evaluate the policy
    V = np.zeros(nS)
    count = np.zeros(nS)
    gamma = 1.0
    for e in episodes:
        G = 0.0
        visited = set()
        s_idx = xy_to_state(e[0][0], env)
        count[s_idx] += 1
        for s, a, r in reversed(e):
            G = r + gamma * G
        V[s_idx] += (G - V[s_idx]) / count[s_idx]
    return V


# print(V.reshape(env.env_size))
def main():
    num_episodes = 9000
    env_size = (4, 4)
    target_state = (3, 3)
    start_state = (0, 0)
    env = GridWorld(
        env_size=env_size,
        start_state=start_state,
        target_state=target_state,
        forbidden_states=[],
    )
    nS = env.num_states
    nA = len(env.action_space)
    policy = np.ones((nS, nA)) / nA
    V = mc_predict(policy, env, num_episodes)
    print(V.reshape(env.env_size))
    
if __name__ == "__main__":
    main()