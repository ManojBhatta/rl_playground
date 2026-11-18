import sys
from typing import List
import numpy as np
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path:
    sys.path.append(root)
from lib.smallgrid.grid_world import GridWorld
from lib.smallgrid.utils import state_to_xy, xy_to_state


def evaluate_policy(
    policy: np.ndarray, env: GridWorld, discount_factor: float = 1.0, theta: float = 0.01
) -> List:
    V = np.zeros(env.num_states)
    i = 0
    while True:
        delta = 0
        for s in range(env.num_states):
            v = 0
            state = state_to_xy(s, env)
            if state in env.forbidden_states:
                V[s] = 0
                continue
            if env._is_done(state):
                V[s] = 0
                continue
            for i, action_prob in enumerate(policy[s]):
                action = env.action_space[i]
                new_state, reward = env._get_next_state_and_reward(state, action)
                next_idx = xy_to_state(new_state, env)
                if env._is_done(new_state):
                    v += action_prob * reward
                else:
                    v += action_prob * (reward + discount_factor * V[next_idx])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def main():
    env = GridWorld(env_size=((4, 4)), start_state=(3, 3), target_state=(3, 3))
    state = env.reset()
    env.render()
    policy_mat = np.ones((env.num_states, len(env.action_space))) / len(
        env.action_space
    )
    v = evaluate_policy(policy_mat, env)
    print()
    env.add_state_values(v)
    env.render(animation_interval=9)


if __name__ == "__main__":
    main()
