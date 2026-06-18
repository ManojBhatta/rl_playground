import sys
from typing import List
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from lib.smallgrid.grid_world import GridWorld

from lib.smallgrid.utils import state_to_xy, xy_to_state
from policy_evaluation import evaluate_policy

discount_factor = 1.0


def value_iteration(env: GridWorld, discount_factor, theta=0.01):
    nA, nS = len(env.action_space), env.num_states
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            xy = state_to_xy(s, env)
            if xy in env.forbidden_states or env._is_done(xy):
                continue
            q_values = np.zeros(nA)
            for i, a in enumerate(env.action_space):
                ns, r = env._get_next_state_and_reward(xy, a)
                ns_idx = xy_to_state(ns, env)
                q_values[i] = r + discount_factor * V[ns_idx]
            max_q = np.max(q_values)
            delta = max(delta, np.abs(max_q - V[s]))
            V[s] = max_q
        if delta < theta:
            break

    policy = np.zeros((nS, nA))
    for s in range(nS):
        xy = state_to_xy(s, env)
        if xy in env.forbidden_states or env._is_done(xy):
            continue
        q_values = np.zeros(nA)
        for i, a in enumerate(env.action_space):
            ns, r = env._get_next_state_and_reward(xy, a)
            ns_idx = xy_to_state(ns, env)
            q_values[i] = r + discount_factor * V[ns_idx]
        best = np.flatnonzero(np.isclose(q_values, q_values.max()))
        policy[s, best] = 1.0 / len(best)
    return policy, V



def main():
    env = GridWorld(
        env_size=((4, 4)), start_state=(3, 3), target_state=(3, 3)
    )
    state = env.reset()
    env.render()

    policy, v = value_iteration(env, discount_factor)
    print(policy)
    env.add_state_values(v)
    env.add_policy(policy)
    env.render(animation_interval=4)


if __name__ == "__main__":
    main()
