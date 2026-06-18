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

def improve_policy(policy,env:GridWorld, V, discount_factor):
    policy_copy = policy.copy()
    nS = env.num_states
    nA = len(env.action_space)
    for s in range(env.num_states):
        xy = state_to_xy(s, env)
        if xy in env.forbidden_states:
            policy[s] = np.zeros(nA)
            continue
        if env._is_done(xy):
            policy[s] = np.zeros(nA) 
            continue
        q_values = np.zeros(nA)
        for i, a in enumerate(env.action_space):
            ns, r = env._get_next_state_and_reward(xy, a)
            Vs = r + discount_factor * V[xy_to_state(ns, env)]
            q_values[i] = Vs
        best = np.flatnonzero(np.isclose(q_values, q_values.max()))
        policy[s] = np.zeros(nA)
        policy[s, best] = 1.0 / len(best)
        
    policy_stable = (np.allclose(policy, policy_copy)) 
    return policy, policy_stable
            

def main():
    env = GridWorld(env_size=((7, 7)), start_state=(3, 3), target_state=(3, 3), forbidden_states=[])
    state = env.reset()
    env.render()
    policy_mat = np.ones((env.num_states, len(env.action_space))) / len(
        env.action_space
    ) 

    # frame 0: the starting policy and its value function
    V = evaluate_policy(policy_mat, env, discount_factor=discount_factor)
    env.add_state_values(V)
    env.add_policy(policy_mat)
    print("Iteration 0 (initial uniform policy)")
    env.render(animation_interval=1)

    n_iters = 60
    for i in range(n_iters):
        new_policy, policy_stable = improve_policy(policy_mat, env, V, discount_factor=discount_factor)
        Vn = evaluate_policy(new_policy, env, discount_factor=discount_factor)
        if policy_stable:
            env.render()
            env.add_state_values(Vn)
            env.add_policy(new_policy)
            env.render(animation_interval=10)
            print(f"Policy converged after {i+1} iterations")
            break
        env.render()
        env.add_state_values(Vn)
        env.add_policy(new_policy)
        env.render(animation_interval=5)
        policy_mat = new_policy
        V = Vn
            

if __name__ == "__main__":
    main()
            

