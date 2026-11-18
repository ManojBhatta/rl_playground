"""
Deep Q-Network on Atari Pong (ALE/Pong-v5).

Nature-DQN style agent with:
  * DeepMind preprocessing: 84x84 grayscale, frame-skip 4, 4-frame stack, no-op starts
  * 3-conv + 2-FC Q-network
  * experience replay buffer
  * target network with hard sync every TARGET_UPDATE_FREQ updates
  * Double-DQN target: online net picks a', target net evaluates it
  * Huber (smooth L1) loss and gradient-norm clipping
  * linearly decaying ε-greedy exploration
  * TensorBoard logging to runs/my_dqn

Usage:
    python VFA/dqn.py --train
    python VFA/dqn.py --eval [--episodes 10] [--render]
"""
import argparse
import copy
import random
from collections import deque
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

gym.register_envs(ale_py)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE_FREQ = 500
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.05
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 1_000
BUFFER_SIZE = 50_000
SAVE_EVERY = 50  # episodes between intermediate checkpoint saves
HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "checkpoints" / "dqn_pong.pth"
LOG_DIR = HERE / "runs" / "my_dqn"
# ──────────────────────────────────────────────────────────────────────────────


def make_env(render: bool = False) -> gym.Env:
    env = gym.make(
        "ALE/Pong-v5",
        render_mode="human" if render else None,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True)
    return FrameStackObservation(env, stack_size=4)


class ExperienceReplay:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )


class DQNet(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


def get_greedy_action(net: DQNet, state):
    with torch.no_grad():
        return int(torch.argmax(net(to_tensor(state))).item())


def get_eps_greedy_action(net: DQNet, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return get_greedy_action(net, state)


def update_network(
    online: DQNet,
    target: DQNet,
    optimiser: torch.optim.Optimizer,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
):
    states_t = to_tensor(states)
    next_states_t = to_tensor(next_states)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    dones_t = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

    with torch.no_grad():
        # Double DQN: select a' with the online net, evaluate it with the target net
        next_best_act = online(next_states_t).argmax(dim=1, keepdim=True)
        next_q_target = target(next_states_t).gather(1, next_best_act)
        next_q_target[dones_t] = 0.0
        td_target = rewards_t + GAMMA * next_q_target

    current_q = online(states_t).gather(1, actions_t)
    loss = F.smooth_l1_loss(current_q, td_target)

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimiser.step()

    return loss.item()


def evaluate(weights_path: Path, episodes: int = 10, render: bool = False):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path} (run with --train first)")

    env = make_env(render=render)
    model = DQNet(act_dim=env.action_space.n)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    returns = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            action = get_greedy_action(model, state)
            state, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        returns.append(total_reward)
        print(f"Episode {episode + 1}: return = {total_reward:.0f}")

    env.close()
    print(f"Mean return over {episodes} eval episodes: {np.mean(returns):.1f}")


def train(weights_path: Path):
    env = make_env()
    n_actions = env.action_space.n
    online = DQNet(act_dim=n_actions)
    target = copy.deepcopy(online)
    target.eval()

    optimiser = torch.optim.Adam(online.parameters(), lr=LR)
    buffer = ExperienceReplay()
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    global_step = 0

    for episode in tqdm(range(NUM_EPISODES), desc="Training", unit="episode"):
        state, _ = env.reset()
        total_reward = 0.0
        epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / max(1, NUM_EPISODES - 1)),
        )

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = get_eps_greedy_action(online, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, done)

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = update_network(online, target, optimiser, *batch)
                writer.add_scalar("train/loss", loss, global_step)
                global_step += 1

                if global_step % TARGET_UPDATE_FREQ == 0:
                    target.load_state_dict(online.state_dict())

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        writer.add_scalar("reward/episode", total_reward, episode)
        writer.add_scalar("reward/rolling_mean_100", np.mean(episode_rewards[-100:]), episode)
        writer.add_scalar("training/epsilon", epsilon, episode)
        tqdm.write(f"Episode {episode + 1}/{NUM_EPISODES} | reward={total_reward:.1f} | epsilon={epsilon:.3f}")

        if (episode + 1) % SAVE_EVERY == 0:
            torch.save(online.state_dict(), weights_path)

    torch.save(online.state_dict(), weights_path)
    writer.flush()
    writer.close()
    env.close()
    print(f"Saved model to {weights_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent on Atari Pong.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train a new agent and save its weights.")
    mode.add_argument("--eval", action="store_true", help="Load saved weights and run greedy evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=WEIGHTS_PATH, help="Path to model weights.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval:
        evaluate(args.checkpoint, episodes=args.episodes, render=args.render)
    else:
        train(args.checkpoint)


if __name__ == "__main__":
    main()
