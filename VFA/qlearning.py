"""
Q-learning with a neural-network action-value function on CartPole-v1.

Off-policy TD control (Sutton & Barto ch. 6.5 / 10) with:
  * MLP Q(s, ·) approximator
  * experience replay buffer (uniform sampling)
  * linearly decaying ε-greedy behaviour policy
  * greedy target:  r + γ · max_a' Q(s', a')   (no separate target network)

Usage:
    python VFA/qlearning.py --train
    python VFA/qlearning.py --eval [--episodes 10] [--render]
"""
import argparse
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPSILON_START = 1.0
EPSILON_END = 0.05
WARMUP_STEPS = 500
TOTAL_TIMESTEPS = 90_000
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 1.00
BUFFER_CAPACITY = 4_000
WEIGHTS_PATH = Path(__file__).resolve().parent / "checkpoints" / "qlearning_cartpole.pth"
# ──────────────────────────────────────────────────────────────────────────────


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def to_tensor(state: np.ndarray) -> torch.Tensor:
    """Convert a numpy state array to a float32 tensor."""
    return torch.tensor(state, dtype=torch.float32)


def get_action(model: nn.Module, state: np.ndarray, epsilon: float) -> int:
    """ε-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(2)
    return get_greedy_action(model, state)


def get_greedy_action(model: nn.Module, state: np.ndarray) -> int:
    with torch.no_grad():
        return torch.argmax(model(to_tensor(state))).item()


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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

    def __len__(self):
        return len(self.buffer)


def update_network(model, optimiser, states, actions, rewards, next_states, dones):
    states = to_tensor(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = to_tensor(rewards)
    next_states = to_tensor(next_states)
    dones = torch.tensor(dones, dtype=torch.bool)

    # Q-learning target: bootstrap from the greedy action in s' (off-policy)
    next_q = model(next_states).max(dim=1).values
    next_q[dones] = 0.0
    td_target = rewards + GAMMA * next_q

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(current_q, td_target)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.item()


def train(weights_path: Path):
    env = gym.make("CartPole-v1")
    model = SimpleMLP()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    buffer = ReplayBuffer()

    # fill the buffer with random experience before learning starts
    state, _ = env.reset()
    for _ in range(WARMUP_STEPS):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = env.reset()[0] if done else next_state

    state, _ = env.reset()
    for step in tqdm(range(TOTAL_TIMESTEPS), desc="Training"):
        eps = EPSILON_START - (EPSILON_START - EPSILON_END) * (step / TOTAL_TIMESTEPS)
        action = get_action(model, state, eps)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)

        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        update_network(model, optimiser, states, actions, rewards, next_states, dones)

        state = env.reset()[0] if done else next_state

    env.close()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Saved trained model to {weights_path}")
    return model


def evaluate(weights_path: Path, episodes: int = 10, render: bool = False):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path} (run with --train first)")

    model = SimpleMLP()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    returns = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            action = get_greedy_action(model, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
        print(f"Episode {episode + 1}: return = {total_reward:.0f}")

    env.close()
    mean_return = np.mean(returns)
    print(f"Mean return over {episodes} eval episodes: {mean_return:.1f}")
    return mean_return


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a CartPole Q-learning agent.")
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
