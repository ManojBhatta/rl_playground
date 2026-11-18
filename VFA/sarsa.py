"""
Episodic semi-gradient SARSA on CartPole-v1 (Sutton & Barto ch. 10.1).

On-policy TD control with an MLP action-value function, updated online
after every transition (no replay buffer, no target network):

    target = r + γ · Q(s', a')      where a' ~ ε-greedy(Q)(s')

Usage:
    python VFA/sarsa.py --train
    python VFA/sarsa.py --eval [--episodes 10] [--render]
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPSILON = 0.2
LR = 0.005
GAMMA = 0.99
NUM_EPISODES = 2_000
WEIGHTS_PATH = Path(__file__).resolve().parent / "checkpoints" / "sarsa_cartpole.pth"
# ──────────────────────────────────────────────────────────────────────────────


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
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


def train(weights_path: Path):
    # sutton_barto_reward=True gives reward in {-1, 0} instead of always +1,
    # which keeps TD targets in a tight range and helps the NN converge.
    train_env = gym.make("CartPole-v1", sutton_barto_reward=True)
    model = SimpleMLP()
    optimiser = torch.optim.SGD(model.parameters(), lr=LR)

    for _ in tqdm(range(NUM_EPISODES), desc="Training"):
        state, _ = train_env.reset()
        action = get_action(model, state, EPSILON)

        while True:
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            next_action = get_action(model, next_state, EPSILON)

            # SARSA TD target: use the *next chosen action* (on-policy)
            with torch.no_grad():
                next_q = (
                    model(to_tensor(next_state))[next_action]
                    if not done
                    else torch.tensor(0.0)
                )
                td_target = reward + GAMMA * next_q

            current_q = model(to_tensor(state))[action]
            loss = (td_target - current_q) ** 2  # squared TD error, single transition

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if done:
                break
            state, action = next_state, next_action

    train_env.close()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Training complete. Weights saved to {weights_path}")
    return model


def evaluate(weights_path: Path, episodes: int = 10, render: bool = False):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path} (run with --train first)")

    model = SimpleMLP()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    test_env = gym.make("CartPole-v1", render_mode="human" if render else None)
    returns = []
    for episode in range(episodes):
        state, _ = test_env.reset()
        total_reward = 0.0
        while True:
            action = get_greedy_action(model, state)
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
        print(f"Episode {episode + 1}: return = {total_reward:.0f}")

    test_env.close()
    mean_return = np.mean(returns)
    print(f"Mean return over {episodes} eval episodes: {mean_return:.1f}")
    return mean_return


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a CartPole SARSA agent.")
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
