"""
Vanilla policy gradient with a learned state-value baseline on Acrobot-v1.

REINFORCE with baseline (Sutton & Barto ch. 13.4):
  * value network V(s) regressed onto Monte-Carlo reward-to-go returns
  * advantage A_t = G_t - V(s_t), normalised per batch
  * policy gradient weighted by A_t instead of G_t (lower variance)

Usage:
    python PG/vpg_w_baseline.py --train
    python PG/vpg_w_baseline.py --eval [--episodes 10] [--render] [--greedy]
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

NUM_EPOCHS = 200
BATCH_SIZE = 4000  # target *timesteps* per batch, not episode count
ENV = "Acrobot-v1"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
POLICY_CHECKPOINT = CKPT_DIR / "vpg_baseline_policy_acrobot.pt"
VALUE_CHECKPOINT = CKPT_DIR / "vpg_baseline_value_acrobot.pt"


class Policy(nn.Module):
    def __init__(self, nobs, nact):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nobs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, nact),
        )

    def forward(self, state):
        return self.net(state)


class ValueBaseline(nn.Module):
    def __init__(self, nobs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nobs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state):
        # squeeze last dim: (batch, 1) -> (batch,) so it matches returns/advantages shape
        return self.net(state).squeeze(-1)


def get_policy(model, obs):
    logits = model(obs)
    return torch.distributions.Categorical(logits=logits)


def get_action(state, policy):
    return get_policy(policy, state).sample().item()


def get_episode(env: gym.Env, policy: nn.Module):
    obs, acts, rews = [], [], []
    state, _ = env.reset()
    device = next(policy.parameters()).device

    while True:
        action = get_action(torch.as_tensor(state, dtype=torch.float32, device=device), policy)
        ns, r, terminated, truncated, _ = env.step(action)
        obs.append(state)
        acts.append(action)
        rews.append(r)
        if terminated or truncated:
            break
        state = ns

    # reward-to-go, computed backward then reversed into chronological order
    weights = []
    G = 0
    for r in reversed(rews):
        G += r
        weights.append(G)
    weights.reverse()

    return len(obs), sum(rews), obs, acts, weights


def compute_policy_loss(model, obs, acts, advantages):
    dist = get_policy(model, obs)
    return -(dist.log_prob(acts) * advantages).mean()


def compute_value_loss(value_net, obs, returns):
    values = value_net(obs)
    return ((values - returns) ** 2).mean()


def evaluate(model_path, num_episodes=10, render=False, greedy=False, device="cpu"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path} (run with --train first)")

    env = gym.make(ENV, render_mode="human" if render else None)
    nobs = env.observation_space.shape[0]
    nact = env.action_space.n

    model = Policy(nobs, nact)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    returns = []
    with torch.no_grad():
        for ep in range(num_episodes):
            state, _ = env.reset()
            ep_return = 0
            while True:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                dist = get_policy(model, state_t)
                action = dist.probs.argmax().item() if greedy else dist.sample().item()
                state, r, terminated, truncated, _ = env.step(action)
                ep_return += r
                if terminated or truncated:
                    break
            returns.append(ep_return)
            print(f"Episode {ep + 1}: return = {ep_return}")

    env.close()
    returns = np.array(returns)
    print(f"\nEvaluated {num_episodes} episodes")
    print(f"Mean return: {returns.mean():.2f} +/- {returns.std():.2f}")
    print(f"Min/Max: {returns.min():.0f} / {returns.max():.0f}")
    return returns


def train(policy_ckpt: Path, value_ckpt: Path, device: str):
    env = gym.make(ENV)
    nobs = env.observation_space.shape[0]
    nact = env.action_space.n

    model = Policy(nobs, nact).to(device)
    value_net = ValueBaseline(nobs).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters())
    value_optimizer = torch.optim.AdamW(params=value_net.parameters())

    for i in range(NUM_EPOCHS):
        batch_obs, batch_acts, batch_weights = [], [], []
        batch_rets, batch_lens = [], []

        while True:
            ep_len, ep_ret, obs, acts, weights = get_episode(env, model)
            batch_obs.extend(obs)
            batch_acts.extend(acts)
            batch_weights.extend(weights)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            if len(batch_obs) >= BATCH_SIZE:
                break

        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        acts_tensor = torch.as_tensor(batch_acts, dtype=torch.long, device=device)
        returns_tensor = torch.as_tensor(batch_weights, dtype=torch.float32, device=device)

        # --- baseline + advantage (no grad through value_net here) ---
        with torch.no_grad():
            baseline = value_net(obs_tensor)
        advantages = returns_tensor - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- policy update ---
        policy_loss = compute_policy_loss(model, obs_tensor, acts_tensor, advantages)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # --- value network update: regress toward raw returns, not advantages ---
        value_loss = compute_value_loss(value_net, obs_tensor, returns_tensor)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        if i % 20 == 0:
            print(f"Epoch {i}: mean_ret={np.mean(batch_rets):.1f}, "
                  f"mean_len={np.mean(batch_lens):.1f}, episodes={len(batch_rets)}, "
                  f"policy_loss={policy_loss.item():.3f}, value_loss={value_loss.item():.3f}")

    policy_ckpt.parent.mkdir(parents=True, exist_ok=True)
    value_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), policy_ckpt)
    torch.save(value_net.state_dict(), value_ckpt)
    print(f"Saved policy to {policy_ckpt} and value net to {value_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="VPG with learned value baseline on Acrobot-v1.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train a new policy + baseline and save them.")
    mode.add_argument("--eval", action="store_true", help="Evaluate a saved policy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=POLICY_CHECKPOINT, help="Policy checkpoint path.")
    parser.add_argument("--value-checkpoint", type=Path, default=VALUE_CHECKPOINT,
                        help="Value-net checkpoint path (written during --train).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes.")
    parser.add_argument("--render", action="store_true", help="Render eval episodes.")
    parser.add_argument("--greedy", action="store_true",
                        help="Use argmax action instead of sampling during eval.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.eval:
        evaluate(args.checkpoint, num_episodes=args.episodes, render=args.render,
                 greedy=args.greedy, device=device)
    else:
        train(args.checkpoint, args.value_checkpoint, device)


if __name__ == "__main__":
    main()
