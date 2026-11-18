"""
Usage:
    python PG/a2c.py --train
    python PG/a2c.py --eval [--episodes 10] [--render] [--greedy]
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

NUM_EPOCHS = 1200
BATCH_SIZE = 2000  # target *timesteps* per batch, not episode count
ENV = "Pendulum-v1"
NUM_STEPS = 5      # n in the n-step return
GAMMA = 0.98
ACTOR_LR = 3e-4
VALUE_TRAIN_ITERS = 10
VALUE_MINIBATCH = 256
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
ACTOR_CHECKPOINT = CKPT_DIR / "a2c_actor_pendulum.pt"
CRITIC_CHECKPOINT = CKPT_DIR / "a2c_critic_pendulum.pt"


class Actor(nn.Module):
    def __init__(self, nobs, nact):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nobs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, nact),
        )
        # sampled as gaussian for continuous actions
        self.log_stds = nn.Parameter(torch.zeros(nact))

    def forward(self, state):
        means = self.net(state)
        stds = torch.exp(self.log_stds).expand_as(means)
        return means, stds


class Critic(nn.Module):
    def __init__(self, nobs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nobs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        # squeeze last dim: (batch, 1) -> (batch,) so it matches returns/advantages shape
        return self.net(state).squeeze(-1)


def get_policy(model, obs):
    means, stds = model(obs)
    return torch.distributions.Normal(means, stds)


def get_action(state, policy):
    return float(get_policy(policy, state).sample().item())


def get_episode(env: gym.Env, policy: nn.Module, value_net: nn.Module, device: str):
    obs, acts, rews = [], [], []
    state, _ = env.reset()

    while True:
        action = get_action(torch.as_tensor(state, dtype=torch.float32, device=device), policy)
        # Pendulum clips the torque to its action bounds internally
        ns, r, terminated, truncated, _ = env.step(np.atleast_1d(action))
        obs.append(state)
        acts.append(action)
        rews.append(r)
        if terminated or truncated:
            break
        state = ns

    obs_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device)
    with torch.no_grad():
        values = value_net(obs_tensor).cpu().numpy()

    l = len(obs)
    weights = []
    for i in range(l):
        # 1. how many reward steps actually remain in the episode
        n_step = min(NUM_STEPS, l - i)
        # 2. accumulate discounted rewards for up to NUM_STEPS
        discounted_r = sum((GAMMA ** k) * rews[i + k] for k in range(n_step))
        # 3. bootstrap from V(s_{i+n}) only if that state lies within the episode
        if i + NUM_STEPS < l:
            discounted_r += (GAMMA ** NUM_STEPS) * values[i + NUM_STEPS]
        weights.append(discounted_r)

    return l, sum(rews), obs, acts, weights


def compute_policy_loss(model, obs, acts, advantages):
    dist = get_policy(model, obs)
    log_probs = dist.log_prob(acts).sum(-1)  # (batch,)
    return -(log_probs * advantages).mean()


def compute_value_loss(value_net, obs, returns):
    values = value_net(obs)
    return ((values - returns) ** 2).mean()


def evaluate(model_path, num_episodes=10, render=False, greedy=False, device="cpu"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path} (run with --train first)")

    env = gym.make(ENV, render_mode="human" if render else None)
    nobs = env.observation_space.shape[0]
    nact = env.action_space.shape[0]

    model = Actor(nobs, nact)
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
                action = dist.mean if greedy else dist.sample()
                state, r, terminated, truncated, _ = env.step(action.cpu().numpy())
                ep_return += r
                if terminated or truncated:
                    break
            returns.append(ep_return)
            print(f"Episode {ep + 1}: return = {ep_return:.1f}")

    env.close()
    returns = np.array(returns)
    print(f"\nEvaluated {num_episodes} episodes")
    print(f"Mean return: {returns.mean():.2f} +/- {returns.std():.2f}")
    print(f"Min/Max: {returns.min():.0f} / {returns.max():.0f}")
    return returns


def train(actor_ckpt: Path, critic_ckpt: Path, device: str):
    env = gym.make(ENV)
    nobs = env.observation_space.shape[0]
    nact = env.action_space.shape[0]

    model = Actor(nobs, nact).to(device)
    value_net = Critic(nobs).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=ACTOR_LR)
    value_optimizer = torch.optim.AdamW(params=value_net.parameters())

    for i in range(NUM_EPOCHS):
        batch_obs, batch_acts, batch_weights = [], [], []
        batch_rets, batch_lens = [], []

        while True:
            ep_len, ep_ret, obs, acts, weights = get_episode(env, model, value_net, device)
            batch_obs.extend(obs)
            batch_acts.extend(acts)
            batch_weights.extend(weights)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            if len(batch_obs) >= BATCH_SIZE:
                break

        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        acts_tensor = torch.as_tensor(batch_acts, dtype=torch.float32, device=device).unsqueeze(-1)
        returns_tensor = torch.as_tensor(np.array(batch_weights), dtype=torch.float32, device=device)

        # --- critic baseline + advantage (no grad through value_net here) ---
        with torch.no_grad():
            baseline = value_net(obs_tensor)
        advantages = returns_tensor - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- actor update ---
        policy_loss = compute_policy_loss(model, obs_tensor, acts_tensor, advantages)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # --- critic update: several minibatch epochs toward n-step returns ---
        value_dataset = torch.utils.data.TensorDataset(obs_tensor, returns_tensor)
        loader = torch.utils.data.DataLoader(value_dataset, batch_size=VALUE_MINIBATCH, shuffle=True)
        for _ in range(VALUE_TRAIN_ITERS):
            for ob_mb, ret_mb in loader:
                v_loss = compute_value_loss(value_net, ob_mb, ret_mb)
                value_optimizer.zero_grad()
                v_loss.backward()
                value_optimizer.step()

        if i % 20 == 0:
            with torch.no_grad():
                value_loss = compute_value_loss(value_net, obs_tensor, returns_tensor)
            print(f"Epoch {i}: mean_ret={np.mean(batch_rets):.1f}, "
                  f"mean_len={np.mean(batch_lens):.1f}, episodes={len(batch_rets)}, "
                  f"policy_loss={policy_loss.item():.3f}, value_loss={value_loss.item():.3f}")

    actor_ckpt.parent.mkdir(parents=True, exist_ok=True)
    critic_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), actor_ckpt)
    torch.save(value_net.state_dict(), critic_ckpt)
    print(f"Saved actor to {actor_ckpt} and critic to {critic_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Batch A2C with n-step returns on Pendulum-v1.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train a new actor + critic and save them.")
    mode.add_argument("--eval", action="store_true", help="Evaluate a saved actor checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=ACTOR_CHECKPOINT, help="Actor checkpoint path.")
    parser.add_argument("--critic-checkpoint", type=Path, default=CRITIC_CHECKPOINT,
                        help="Critic checkpoint path (written during --train).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes.")
    parser.add_argument("--render", action="store_true", help="Render eval episodes.")
    parser.add_argument("--greedy", action="store_true",
                        help="Act with the Gaussian mean instead of sampling during eval.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.eval:
        evaluate(args.checkpoint, num_episodes=args.episodes, render=args.render,
                 greedy=args.greedy, device=device)
    else:
        train(args.checkpoint, args.critic_checkpoint, device)


if __name__ == "__main__":
    main()
