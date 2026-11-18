"""
Proximal Policy Optimization (PPO, clip variant) for continuous control.
Usage:
    python PG/ppo.py --train
    python PG/ppo.py --eval [--episodes 10] [--render] [--greedy]
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import argparse
from pathlib import Path

NUM_EPOCHS = 1200
BATCH_SIZE = 2000        # target *timesteps* per rollout
ENV = "Pendulum-v1"
GAMMA = 0.98
LAM = 0.95               # GAE lambda
CLIP_EPS = 0.2
POLICY_LR = 3e-4
VALUE_LR = 1e-3
ENTROPY_COEF = 0.001
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 10          # gradient passes over each rollout
MINIBATCH_SIZE = 256
TARGET_KL = 0.02         # early-stop a PPO_EPOCHS pass if KL exceeds this
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
ACTOR_CHECKPOINT = CKPT_DIR / "ppo_actor.pt"
CRITIC_CHECKPOINT = CKPT_DIR / "ppo_critic.pt"


class Actor(nn.Module):
    def __init__(self, nobs, nact):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nobs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, nact),
        )
        self.log_stds = nn.Parameter(torch.ones(nact) * -0.5)

    def forward(self, state):
        means = self.net(state)
        stds = torch.exp(self.log_stds).expand_as(means)
        return means, stds


class Critic(nn.Module):
    def __init__(self, nobs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nobs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def get_policy(model, obs):
    means, stds = model(obs)
    return torch.distributions.Normal(means, stds)


def get_action(state, policy, action_low, action_high):
    """Sample an action and return (clipped_action, log_prob_of_unclipped_action, value)."""
    dist = get_policy(policy, state)
    raw_action = dist.sample()
    log_prob = dist.log_prob(raw_action).sum(-1)
    action = torch.clamp(raw_action, action_low, action_high)
    return action, raw_action, log_prob


def compute_gae(rews, values, final_value, terminated_flag, gamma, lam):
    """
    rews, values: python lists / 1D arrays over one complete episode.
    final_value: critic's estimate of the state *after* the last recorded
                 step (only used if the episode ended by truncation).
    terminated_flag: True if the episode ended by genuine termination
                      (no future value), False if it ended by truncation
                      (bootstrap with final_value).
    """
    l = len(rews)
    advs = np.zeros(l, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(l)):
        if t == l - 1:
            next_value = 0.0 if terminated_flag else final_value
        else:
            next_value = values[t + 1]
        delta = rews[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advs[t] = last_gae
    returns = advs + np.array(values, dtype=np.float32)
    return advs, returns


def get_episode(env, policy, value_net, device, action_low, action_high):
    obs, acts, rews, logprobs, values = [], [], [], [], []
    state, _ = env.reset()
    terminated_flag = False

    while True:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_t, raw_action_t, logprob_t = get_action(state_t, policy, action_low, action_high)
            value_t = value_net(state_t.unsqueeze(0)).item()

        action_np = action_t.cpu().numpy()
        ns, r, terminated, truncated, _ = env.step(action_np)

        obs.append(state)
        acts.append(raw_action_t.cpu().numpy())  # store the *unclipped* sampled action
        rews.append(r)
        logprobs.append(logprob_t.item())
        values.append(value_t)

        state = ns
        if terminated or truncated:
            terminated_flag = terminated
            final_state = ns
            break

    with torch.no_grad():
        final_state_t = torch.as_tensor(final_state, dtype=torch.float32, device=device)
        final_value = value_net(final_state_t.unsqueeze(0)).item()

    advs, returns = compute_gae(rews, values, final_value, terminated_flag, GAMMA, LAM)

    return len(obs), sum(rews), obs, acts, logprobs, advs, returns


def evaluate(model_path, num_episodes=10, render=False, greedy=False, device="cpu"):
    env = gym.make(ENV, render_mode="human" if render else None)
    nobs = env.observation_space.shape[0]
    nact = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

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
                action = torch.clamp(action, action_low, action_high)
                action_np = action.cpu().numpy()
                state, r, terminated, truncated, _ = env.step(action_np)
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


def train(actor_checkpoint: Path, critic_checkpoint: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(ENV)
    nobs = env.observation_space.shape[0]
    nact = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    model = Actor(nobs, nact).to(device)
    value_net = Critic(nobs).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=POLICY_LR)
    value_optimizer = torch.optim.AdamW(params=value_net.parameters(), lr=VALUE_LR)

    for i in range(NUM_EPOCHS):
        batch_obs, batch_acts, batch_logprobs = [], [], []
        batch_advs, batch_returns = [], []
        batch_rets, batch_lens = [], []

        while True:
            ep_len, ep_ret, obs, acts, logprobs, advs, returns = get_episode(
                env, model, value_net, device, action_low, action_high
            )
            batch_obs.extend(obs)
            batch_acts.extend(acts)
            batch_logprobs.extend(logprobs)
            batch_advs.extend(advs)
            batch_returns.extend(returns)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            if len(batch_obs) >= BATCH_SIZE:
                break

        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        acts_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.float32, device=device)
        old_logprobs_tensor = torch.as_tensor(batch_logprobs, dtype=torch.float32, device=device)
        advs_tensor = torch.as_tensor(np.array(batch_advs), dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(np.array(batch_returns), dtype=torch.float32, device=device)

        with torch.no_grad():
            old_values_tensor = value_net(obs_tensor)

        # normalize advantages once per rollout (not per minibatch)
        advs_tensor = (advs_tensor - advs_tensor.mean()) / (advs_tensor.std() + 1e-8)

        n = obs_tensor.shape[0]
        approx_kl = 0.0
        for ppo_epoch in range(PPO_EPOCHS):
            idx = torch.randperm(n, device=device)
            kls = []
            for start in range(0, n, MINIBATCH_SIZE):
                mb_idx = idx[start:start + MINIBATCH_SIZE]

                mb_obs = obs_tensor[mb_idx]
                mb_acts = acts_tensor[mb_idx]
                mb_old_logprobs = old_logprobs_tensor[mb_idx]
                mb_advs = advs_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]
                mb_old_values = old_values_tensor[mb_idx]

                dist = get_policy(model, mb_obs)
                new_logprobs = dist.log_prob(mb_acts).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean().item())

                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                new_values = value_net(mb_obs)
                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -CLIP_EPS, CLIP_EPS
                )
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), MAX_GRAD_NORM)
                value_optimizer.step()

            approx_kl = np.mean(kls)
            if approx_kl > TARGET_KL:
                # policy has moved far enough from the data-collection policy;
                # stop taking more gradient epochs on this rollout
                break

        if i % 10 == 0 or i == NUM_EPOCHS - 1:
            print(f"Epoch {i}: mean_ret={np.mean(batch_rets):.1f}, "
                  f"mean_len={np.mean(batch_lens):.1f}, episodes={len(batch_rets)}, "
                  f"policy_loss={policy_loss.item():.3f}, value_loss={value_loss.item():.3f}, "
                  f"entropy={entropy.item():.3f}, approx_kl={approx_kl:.4f}, "
                  f"ppo_epochs_used={ppo_epoch + 1}, log_std={model.log_stds.data.cpu().numpy()}")

    actor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    critic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), actor_checkpoint)
    torch.save(value_net.state_dict(), critic_checkpoint)
    print(f"Saved actor to {actor_checkpoint} and critic to {critic_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="PPO (clip) with GAE on Pendulum-v1.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train a new actor + critic and save them.")
    mode.add_argument("--eval", action="store_true", help="Evaluate a saved actor checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=ACTOR_CHECKPOINT, help="Actor checkpoint path.")
    parser.add_argument("--critic-checkpoint", type=Path, default=CRITIC_CHECKPOINT,
                        help="Critic checkpoint path (written during --train).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Render eval episodes")
    parser.add_argument("--greedy", action="store_true", help="Use the distribution mean instead of sampling during eval")
    args = parser.parse_args()

    if args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluate(
            args.checkpoint,
            num_episodes=args.episodes,
            render=args.render,
            greedy=args.greedy,
            device=device,
        )
    else:
        train(args.checkpoint, args.critic_checkpoint)


if __name__ == "__main__":
    main()
