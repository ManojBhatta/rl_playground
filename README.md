# RL Playground

Implementations of reinforcement learning algorithms from Sutton & Barto's *Reinforcement Learning: An Introduction*. They start with tabular dynamic programming and go up to policy gradients, actor-critic and PPO. Each algorithm is a single, self-contained script.

| Stage | Algorithms | Environment |
|---|---|---|
| [DP/](DP/) | Policy evaluation, policy iteration, value iteration | Grid world (tabular, known model) |
| [MC&TD/](MC&TD/) | MC prediction, MC control, TD(0) prediction, SARSA | Grid world (tabular, model-free) |
| [VFA/](VFA/) | Semi-gradient SARSA, Q-learning, DQN | CartPole-v1, Atari Pong |
| [PG/](PG/) | REINFORCE (reward-to-go), REINFORCE with baseline, A2C, PPO | Acrobot-v1, Pendulum-v1 |

## Setup

The project uses [uv](https://github.com/astral-sh/uv) and Python 3.11+.

```bash
uv sync
```

Run every command below from the repository root with `uv run python …`, or activate `.venv` first.

## 1. Dynamic Programming — `DP/`

These are planning methods. They assume a full model of the grid world (ch. 4).

| Script | What it does |
|---|---|
| `policy_evaluation.py` | Iterative policy evaluation: computes V^π for a fixed policy with Bellman expectation backups |
| `policy_iteration.py` | Alternates policy evaluation and greedy policy improvement until the policy stops changing |
| `value_iteration.py` | Applies Bellman optimality backups until ‖ΔV‖ < θ, then extracts the greedy policy |

```bash
python DP/value_iteration.py
```

## 2. Monte Carlo & Temporal-Difference — `MC&TD/`

These are model-free tabular methods on the same grid world (ch. 5–6).

| Script | What it does |
|---|---|
| `mc_prediction.py` | Monte Carlo estimate of V^π from sampled episodes |
| `mc_control.py` | On-policy every-visit MC control with an ε-greedy policy |
| `td_prediction.py` | TD(0) prediction: `V(s) ← V(s) + α[r + γV(s') − V(s)]` |
| `td_control.py` | Tabular SARSA with annealed ε-greedy exploration |

```bash
python "MC&TD/td_control.py"
```

## 3. Value Function Approximation — `VFA/`

These scripts swap the table for a neural network (ch. 9–11) and end with DQN.

| Script | Algorithm | Env | Details |
|---|---|---|---|
| `sarsa.py` | Episodic semi-gradient SARSA | CartPole-v1 | Online, on-policy updates with target `r + γQ(s', a')`; 4-32-2 MLP; SGD |
| `qlearning.py` | Q-learning + experience replay | CartPole-v1 | Off-policy target `r + γ max Q(s', ·)`; replay buffer; linear ε-decay; 4-32-64-32-2 MLP; AdamW |
| `dqn.py` | DQN (Double-DQN target) | ALE/Pong-v5 | DeepMind Atari preprocessing (84×84 grayscale, 4-frame stack); conv Q-net; target network with hard sync; Huber loss; gradient clipping; TensorBoard logging |

```bash
python VFA/qlearning.py --eval --render      # watch the trained agent
python VFA/qlearning.py --train              # retrain from scratch
python VFA/dqn.py --train                    # logs to VFA/runs/my_dqn
tensorboard --logdir VFA/runs
```

Training these CartPole agents was finicky:

- Results vary a lot from run to run with the same hyperparameters.
- AdamW with lr=1e-4 worked better than SGD.
- A deeper MLP (4→32→64→32→2) worked better than a single hidden layer.
- The default +1 reward worked better than `sutton_barto_reward` for Q-learning.

That instability is what the replay buffer and target network in DQN are meant to fix.

## 4. Policy Gradient & Actor-Critic — `PG/`

These scripts optimise the policy directly (ch. 13), ending with PPO.

| Script | Algorithm | Env | Details |
|---|---|---|---|
| `vpg.py` | Vanilla policy gradient (REINFORCE) | Acrobot-v1 | Categorical policy; each log π(a\|s) weighted by the reward-to-go G_t; batches of ~4000 steps |
| `vpg_w_baseline.py` | REINFORCE with learned baseline | Acrobot-v1 | Value net V(s) fit to MC returns; advantage `G_t − V(s_t)`, normalised per batch |
| `a2c.py` | Advantage actor-critic (batch A2C) | Pendulum-v1 | Gaussian policy with learned log-std; critic trained on 5-step bootstrapped returns; several minibatch epochs on the critic per batch |
| `ppo.py` | PPO (clipped surrogate) | Pendulum-v1 | Builds on `a2c.py`: GAE(λ) advantages; 10 epochs of minibatch updates per rollout; clipped value loss; entropy bonus; approximate-KL early stopping; gradient-norm clipping |

```bash
python PG/vpg_w_baseline.py --eval --render
python PG/a2c.py --eval --greedy --render    # act with the Gaussian mean
python PG/a2c.py --train
python PG/ppo.py --eval --render
```

## Command-line interface

Every VFA and PG script takes the same flags:

| Flag | Meaning |
|---|---|
| `--train` / `--eval` | Required. Choose one: train and save a checkpoint, or load one and evaluate |
| `--checkpoint PATH` | Override the default checkpoint path |
| `--episodes N` | Number of evaluation episodes (default 10) |
| `--render` | Open a window and render evaluation episodes |
| `--greedy` | (PG only) Act deterministically instead of sampling from the policy |

Checkpoints live in `VFA/checkpoints/` and `PG/checkpoints/`. Paths are resolved relative to each script, so the scripts work from any directory.

## Pretrained checkpoints

These are 10-episode evaluations of the included checkpoints. Returns are stochastic, so expect some spread.

| Checkpoint | Script | Env | Mean return | Notes |
|---|---|---|---|---|
| `VFA/checkpoints/qlearning_cartpole.pth` | `qlearning.py` | CartPole-v1 | ~200 (max 500) | Several episodes reach the 500 cap |
| `VFA/checkpoints/sarsa_cartpole.pth` | `sarsa.py` | CartPole-v1 | ~28 | Weak; semi-gradient SARSA with one online update per step is unstable here |
| `PG/checkpoints/vpg_acrobot.pt` | `vpg.py` | Acrobot-v1 | ~−105 | Random policy ≈ −500 |
| `PG/checkpoints/vpg_baseline_policy_acrobot.pt` | `vpg_w_baseline.py` | Acrobot-v1 | ~−89 | Lower variance than plain VPG |
| `PG/checkpoints/a2c_actor_pendulum.pt` | `a2c.py` | Pendulum-v1 | ~−145 | Random policy ≈ −1200 |
| `PG/checkpoints/ppo_actor.pt` | `ppo.py` | Pendulum-v1 | ~−130 | High spread; individual episodes range from about −1 to −375 |
| — | `dqn.py` | Pong | — | No checkpoint yet; run `--train` |

The value-network and critic checkpoints (`vpg_baseline_value_acrobot.pt`, `a2c_critic_pendulum.pt`, `ppo_critic.pt`) are saved alongside the policies but are only needed for training.

## Acknowledgements

- Richard S. Sutton & Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
- The grid-world environment in `lib/smallgrid/` is adapted from the Intelligent Unmanned Systems Laboratory at Westlake University (Shiyu Zhao, *Mathematical Foundations of Reinforcement Learning*).
- Mnih et al., *Human-level control through deep reinforcement learning* (2015); van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* (2016).
- OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com/) (Joshua Achiam, 2018). The policy-gradient scripts follow its structure: reward-to-go, batches collected by timestep count, and the PPO-clip formulation.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2016); Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
