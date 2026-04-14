from __future__ import annotations
import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ===================== POLICY NETWORK =====================
class PolicyNet(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


# ===================== ENV IMPORT =====================
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=1000)

    # ✅ REQUIRED ENV PARAMS (FIXED ERROR)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--wall_obstacles", action="store_true")

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    for ep in range(args.episodes):

        # ✅ FIXED ENV CREATION
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        s = env.reset(seed=args.seed + ep)

        log_probs = []
        rewards = []

        for _ in range(args.max_steps):

            s_t = torch.tensor(s, dtype=torch.float32)

            probs = policy(s_t)
            dist = torch.distributions.Categorical(probs)

            a = dist.sample()
            log_prob = dist.log_prob(a)

            s2, r, done = env.step(ACTIONS[a.item()], render=False)

            log_probs.append(log_prob)
            rewards.append(r)

            s = s2

            if done:
                break

        # ===================== RETURNS =====================
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # normalize (important)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ===================== LOSS =====================
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================== PRINT =====================
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} reward={sum(rewards):.1f}")

        # ===================== SAVE =====================
        if (ep + 1) % 100 == 0:
            save_path = f"weights_ep{ep+1}.pth"
            torch.save(policy.state_dict(), save_path)
            print("Saved checkpoint:", save_path)

    torch.save(policy.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()