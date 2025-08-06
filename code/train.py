"""
train.py
--------
Full primal-dual + follower-supervision training loop.

Example
-------
python train.py --buffer C:/backupcgi/final_bak/processed/replay_buffer.npz \
                --epochs 10 --batch 512 --horizon 12 --logdir runs/demo

TensorBoard
-----------
tensorboard --logdir runs/demo
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from replay_buffer import ReplayBuffer
from models import Actor, CriticLambda, CriticY

# ------------------------------------------------------------------- #
# misc
# ------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

VAL_FRAC = 0.10     # % trajectories for validation
RHO      = 1.0      # augmented-Lagrange weight
BC_W     = 1e-3     # behaviour-cloning weight (light)

# ------------------------------------------------------------------- #
def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--buffer",  required=True)
    p.add_argument("--epochs",  type=int,   default=30)
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--horizon", type=int,   default=12)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--logdir",  type=str,   default="runs/demo")
    return p


# ------------------------------------------------------------------- #
def main(cfg):
    rb = ReplayBuffer.load(cfg.buffer)

    N = len(rb)
    val_idx = np.random.choice(N, int(N * VAL_FRAC), replace=False)
    train_idx = np.setdiff1d(np.arange(N), val_idx)

    actor  = Actor().to(DEVICE)
    cl     = CriticLambda().to(DEVICE)
    cy     = CriticY().to(DEVICE)

    optA = torch.optim.AdamW(actor.parameters(),  lr=cfg.lr)
    optL = torch.optim.AdamW(cl.parameters(),     lr=cfg.lr * 10)
    optY = torch.optim.AdamW(cy.parameters(),     lr=cfg.lr)

    schedA = torch.optim.lr_scheduler.OneCycleLR(
        optA,
        max_lr=cfg.lr,
        total_steps=cfg.epochs * (len(train_idx) // cfg.batch + 1),
    )

    writer = SummaryWriter(cfg.logdir)
    global_step = 0
    best_val = float("inf"); patience = 5; wait = 0

    for epoch in range(cfg.epochs):
        np.random.shuffle(train_idx)

        actor.train(); cl.train(); cy.train()

        for b in range(0, len(train_idx), cfg.batch):
            # -------- sample mini-batch --------
            batch_ids = train_idx[b:b + cfg.batch]
            S_np, A_np, R_np, G_np, M_np = rb.sample(len(batch_ids), cfg.horizon)

            S      = torch.tensor(S_np, device=DEVICE)
            A_exec = torch.tensor(A_np, device=DEVICE)               # already in (-1,1) scale
            R      = torch.tensor(R_np, device=DEVICE)
            G      = torch.tensor(G_np, device=DEVICE)
            M      = torch.tensor(M_np, device=DEVICE)

            g_pos = G.clamp(min=0)

            # -------------------------------------------------------
            # 1) Critic-λ update (dual ascent, actor frozen)
            # -------------------------------------------------------
            with torch.no_grad():
                A_det = actor(S)                            # detach graph
            lam_pred = cl(S, A_det)
            lossL = -(lam_pred * g_pos).mean() - RHO * (g_pos ** 2).mean()

            optL.zero_grad()
            lossL.backward()
            optL.step()

            # -------------------------------------------------------
            # 2) Actor update (λ treated as constant)
            # -------------------------------------------------------
            A_pred = actor(S)
            with torch.no_grad():
                lam_const = cl(S, A_pred).clamp(0.0, 5.0)

            loss_rl = (lam_const * G).mean() + RHO * (g_pos ** 2).mean()
            loss_bc = BC_W * nn.functional.mse_loss(A_pred, A_exec)
            lossA   = -(R * M).mean() + loss_rl + loss_bc

            optA.zero_grad()
            lossA.backward()
            optA.step()
            schedA.step()

            # -------------------------------------------------------
            # 3) Critic-y update (supervised compliance)
            # -------------------------------------------------------
            with torch.no_grad():
                A_det = actor(S)
            q_pred = cy(S, A_det)
            lossY  = nn.functional.mse_loss(q_pred * M.unsqueeze(-1),
                                            A_exec * M.unsqueeze(-1))

            optY.zero_grad()
            lossY.backward()
            optY.step()

            # ---------------- logging ----------------
            if global_step % 100 == 0:
                writer.add_scalar("train/loss_actor",  lossA.item(),  global_step)
                writer.add_scalar("train/loss_lambda", lossL.item(),  global_step)
                writer.add_scalar("train/loss_y",      lossY.item(),  global_step)
            global_step += 1

        # ---------------- validation ----------------
        actor.eval(); cl.eval()
        with torch.no_grad():
            S_np, A_np, R_np, G_np, _ = rb.sample(len(val_idx), cfg.horizon)
            S = torch.tensor(S_np, device=DEVICE)
            A_pred = actor(S)
            lam_pred = cl(S, A_pred)
            val_metric = (lam_pred * torch.tensor(G_np, device=DEVICE)).abs().mean().item()
            writer.add_scalar("val/|λ·g|", val_metric, epoch)

        print(f"Epoch {epoch:02d}  |  val |λ·g| = {val_metric:.4f}")

        if val_metric < best_val:
            best_val = val_metric; wait = 0
            torch.save(actor.state_dict(),      Path(cfg.logdir) / "actor.pt")
            torch.save(cl.state_dict(),         Path(cfg.logdir) / "critic_lambda.pt")
            torch.save(cy.state_dict(),         Path(cfg.logdir) / "critic_y.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early-stop triggered.")
                break


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    args = make_parser().parse_args()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(Path(args.logdir) / "cfg.json", "w"), indent=2)
    main(args)