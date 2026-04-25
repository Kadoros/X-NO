"""
benchmark_masterpiece.py — Heinn-X vs FNO: The Anti-Fourier Benchmark
(Vandermonde Float64 폭발 버그 완벽 수정본)
"""

import math, random
from fractions import Fraction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from XNO import HeinnXConv1D, ChebConv1D, SpectralConv1D, make_no_1d

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Device: {DEVICE}\n")


# ════════════════════════════════════════════════════════════════════
#  DATA GENERATION: FNO의 악몽 "비주기적 충격파 시스템"
# ════════════════════════════════════════════════════════════════════


def generate_shock_physics_master(n_samples, master_res=128):
    x = torch.linspace(-1, 1, master_res)
    us, vs = [], []
    for _ in range(n_samples):
        c1, c2, c3 = torch.randn(3)
        u = c1 * x + c2 * (x**2) + c3 * (x**3)
        shock_idx = random.randint(master_res // 4, master_res * 3 // 4)
        u[shock_idx:] += torch.randn(1).item() * 2.0

        v = torch.cumsum(u, dim=0) * (2.0 / master_res)
        us.append(u)
        vs.append(v)

    U, V = torch.stack(us), torch.stack(vs)
    return (U - U.mean()) / U.std(), (V - V.mean()) / V.std()


def make_input_1d(U):
    B, N = U.shape
    return torch.cat(
        [U.unsqueeze(-1), torch.linspace(0, 1, N).view(1, N, 1).expand(B, -1, -1)], -1
    )


def rel_l2(pred, tgt):
    return ((pred - tgt).norm() / (tgt.norm() + 1e-8)).item()


def train_epoch(model, loader, opt):
    model.train()
    tot = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
    return tot / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    return rel_l2(
        torch.cat([model(xb.to(DEVICE)).cpu() for xb, _ in loader]),
        torch.cat([yb for _, yb in loader]),
    )


def run_anti_fourier_benchmark():
    print("\n" + "=" * 65)
    print("  1D 비주기 충격파(Anti-Fourier) 벤치마크")
    print("  목적: FNO가 붕괴하는 현실 물리계에서 Heinn-X의 수학적 우위 증명")
    print("=" * 65)

    MODES = 8

    U_master, V_master = generate_shock_physics_master(1000, 128)

    U_tr, V_tr = U_master[:800, ::2], V_master[:800, ::2]
    tr = DataLoader(
        TensorDataset(make_input_1d(U_tr), V_tr.unsqueeze(-1)), 32, shuffle=True
    )

    U_te, V_te = U_master[800:, ::2], V_master[800:, ::2]
    te = DataLoader(TensorDataset(make_input_1d(U_te), V_te.unsqueeze(-1)), 32)

    fno = make_no_1d(lambda: SpectralConv1D(32, 32, MODES)).to(DEVICE)
    cheb = make_no_1d(lambda: ChebConv1D(32, 32, MODES)).to(DEVICE)
    hx = make_no_1d(lambda: HeinnXConv1D(32, 32, MODES)).to(DEVICE)

    results = {}
    for name, model in [("FNO", fno), ("ChebNO", cheb), ("Heinn-X", hx)]:
        print(f"\n── {name} 학습 ─────────────────────────────────")
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
        for ep in range(1, 51):
            loss = train_epoch(model, tr, opt)
            sched.step()
            if ep % 10 == 0:
                print(f"  Ep{ep:3d}  loss={loss:.4e}  rel-L2={evaluate(model, te):.4f}")

        results[name] = {}
        for res, step in [(64, 2), (32, 4), (16, 8)]:
            Ut, Vt = U_master[800:, ::step], V_master[800:, ::step]
            results[name][res] = evaluate(
                model,
                DataLoader(TensorDataset(make_input_1d(Ut), Vt.unsqueeze(-1)), 32),
            )
    return results


if __name__ == "__main__":
    r_anti = run_anti_fourier_benchmark()

    print("\n" + "=" * 65 + "\n  FINAL SUMMARY: The Fall of Fourier\n" + "=" * 65)
    print("\n  [1] 비주기 충격파 시스템 (Non-Periodic Shocks):")
    for res in [64, 32, 16]:
        ef, ec, eh = r_anti["FNO"][res], r_anti["ChebNO"][res], r_anti["Heinn-X"][res]
        tag = (
            "FNO"
            if min(ef, ec, eh) == ef
            else ("Cheb" if min(ef, ec, eh) == ec else "HX✓")
        )
        print(f"    res={res:3d}  FNO={ef:.4f}  Cheb={ec:.4f}  HX={eh:.4f}  [{tag}]")
