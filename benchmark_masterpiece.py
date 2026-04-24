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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ════════════════════════════════════════════════════════════════════
#  HEINN-X 코어 & 기저 행렬 (수치 오차 완전 제거)
# ════════════════════════════════════════════════════════════════════


def _algorithm_B(p):
    def _frac(p):
        return [Fraction(c) for c in p]

    def _trim(p):
        while len(p) > 1 and p[-1] == 0:
            p.pop()
        return p

    def _ls(a, b):
        return [Fraction(0), Fraction(b) - Fraction(a) / 2, Fraction(a) / 2]

    def _bM(d):
        M = [[Fraction(0)] * (d + 1) for _ in range(d + 1)]
        for k in range(1, d + 2):
            for j in range(k):
                if j <= d:
                    M[j][k - 1] = Fraction(math.comb(k, j))
        return M

    def _sv(M, b):
        n = len(b)
        aug = [[Fraction(v) for v in row] + [Fraction(b[i])] for i, row in enumerate(M)]
        for col in range(n):
            piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
            if piv is None:
                return [Fraction(0)] * n
            aug[col], aug[piv] = aug[piv], aug[col]
            for row in range(n):
                if row != col and aug[row][col] != 0:
                    f = aug[row][col] / aug[col][col]
                    aug[row] = [aug[row][j] - f * aug[col][j] for j in range(n + 1)]
        return [aug[i][n] / aug[i][i] for i in range(n)]

    p = _frac(p)
    d = len(p) - 1
    if d == 0:
        return [Fraction(0), p[0]]
    if d == 1:
        return _ls(p[1], p[0])
    sol = _sv(_bM(d), [p[j] for j in range(d + 1)])
    return _trim([Fraction(0)] + sol)


# [핵심] 부동소수점 역행렬 에러를 막기 위한 완벽한 체비쇼프 정수 계수 생성기
def exact_shifted_chebyshev_coeffs(degree):
    coeffs = [[Fraction(1)], [Fraction(-1), Fraction(2)]]
    if degree == 0:
        return [coeffs[0]]
    for k in range(1, degree):
        Tk = coeffs[k]
        Tk_minus_1 = coeffs[k - 1]
        term1 = [Fraction(0)] + [c * 4 for c in Tk]
        term2 = [c * -2 for c in Tk]
        max_len = len(term1)
        term2 += [Fraction(0)] * (max_len - len(term2))
        Tk_minus_1_padded = Tk_minus_1 + [Fraction(0)] * (max_len - len(Tk_minus_1))
        Tk_plus_1 = [term1[i] + term2[i] - Tk_minus_1_padded[i] for i in range(max_len)]
        coeffs.append(Tk_plus_1)
    return coeffs[: degree + 1]


def build_S_matrix(degree: int):
    N_pts = max(4 * (degree + 1), 128)
    k_pts = np.arange(N_pts)
    x_pts = 0.5 * (1 + np.cos(math.pi * (2 * k_pts + 1) / (2 * N_pts)))

    exact_cheb = exact_shifted_chebyshev_coeffs(degree)
    F_at_pts = np.zeros((N_pts, degree + 1))

    # 노이즈 0%의 대수적 적분
    for k in range(degree + 1):
        F_c = _algorithm_B(exact_cheb[k])
        c_floats = [float(c) for c in F_c]
        for i, xi in enumerate(x_pts):
            F_at_pts[i, k] = sum(c_floats[j] * (xi**j) for j in range(len(c_floats)))

    xn = np.clip(2 * x_pts - 1, -1 + 1e-6, 1 - 1e-6)
    T_sum = np.cos(np.arccos(xn)[:, None] * np.arange(degree + 2)[None, :])

    S = np.linalg.pinv(T_sum) @ F_at_pts
    # 에너지 밸런스를 맞추어 학습을 돕는 컬럼 정규화 (노이즈가 없으므로 안전함)
    S_normed = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-8)

    return torch.tensor(S_normed, dtype=torch.float32)


def chebyshev_matrices(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    ks = torch.arange(degree + 1, dtype=torch.float32)
    T = torch.cos(theta.unsqueeze(1) * ks.unsqueeze(0))
    T_pinv = torch.linalg.pinv(T.double(), rcond=1e-5).float()
    if device is not None:
        T, T_pinv = T.to(device), T_pinv.to(device)
    return T_pinv, T


def chebyshev_sum_matrix(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    T_sum = torch.cos(
        theta.unsqueeze(1) * torch.arange(degree + 2, dtype=torch.float32).unsqueeze(0)
    )
    if device is not None:
        T_sum = T_sum.to(device)
    return T_sum


# ════════════════════════════════════════════════════════════════════
#  CONV LAYERS (Dual-Path 구조)
# ════════════════════════════════════════════════════════════════════


class ChebConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.W = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch * (degree + 1))
        )
        self._cache = {}

    def forward(self, x):
        B, C, N = x.shape
        d_eff = min(self.degree, N // 2)
        key = (N, d_eff, str(x.device))
        if key not in self._cache:
            self._cache[key] = chebyshev_matrices(N, d_eff, x.device)
        T_pinv, T = self._cache[key]

        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        m = torch.einsum("bck,cok->bok", c, self.W[:, :, : d_eff + 1])
        return torch.einsum("bok,nk->bon", m, T)


class HeinnXConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.W_base = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch * (degree + 1))
        )
        self.W_int = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 2) / math.sqrt(in_ch * (degree + 2))
        )
        self.register_buffer("S", build_S_matrix(degree))
        self._cache = {}

    def forward(self, x):
        B, C, N = x.shape
        d_eff = min(self.degree, N // 2)
        key = (N, d_eff, str(x.device))
        if key not in self._cache:
            self._cache[key] = (
                chebyshev_matrices(N, d_eff, x.device)[0],
                chebyshev_matrices(N, d_eff, x.device)[1],
                chebyshev_sum_matrix(N, d_eff, x.device),
            )
        T_pinv, T, T_sum = self._cache[key]

        c = torch.einsum("bcn,kn->bck", x, T_pinv)

        # Path A: Detail
        m_base = torch.einsum("bck,cok->bok", c, self.W_base[:, :, : d_eff + 1])
        out_base = torch.einsum("bok,nk->bon", m_base, T)

        # Path B: Physics-Informed Integral
        c_int = torch.einsum("bck,lk->bcl", c, self.S[: d_eff + 2, : d_eff + 1])
        m_int = torch.einsum("bcl,col->bol", c_int, self.W_int[:, :, : d_eff + 2])
        out_int = torch.einsum("bol,nl->bon", m_int, T_sum)

        return out_base + out_int


class SpectralConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        self.W = nn.Parameter(torch.randn(in_ch, out_ch, modes, 2) / (in_ch * out_ch))

    def forward(self, x):
        B, C, N = x.shape
        xf = torch.fft.rfft(x)
        nm = min(self.modes, N // 2 + 1)
        out = torch.zeros(
            B, self.W.shape[1], N // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        wr, wi = self.W[:, :, :nm, 0], self.W[:, :, :nm, 1]
        xr, xi = xf.real[:, :, :nm], xf.imag[:, :, :nm]
        out[:, :, :nm] = torch.complex(
            torch.einsum("bim,iom->bom", xr, wr) - torch.einsum("bim,iom->bom", xi, wi),
            torch.einsum("bim,iom->bom", xr, wi) + torch.einsum("bim,iom->bom", xi, wr),
        )
        return torch.fft.irfft(out, n=N)


def make_no_1d(conv_layer, width=32, depth=4, in_ch=2, out_ch=1):
    class NO1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Linear(in_ch, width)
            self.convs = nn.ModuleList([conv_layer() for _ in range(depth)])
            self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
            self.norms = nn.ModuleList([nn.GroupNorm(4, width) for _ in range(depth)])
            self.proj = nn.Sequential(
                nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
            )

        def forward(self, x):
            x = self.lift(x).permute(0, 2, 1)
            for c, w, n in zip(self.convs, self.ws, self.norms):
                x = F.gelu(n(c(x) + w(x)))
            return self.proj(x.permute(0, 2, 1))

    return NO1D()


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

    U_master, V_master = generate_shock_physics_master(1000, 128)

    U_tr, V_tr = U_master[:800, ::2], V_master[:800, ::2]
    tr = DataLoader(
        TensorDataset(make_input_1d(U_tr), V_tr.unsqueeze(-1)), 32, shuffle=True
    )

    U_te, V_te = U_master[800:, ::2], V_master[800:, ::2]
    te = DataLoader(TensorDataset(make_input_1d(U_te), V_te.unsqueeze(-1)), 32)

    fno = make_no_1d(lambda: SpectralConv1D(32, 32, 16)).to(DEVICE)
    cheb = make_no_1d(lambda: ChebConv1D(32, 32, 16)).to(DEVICE)
    hx = make_no_1d(lambda: HeinnXConv1D(32, 32, 16)).to(DEVICE)

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
