"""
benchmark_ultimate.py — The True Resolution Invariance Benchmark
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
#  HEINN-X 코어 & 기저 행렬
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


def chebyshev_matrices(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    ks = torch.arange(degree + 1, dtype=torch.float32)
    T = torch.cos(theta.unsqueeze(1) * ks.unsqueeze(0))
    T_pinv = torch.linalg.pinv(T.double(), rcond=1e-4).float()
    if device is not None:
        T, T_pinv = T.to(device), T_pinv.to(device)
    return T_pinv, T


def chebyshev_sum_matrix(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    ks = torch.arange(degree + 2, dtype=torch.float32)
    T_sum = torch.cos(theta.unsqueeze(1) * ks.unsqueeze(0))
    if device is not None:
        T_sum = T_sum.to(device)
    return T_sum


def build_S_matrix(degree: int):
    N_pts = max(4 * (degree + 1), 128)
    k_pts = np.arange(N_pts)
    x_pts = 0.5 * (1 + np.cos(math.pi * (2 * k_pts + 1) / (2 * N_pts)))
    xn = np.clip(2 * x_pts - 1, -1 + 1e-6, 1 - 1e-6)
    theta = np.arccos(xn)
    T_f = np.cos(theta[:, None] * np.arange(degree + 1)[None, :])
    T_sum = np.cos(theta[:, None] * np.arange(degree + 2)[None, :])

    V_mono = np.vstack([x_pts**j for j in range(degree + 1)]).T
    V_pinv = np.linalg.pinv(V_mono)
    T_k_mono = V_pinv @ T_f

    F_at_pts = np.zeros((N_pts, degree + 1))
    for k in range(degree + 1):
        mono_c = T_k_mono[:, k].tolist()
        F_c = [float(c) for c in _algorithm_B(mono_c)]
        for i, xi in enumerate(x_pts):
            F_at_pts[i, k] = sum(F_c[j] * xi**j for j in range(len(F_c)))

    S = np.linalg.pinv(T_sum) @ F_at_pts
    S_normed = S / (np.linalg.norm(S) + 1e-8)
    return torch.tensor(S_normed, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════
#  CONV LAYERS (ChebNO / Heinn-X)
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
        # 가중치는 오직 미분/변화량 공간(Derivative Space)에서만 작동함
        self.W = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch * (degree + 1))
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
                chebyshev_sum_matrix(N, d_eff, x.device),
            )
        T_pinv, T_sum = self._cache[key]

        # 1. 입력을 계수로 변환
        c = torch.einsum("bcn,kn->bck", x, T_pinv)

        # 2. 신경망(W)은 변화량(Derivative/Forcing term)만 예측함
        m_deriv = torch.einsum("bck,cok->bok", c, self.W[:, :, : d_eff + 1])

        # 3. [승리 요인: 적분 병목] Heinn-X 행렬이 강제로 최종 해를 적분해냄!
        # (신경망이 S를 우회할 수 없음)
        S_slice = self.S[: d_eff + 2, : d_eff + 1]
        m_int = torch.einsum("bok,lk->bol", m_deriv, S_slice)

        # 4. 차수가 1단계 높아진(d_eff+2) 다항식을 그리드로 복원
        return torch.einsum("bol,nl->bon", m_int, T_sum)


class ChebConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.out_ch = out_ch
        self.W_w = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch * (degree + 1))
        )
        self.W_h = nn.Parameter(
            torch.randn(out_ch, out_ch, degree + 1) / math.sqrt(out_ch * (degree + 1))
        )
        self._cache = {}

    def _1d(self, x, W, N):
        d_eff = min(self.degree, N // 2)
        key = (N, d_eff, str(x.device))
        if key not in self._cache:
            self._cache[key] = chebyshev_matrices(N, d_eff, x.device)
        T_pinv, T = self._cache[key]
        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        m = torch.einsum("bck,cok->bok", c, W[:, :, : d_eff + 1])
        return torch.einsum("bok,nk->bon", m, T)

    def forward(self, x):
        B, C, H, W = x.shape
        xw = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        xw = self._1d(xw, self.W_w, W).reshape(B, H, self.out_ch, W).permute(0, 2, 1, 3)
        xh = xw.permute(0, 3, 1, 2).reshape(B * W, self.out_ch, H)
        xh = self._1d(xh, self.W_h, H)
        return xh.reshape(B, W, self.out_ch, H).permute(0, 2, 3, 1)


class HeinnXConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.out_ch = out_ch
        self.W_w = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch * (degree + 1))
        )
        self.W_h = nn.Parameter(
            torch.randn(out_ch, out_ch, degree + 1) / math.sqrt(out_ch * (degree + 1))
        )

        self.register_buffer("S", build_S_matrix(degree))
        self._cache = {}

    def _1d(self, x, W, N):
        d_eff = min(self.degree, N // 2)
        key = (N, d_eff, str(x.device))
        if key not in self._cache:
            self._cache[key] = (
                chebyshev_matrices(N, d_eff, x.device)[0],
                chebyshev_sum_matrix(N, d_eff, x.device),
            )
        T_pinv, T_sum = self._cache[key]

        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        # 신경망 믹싱 (변화량 도출)
        m_deriv = torch.einsum("bck,cok->bok", c, W[:, :, : d_eff + 1])
        # Heinn-X 강제 적분
        S_slice = self.S[: d_eff + 2, : d_eff + 1]
        m_int = torch.einsum("bok,lk->bol", m_deriv, S_slice)

        return torch.einsum("bol,nl->bon", m_int, T_sum)

    def forward(self, x):
        B, C, H, W = x.shape
        xw = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        xw = self._1d(xw, self.W_w, W).reshape(B, H, self.out_ch, W).permute(0, 2, 1, 3)
        xh = xw.permute(0, 3, 1, 2).reshape(B * W, self.out_ch, H)
        xh = self._1d(xh, self.W_h, H)
        return xh.reshape(B, W, self.out_ch, H).permute(0, 2, 3, 1)


# ════════════════════════════════════════════════════════════════════
#  FNO
# ════════════════════════════════════════════════════════════════════


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


class SpectralConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.m1, self.m2 = modes1, modes2
        self.w1 = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, 2) / (in_ch * out_ch)
        )
        self.w2 = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, 2) / (in_ch * out_ch)
        )

    def _cm2(self, xr, xi, wr, wi):
        return torch.complex(
            torch.einsum("bixy,ioxy->boxy", xr, wr)
            - torch.einsum("bixy,ioxy->boxy", xi, wi),
            torch.einsum("bixy,ioxy->boxy", xr, wi)
            + torch.einsum("bixy,ioxy->boxy", xi, wr),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        m1, m2 = min(self.m1, H // 2), min(self.m2, W // 2 + 1)
        xf = torch.fft.rfft2(x)
        out = torch.zeros(
            B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        wr1, wi1 = self.w1[:, :, :m1, :m2, 0], self.w1[:, :, :m1, :m2, 1]
        wr2, wi2 = self.w2[:, :, :m1, :m2, 0], self.w2[:, :, :m1, :m2, 1]

        out[:, :, :m1, :m2] = self._cm2(
            xf.real[:, :, :m1, :m2], xf.imag[:, :, :m1, :m2], wr1, wi1
        )
        out[:, :, -m1:, :m2] = self._cm2(
            xf.real[:, :, -m1:, :m2], xf.imag[:, :, -m1:, :m2], wr2, wi2
        )
        return torch.fft.irfft2(out, s=(H, W))


# ════════════════════════════════════════════════════════════════════
#  MODEL WRAPPERS (LayerNorm 추가로 훈련 안정성 극대화)
# ════════════════════════════════════════════════════════════════════


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


def make_no_2d(conv_layer, width=32, depth=4, in_ch=3, out_ch=1):
    class NO2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Linear(in_ch, width)
            self.convs = nn.ModuleList([conv_layer() for _ in range(depth)])
            self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
            self.norms = nn.ModuleList([nn.GroupNorm(4, width) for _ in range(depth)])
            self.proj = nn.Sequential(
                nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
            )

        def forward(self, x):
            x = self.lift(x).permute(0, 3, 1, 2)
            for c, w, n in zip(self.convs, self.ws, self.norms):
                x = F.gelu(n(c(x) + w(x)))
            return self.proj(x.permute(0, 2, 3, 1))

    return NO2D()


# ════════════════════════════════════════════════════════════════════
#  DATA GENERATION (진정한 의미의 Subsampling)
# ════════════════════════════════════════════════════════════════════


def generate_burgers_master(n_samples, master_res=128):
    """
    초고해상도(128)에서 단 1번만 생성하여 일관된 물리현상 유지.
    테스트할 때는 이 배열을 [:, ::2] 식으로 슬라이스해서 꺼내 씁니다.
    """
    x = torch.linspace(0, 2 * math.pi, master_res)
    us, vs = [], []
    for _ in range(n_samples):
        u = torch.zeros(master_res)
        for k in range(1, 6):
            u += (
                (torch.rand(1).item() - 0.5)
                * 2
                * torch.sin(k * x + torch.rand(1).item() * 2 * math.pi)
            )
        uft = torch.fft.rfft(u)
        kv = torch.arange(uft.shape[0], dtype=torch.float32)
        uft = uft * torch.exp(-0.01 * kv**2 * 0.5)
        ur = torch.fft.irfft(uft, n=master_res)
        uft = uft - 0.1 * (1j * kv * torch.fft.rfft(ur**2 / 2))
        v = torch.fft.irfft(uft, n=master_res).real
        us.append(u)
        vs.append(v)

    U, V = torch.stack(us), torch.stack(vs)
    # 전체 데이터 기준 정규화
    return (U - U.mean()) / U.std(), (V - V.mean()) / V.std()


def generate_helmholtz_master(n_samples, master_res=64):
    """
    2D Darcy의 완벽한 대안: Helmholtz PDE (-Δu + u = f)
    a를 생성하고 이를 바탕으로 u를 완벽히 결정하여 모델이 100% 학습 가능.
    """
    As, Us = [], []
    N = master_res
    kx = torch.fft.fftfreq(N) * N
    ky = torch.fft.rfftfreq(N) * N
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
    K2 = (
        (2 * math.pi * Kx / N) ** 2 + (2 * math.pi * Ky / N) ** 2 + 1.0
    )  # -Laplacian + Identity

    for _ in range(n_samples):
        raw = torch.randn(N, N)
        a_ft = torch.fft.rfft2(raw) * torch.exp(-0.1 * (Kx**2 + Ky**2))
        a = torch.fft.irfft2(a_ft, s=(N, N))

        # 비선형 Source term 구성
        f = a**2 + torch.sin(a)

        # 스펙트럴 정확 해 (u)
        u_ft = torch.fft.rfft2(f) / K2
        u = torch.fft.irfft2(u_ft, s=(N, N))

        As.append(a)
        Us.append(u)

    A, U = torch.stack(As), torch.stack(Us)
    # 전체 데이터 기준 정규화
    return (A - A.mean()) / A.std(), (U - U.mean()) / U.std()


def make_input_1d(U):
    B, N = U.shape
    return torch.cat(
        [U.unsqueeze(-1), torch.linspace(0, 1, N).view(1, N, 1).expand(B, -1, -1)], -1
    )


def make_input_2d(A):
    B, H, W = A.shape
    g1, g2 = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    return torch.cat(
        [A.unsqueeze(-1), torch.stack([g1, g2], -1).unsqueeze(0).expand(B, -1, -1, -1)],
        -1,
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


# ════════════════════════════════════════════════════════════════════
#  BENCHMARK
# ════════════════════════════════════════════════════════════════════


def run_1d():
    print(
        "\n"
        + "=" * 65
        + "\n  1D 벤치마크: True Resolution Invariance (Master N=128)\n"
        + "=" * 65
    )
    U_master, V_master = generate_burgers_master(1000, 128)

    # Train at N=64 (subsample ::2)
    U_tr, V_tr = U_master[:800, ::2], V_master[:800, ::2]
    tr = DataLoader(
        TensorDataset(make_input_1d(U_tr), V_tr.unsqueeze(-1)), 32, shuffle=True
    )

    # Test at N=64
    U_te, V_te = U_master[800:, ::2], V_master[800:, ::2]
    te = DataLoader(TensorDataset(make_input_1d(U_te), V_te.unsqueeze(-1)), 32)

    fno = make_no_1d(lambda: SpectralConv1D(32, 32, 16)).to(DEVICE)
    cheb = make_no_1d(lambda: ChebConv1D(32, 32, 16)).to(DEVICE)
    hx = make_no_1d(lambda: HeinnXConv1D(32, 32, 16)).to(DEVICE)

    results = {}
    for name, model in [("FNO", fno), ("ChebNO", cheb), ("Heinn-X", hx)]:
        print(f"\n── {name} 학습 ─────────────────────────────────")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
        for ep in range(1, 51):
            loss = train_epoch(model, tr, opt)
            sched.step()
            if ep % 10 == 0:
                print(f"  Ep{ep:3d}  loss={loss:.4e}  rel-L2={evaluate(model, te):.4f}")

        results[name] = {}
        for res, step in [(64, 2), (32, 4), (16, 8)]:
            # [승리 요인] 완전 동일한 마스터 데이터에서 슬라이싱!
            Ut, Vt = U_master[800:, ::step], V_master[800:, ::step]
            results[name][res] = evaluate(
                model,
                DataLoader(TensorDataset(make_input_1d(Ut), Vt.unsqueeze(-1)), 32),
            )
    return results


def run_2d():
    print(
        "\n" + "=" * 65 + "\n  2D 벤치마크: Helmholtz PDE (Master 64x64)\n" + "=" * 65
    )
    A_master, U_master = generate_helmholtz_master(500, 64)

    # Train at 32x32 (subsample ::2)
    A_tr, U_tr = A_master[:400, ::2, ::2], U_master[:400, ::2, ::2]
    tr = DataLoader(
        TensorDataset(make_input_2d(A_tr), U_tr.unsqueeze(-1)), 16, shuffle=True
    )

    # Test at 32x32
    A_te, U_te = A_master[400:, ::2, ::2], U_master[400:, ::2, ::2]
    te = DataLoader(TensorDataset(make_input_2d(A_te), U_te.unsqueeze(-1)), 16)

    fno = make_no_2d(lambda: SpectralConv2D(32, 32, 12, 12)).to(DEVICE)
    cheb = make_no_2d(lambda: ChebConv2D(32, 32, 12)).to(DEVICE)
    hx = make_no_2d(lambda: HeinnXConv2D(32, 32, 12)).to(DEVICE)

    results = {}
    for name, model in [("FNO", fno), ("ChebNO", cheb), ("Heinn-X", hx)]:
        print(f"\n── {name} 학습 ─────────────────────────────────")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
        for ep in range(1, 41):
            loss = train_epoch(model, tr, opt)
            sched.step()
            if ep % 10 == 0:
                print(f"  Ep{ep:3d}  loss={loss:.4e}  rel-L2={evaluate(model, te):.4f}")

        results[name] = {}
        for res, step in [(32, 2), (16, 4)]:
            At, Ut = A_master[400:, ::step, ::step], U_master[400:, ::step, ::step]
            results[name][res] = evaluate(
                model,
                DataLoader(TensorDataset(make_input_2d(At), Ut.unsqueeze(-1)), 16),
            )
    return results


if __name__ == "__main__":
    r1d = run_1d()
    r2d = run_2d()

    print("\n" + "=" * 65 + "\n  FINAL SUMMARY\n" + "=" * 65)
    print("\n  1D Burgers (True Resolution Invariance):")
    for res in [64, 32, 16]:
        ef, ec, eh = r1d["FNO"][res], r1d["ChebNO"][res], r1d["Heinn-X"][res]
        tag = (
            "FNO"
            if min(ef, ec, eh) == ef
            else ("Cheb" if min(ef, ec, eh) == ec else "HX✓")
        )
        print(f"    res={res:3d}  FNO={ef:.4f}  Cheb={ec:.4f}  HX={eh:.4f}  [{tag}]")

    print("\n  2D Helmholtz PDE (True Physical Matching):")
    for res in [32, 16]:
        ef, ec, eh = r2d["FNO"][res], r2d["ChebNO"][res], r2d["Heinn-X"][res]
        tag = (
            "FNO"
            if min(ef, ec, eh) == ef
            else ("Cheb" if min(ef, ec, eh) == ec else "HX✓")
        )
        print(f"    {res}x{res}  FNO={ef:.4f}  Cheb={ec:.4f}  HX={eh:.4f}  [{tag}]")
