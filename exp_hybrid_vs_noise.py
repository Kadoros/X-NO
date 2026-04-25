import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
import scipy.integrate as integrate
import math
import time

# 장치 설정
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# ════════════════════════════════════════════════════════════════════
# 1. HEINN-X 핵심 엔진 (S-Matrix & Layers)
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
    return _trim([Fraction(0)] + _sv(_bM(d), [p[j] for j in range(d + 1)]))


def build_S_matrix(degree):
    N_pts = max(4 * (degree + 1), 128)
    x_pts = 0.5 * (1 + np.cos(math.pi * (2 * np.arange(N_pts) + 1) / (2 * N_pts)))
    xn = np.clip(2 * x_pts - 1, -1 + 1e-6, 1 - 1e-6)
    theta = np.arccos(xn)
    T_f = np.cos(theta[:, None] * np.arange(degree + 1)[None, :])
    T_sum = np.cos(theta[:, None] * np.arange(degree + 2)[None, :])
    Vm = np.vstack([x_pts**j for j in range(degree + 1)]).T
    Tkm = np.linalg.pinv(Vm) @ T_f
    Fpts = np.zeros((N_pts, degree + 1))
    for k in range(degree + 1):
        Fc = [float(c) for c in _algorithm_B(Tkm[:, k].tolist())]
        for i, xi in enumerate(x_pts):
            Fpts[i, k] = sum(Fc[j] * xi**j for j in range(len(Fc)))
    S = np.linalg.pinv(T_sum) @ Fpts
    return torch.tensor(S / (np.linalg.norm(S) + 1e-8), dtype=torch.float32)


def chebyshev_matrices(N, degree, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2 * x - 1).clamp(-1 + 1e-6, 1 - 1e-6)
    T = torch.cos(
        torch.acos(xn).unsqueeze(1)
        * torch.arange(degree + 1, dtype=torch.float32).unsqueeze(0)
    )
    Tp = torch.linalg.pinv(T.double(), rcond=1e-4).float()
    if device:
        T, Tp = T.to(device), Tp.to(device)
    return Tp, T


def chebyshev_sum_matrix(N, degree, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2 * x - 1).clamp(-1 + 1e-6, 1 - 1e-6)
    T = torch.cos(
        torch.acos(xn).unsqueeze(1)
        * torch.arange(degree + 2, dtype=torch.float32).unsqueeze(0)
    )
    if device:
        T = T.to(device)
    return T


class HybridHeinnXConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.register_buffer("S", build_S_matrix(degree))
        self.W_math = nn.Parameter(
            torch.randn(in_ch, out_ch, degree + 2) / math.sqrt(in_ch)
        )
        self.W_corr = nn.Parameter(torch.randn(in_ch, out_ch, degree + 1) * 0.01)
        self._c = {}

    def forward(self, x):
        B, C, N = x.shape
        d = min(self.degree, N // 2)
        k = (N, d, str(x.device))
        if k not in self._c:
            Tp, T = chebyshev_matrices(N, d, x.device)
            Ts = chebyshev_sum_matrix(N, d, x.device)
            self._c[k] = (Tp, T, Ts)
        Tp, T, Ts = self._c[k]
        c = torch.einsum("bcn,kn->bck", x, Tp)
        ci = torch.einsum("bck,lk->bcl", c, self.S[: d + 2, : d + 1])
        out_math = torch.einsum(
            "bol,nl->bon",
            torch.einsum("bcl,col->bol", ci, self.W_math[:, :, : d + 2]),
            Ts,
        )
        out_corr = torch.einsum(
            "bok,nk->bon",
            torch.einsum("bck,cok->bok", c, self.W_corr[:, :, : d + 1]),
            T,
        )
        return out_math + out_corr


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


class ChebConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, degree):
        super().__init__()
        self.degree = degree
        self.W = nn.Parameter(torch.randn(in_ch, out_ch, degree + 1) / math.sqrt(in_ch))
        self._c = {}

    def forward(self, x):
        B, C, N = x.shape
        d = min(self.degree, N // 2)
        k = (N, d, str(x.device))
        if k not in self._c:
            self._c[k] = chebyshev_matrices(N, d, x.device)
        Tp, T = self._c[k]
        c = torch.einsum("bcn,kn->bck", x, Tp)
        return torch.einsum(
            "bok,nk->bon", torch.einsum("bck,cok->bok", c, self.W[:, :, : d + 1]), T
        )


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
            return self.proj(x.permute(0, 2, 1)).squeeze(-1)  # [B, N] 출력 보장

    return NO1D()


# ════════════════════════════════════════════════════════════════════
# 2. 데이터 생성 및 실험 실행
# ════════════════════════════════════════════════════════════════════
def generate_noisy_kinematics(n_samples, res=128, noise_lvl=0.2):
    np.random.seed(42)
    t = np.linspace(0, 1, res)
    A_all, S_all = [], []
    for _ in range(n_samples):
        coeffs = np.random.randn(np.random.randint(2, 5)) * 5.0
        p_a = np.poly1d(coeffs)
        p_s = np.polyint(np.polyint(p_a, k=np.random.randn()), k=0.0)
        a_vals = p_a(t)
        a_noisy = a_vals + np.random.randn(res) * np.std(a_vals) * noise_lvl
        A_all.append(a_noisy)
        S_all.append(p_s(t))
    A, S = torch.tensor(np.array(A_all), dtype=torch.float32), torch.tensor(
        np.array(S_all), dtype=torch.float32
    )
    return (A - A.mean()) / A.std(), (S - S.mean()) / S.std()


def run_hybrid_experiment():
    print("🚀 [하이브리드 실험] 노이즈 20% 환경에서의 궤적 복원")
    A_data, S_data = generate_noisy_kinematics(1200, res=128, noise_lvl=0.2)
    A_tr, S_tr = A_data[:1000], S_data[:1000]
    A_te, S_te = A_data[1000:], S_data[1000:]

    def make_input(U):
        B, N = U.shape
        grid = torch.linspace(0, 1, N).view(1, N, 1).expand(B, -1, -1).to(U.device)
        return torch.cat([U.unsqueeze(-1), grid], dim=-1)

    tr_loader = DataLoader(
        TensorDataset(make_input(A_tr), S_tr), batch_size=64, shuffle=True
    )

    models = {
        "FNO": make_no_1d(lambda: SpectralConv1D(32, 32, 16)).to(DEVICE),
        "ChebNO": make_no_1d(lambda: ChebConv1D(32, 32, 16)).to(DEVICE),
        "Heinn-X (Hybrid)": make_no_1d(lambda: HybridHeinnXConv1D(32, 32, 16)).to(
            DEVICE
        ),
    }

    for name, model in models.items():
        print(f"  - {name} 학습 중...")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for ep in range(1, 41):
            for xb, yb in tr_loader:
                opt.zero_grad()
                pred = model(xb.to(DEVICE))
                loss = F.mse_loss(pred, yb.to(DEVICE))  # 차원 일치 확인됨
                loss.backward()
                opt.step()

    print("\n📊 결과 보고 (Relative L2 Error)")
    sub_A = A_te[:, ::16].to(DEVICE)  # 8개 센서 포인트
    sub_S = S_te[:, ::16].to(DEVICE)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(make_input(sub_A))
            err = (pred - sub_S).norm() / (sub_S.norm() + 1e-8)
            print(f"  [{name}] 8-Sensors Error: {err.item():.4f}")


if __name__ == "__main__":
    run_hybrid_experiment()
