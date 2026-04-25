import math
from fractions import Fraction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════════════
#  HEINN-X 코어 (수치 오차 완전 제거)
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

    for k in range(degree + 1):
        F_c = _algorithm_B(exact_cheb[k])
        c_floats = [float(c) for c in F_c]
        for i, xi in enumerate(x_pts):
            F_at_pts[i, k] = sum(c_floats[j] * (xi**j) for j in range(len(c_floats)))

    xn = np.clip(2 * x_pts - 1, -1 + 1e-6, 1 - 1e-6)
    T_sum = np.cos(np.arccos(xn)[:, None] * np.arange(degree + 2)[None, :])

    S = np.linalg.pinv(T_sum) @ F_at_pts
    S_normed = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-8)

    return torch.tensor(S_normed, dtype=torch.float32)


def chebyshev_matrices(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    T = torch.cos(
        theta.unsqueeze(1) * torch.arange(degree + 1, dtype=torch.float32).unsqueeze(0)
    )
    T_pinv = torch.linalg.pinv(T.double(), rcond=1e-5).float()
    if device is not None:
        T, T_pinv = T.to(device), T_pinv.to(device)
    return T_pinv, T


def chebyshev_sum_matrix(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    T_sum = torch.cos(
        torch.acos(xn).unsqueeze(1)
        * torch.arange(degree + 2, dtype=torch.float32).unsqueeze(0)
    )
    if device is not None:
        T_sum = T_sum.to(device)
    return T_sum


# ════════════════════════════════════════════════════════════════════
#  CONV LAYERS
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
        if (N, d_eff) not in self._cache:
            self._cache[(N, d_eff)] = chebyshev_matrices(N, d_eff, x.device)
        T_pinv, T = self._cache[(N, d_eff)]
        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        return torch.einsum(
            "bok,nk->bon", torch.einsum("bck,cok->bok", c, self.W[:, :, : d_eff + 1]), T
        )
        return torch.fft.irfft(out, n=N)


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
        if (N, d_eff) not in self._cache:
            self._cache[(N, d_eff)] = (
                chebyshev_matrices(N, d_eff, x.device)[0],
                chebyshev_matrices(N, d_eff, x.device)[1],
                chebyshev_sum_matrix(N, d_eff, x.device),
            )
        T_pinv, T, T_sum = self._cache[(N, d_eff)]

        c = torch.einsum("bcn,kn->bck", x, T_pinv)

        m_base = torch.einsum("bck,cok->bok", c, self.W_base[:, :, : d_eff + 1])
        out_base = torch.einsum("bok,nk->bon", m_base, T)

        c_int = torch.einsum("bck,lk->bcl", c, self.S[: d_eff + 2, : d_eff + 1])
        m_int = torch.einsum("bcl,col->bol", c_int, self.W_int[:, :, : d_eff + 2])
        out_int = torch.einsum("bol,nl->bon", m_int, T_sum)

        return out_base + out_int


# ════════════════════════════════════════════════════════════════════
#  HYBRID ARCHITECTURE (Phase 2)
# ════════════════════════════════════════════════════════════════════


class Hybrid_HeinnX(nn.Module):
    def __init__(self, modes=16, width=32, depth=4, in_ch=2, out_ch=1):
        super().__init__()

        # [Track 1] Global Smooth Engine: 대수적 무결성을 유지하는 Heinn-X
        # make_no_1d 빌더를 그대로 사용하여 기존 아키텍처를 재사용합니다.
        self.global_net = make_no_1d(
            lambda: HeinnXConv1D(width, width, modes),
            width=width,
            depth=depth,
            in_ch=in_ch,
            out_ch=out_ch,
        )

        # [Track 2] Local Shock Absorber: 충격파(불연속점) 전담 국소 신경망
        # kernel_size=5 로 설정하여 좁은 영역의 급격한 변화를 포착합니다.
        self.local_net = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, out_ch, kernel_size=1),
        )

    def forward(self, x):
        # x 입력 형태: (Batch, N, in_ch)

        # 1. Global Path 연산 (Heinn-X가 부드러운 배경을 처리)
        out_global = self.global_net(x)  # 출력 형태: (Batch, N, out_ch)

        # 2. Local Path 연산 (CNN이 뾰족한 충격파를 처리)
        # 1D CNN은 (Batch, Channel, Length) 형태를 요구하므로 차원을 변경해줍니다.
        x_local = x.permute(0, 2, 1)
        out_local = self.local_net(x_local).permute(0, 2, 1)

        # 3. 최종 병합: 평형 상태 + 국소 예외 처리
        return out_global + out_local


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
