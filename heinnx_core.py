import math
from fractions import Fraction
import numpy as np
import torch
import torch.nn as nn


# ── [수학 엔진] Heinn-X Algorithm B ──
def _algorithm_B(p):
    def _frac(p):
        return [Fraction(c) for c in p]

    def _trim(p):
        while len(p) > 1 and p[-1] == 0:
            p.pop()
        return p

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
        return [Fraction(0), p[0] - p[1] / 2, p[1] / 2]
    sol = _sv(_bM(d), [p[j] for j in range(d + 1)])
    return _trim([Fraction(0)] + sol)


# ── [수치 엔진] Exact Chebyshev Coeffs ──
def exact_shifted_chebyshev_coeffs(degree):
    coeffs = [[Fraction(1)], [Fraction(-1), Fraction(2)]]
    if degree == 0:
        return [coeffs[0]]
    for k in range(1, degree):
        Tk = coeffs[k]
        Tk_m1 = coeffs[k - 1]
        t1 = [Fraction(0)] + [c * 4 for c in Tk]
        t2 = [c * -2 for c in Tk]
        m_l = len(t1)
        t2 += [Fraction(0)] * (m_l - len(t2))
        Tk_m1_p = Tk_m1 + [Fraction(0)] * (m_l - len(Tk_m1))
        coeffs.append([t1[i] + t2[i] - Tk_m1_p[i] for i in range(m_l)])
    return coeffs[: degree + 1]


def build_heinnx_s_matrix(degree: int):
    N_pts = max(4 * (degree + 1), 128)
    x_pts = 0.5 * (1 + np.cos(math.pi * (2 * np.arange(N_pts) + 1) / (2 * N_pts)))
    exact_cheb = exact_shifted_chebyshev_coeffs(degree)
    F_at_pts = np.zeros((N_pts, degree + 1))
    for k in range(degree + 1):
        F_c = _algorithm_B(exact_cheb[k])
        c_f = [float(c) for c in F_c]
        for i, xi in enumerate(x_pts):
            F_at_pts[i, k] = sum(c_f[j] * (xi**j) for j in range(len(c_f)))
    xn = np.clip(2 * x_pts - 1, -1 + 1e-6, 1 - 1e-6)
    T_sum = np.cos(np.arccos(xn)[:, None] * np.arange(degree + 2)[None, :])
    S = np.linalg.pinv(T_sum) @ F_at_pts
    S_normed = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-8)
    return torch.tensor(S_normed, dtype=torch.float32)


# ── [기저 행렬 유틸] ──
def chebyshev_matrices(N: int, degree: int, device=None):
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    T = torch.cos(
        torch.acos(xn).unsqueeze(1)
        * torch.arange(degree + 1, dtype=torch.float32).unsqueeze(0)
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


# ── [실전 레이어] ──
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
        self.register_buffer("S", build_heinnx_s_matrix(degree))
        self._cache = {}

    def forward(self, x):
        B, C, N = x.shape
        d_eff = min(self.degree, N // 2)
        key = (N, d_eff, str(x.device))
        if key not in self._cache:
            T_pinv, T = chebyshev_matrices(N, d_eff, x.device)
            T_sum = chebyshev_sum_matrix(N, d_eff, x.device)
            self._cache[key] = (T_pinv, T, T_sum)
        T_pinv, T, T_sum = self._cache[key]

        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        m_base = torch.einsum("bck,cok->bok", c, self.W_base[:, :, : d_eff + 1])
        out_base = torch.einsum("bok,nk->bon", m_base, T)

        c_int = torch.einsum("bck,lk->bcl", c, self.S[: d_eff + 2, : d_eff + 1])
        m_int = torch.einsum("bcl,col->bol", c_int, self.W_int[:, :, : d_eff + 2])
        out_int = torch.einsum("bol,nl->bon", m_int, T_sum)
        return out_base + out_int


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
