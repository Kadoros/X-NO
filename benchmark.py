"""
benchmark_v6.py  —  Heinn-X v6  (모든 문제 수정)
════════════════════════════════════════════════════════════════════════

[v5에서 남은 문제들]

문제 A (에러): 해상도 불일치
  - T_pinv가 train_res=64 고정 → test_res=32 입력 시 차원 오류
  → 수정: forward()에서 입력 N에 맞게 기저 행렬 동적 계산

문제 B (학습 실패의 진짜 원인): S 행렬 신호 폭발
  - indefinite sum = 누적합 연산 → operator norm ≈ O(degree²)
  - S의 최대 singular value ≈ 220 → 신호 13x 증폭
  - 4 레이어면 13^4 ≈ 28000x → gradient 발산/소실
  → 수정: S를 최대 singular value로 정규화 + 학습 가능한 α 스케일

문제 C (아키텍처 개념): Heinn-X를 메인 연산자로 쓰는 것 자체가 부적절
  - Heinn-X의 S는 누적합 연산자 → 고주파 성분 잘 잡지 못함
  - 반면 학습 데이터(Burgers, Darcy)는 고주파 성분이 핵심
  → 수정: 두 가지 버전 제공
    v6a: ChebConv (S 없이, 순수 Chebyshev mixing — FNO와 공정한 비교)
    v6b: HeinnXConv (S 정규화 + α 스케일 — Heinn-X 수학 최대한 활용)

[아키텍처]
  FNO:     x → FFT → W_complex → iFFT
  ChebConv: x → T_pinv → W_real → T_eval       (같은 구조, 다른 기저)
  HeinnXConv: x → T_pinv → W_real → S_norm*α → T_eval_sum  (적분 포함)

Run: python benchmark_v6.py
════════════════════════════════════════════════════════════════════════
"""

import math, time, random
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
#  HEINN-X 수학 코어
# ════════════════════════════════════════════════════════════════════


def _algorithm_B(p):
    """exact rational indefinite sum"""

    def _frac(p):
        return [Fraction(c) for c in p]

    def _trim(p):
        while len(p) > 1 and p[-1] == 0:
            p.pop()
        return p

    def _ls(a, b):
        a, b = Fraction(a), Fraction(b)
        return [Fraction(0), b - a / 2, a / 2]

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


# ════════════════════════════════════════════════════════════════════
#  기저 행렬 (동적 계산 — 임의 N 지원)
# ════════════════════════════════════════════════════════════════════


def chebyshev_matrices(N: int, degree: int, device=None):
    """
    N 포인트에 대한 Chebyshev 기저 행렬 계산.
    동적으로 호출되므로 해상도 불일치 문제 해결.

    반환:
      T_pinv: [d+1, N] — 분석 행렬 (x → Cheby coeffs)
      T_eval: [N, d+1] — 합성 행렬 (Cheby coeffs → x)
    """
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    ks = torch.arange(degree + 1, dtype=torch.float32)
    T = torch.cos(theta.unsqueeze(1) * ks.unsqueeze(0))  # [N, d+1]
    T_pinv = torch.linalg.pinv(T.double()).float()  # [d+1, N]
    if device is not None:
        T, T_pinv = T.to(device), T_pinv.to(device)
    return T_pinv, T


def chebyshev_sum_matrix(N: int, degree: int, device=None):
    """
    T_eval_sum: [N, d+2] — Chebyshev 기저 (sum 출력용, degree+1까지)
    """
    x = torch.linspace(0, 1, N, dtype=torch.float32)
    xn = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(xn)
    ks = torch.arange(degree + 2, dtype=torch.float32)
    T_sum = torch.cos(theta.unsqueeze(1) * ks.unsqueeze(0))  # [N, d+2]
    if device is not None:
        T_sum = T_sum.to(device)
    return T_sum


def build_S_matrix(degree: int) -> torch.Tensor:
    """
    Chebyshev 계수 공간의 indefinite sum 연산자 S: [d+2, d+1]
    S @ c_f = c_Σf

    수치 안정성: S를 최대 singular value로 정규화하여 반환.
    실제 스케일은 HeinnXConv의 학습 가능한 α로 복원.
    """
    N_pts = max(4 * (degree + 1), 128)
    k_pts = np.arange(N_pts)
    x_pts = 0.5 * (1 + np.cos(math.pi * (2 * k_pts + 1) / (2 * N_pts)))

    xn = np.clip(2 * x_pts - 1, -1 + 1e-6, 1 - 1e-6)
    theta = np.arccos(xn)
    T_f = np.cos(theta[:, None] * np.arange(degree + 1)[None, :])  # [N, d+1]
    T_sum = np.cos(theta[:, None] * np.arange(degree + 2)[None, :])  # [N, d+2]

    # T_k → monomial → algorithm_B → sample values
    V_mono = np.vstack([x_pts**j for j in range(degree + 1)]).T
    V_mono_pinv = np.linalg.pinv(V_mono)
    T_k_mono = V_mono_pinv @ T_f  # [d+1, d+1]

    F_at_samples = np.zeros((N_pts, degree + 1))
    for k in range(degree + 1):
        mono_c = T_k_mono[:, k].tolist()
        F_mono = [float(c) for c in _algorithm_B(mono_c)]
        for i, xi in enumerate(x_pts):
            F_at_samples[i, k] = sum(F_mono[j] * xi**j for j in range(len(F_mono)))

    T_sum_pinv = np.linalg.pinv(T_sum)
    S = T_sum_pinv @ F_at_samples  # [d+2, d+1]

    # 정규화: 최대 singular value로 나눔
    sv = np.linalg.svd(S, compute_uv=False)
    S_norm = S / sv[0]

    return torch.tensor(S_norm, dtype=torch.float32), float(sv[0])


# ════════════════════════════════════════════════════════════════════
#  CONV LAYERS
# ════════════════════════════════════════════════════════════════════


class ChebConv1D(nn.Module):
    """
    순수 Chebyshev mixing (S 없음).
    FNO SpectralConv과 1:1 대응 구조.

    FNO:      x → FFT → W_complex → iFFT
    ChebConv: x → T_pinv → W_real → T_eval

    파라미터 수: in_ch × out_ch × (degree+1) [실수]
    FNO 대비:   in_ch × out_ch × modes × 2  [복소수]
    → degree+1 = 2*modes 로 설정하면 동일한 파라미터 수
    """

    def __init__(self, in_ch: int, out_ch: int, degree: int):
        super().__init__()
        self.degree = degree
        scale = (2.0 / (in_ch * (degree + 1))) ** 0.5
        self.W = nn.Parameter(scale * torch.randn(in_ch, out_ch, degree + 1))
        self._cache = {}

    def _get_matrices(self, N, device):
        if N not in self._cache:
            T_pinv, T_eval = chebyshev_matrices(N, self.degree, device)
            self._cache[N] = (T_pinv, T_eval)
        return self._cache[N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, N] → [B, out_ch, N]"""
        B, C, N = x.shape
        T_pinv, T_eval = self._get_matrices(N, x.device)
        c = torch.einsum("bcn,kn->bck", x, T_pinv)  # [B, in_ch, d+1]
        m = torch.einsum("bck,cok->bok", c, self.W)  # [B, out_ch, d+1]
        return torch.einsum("bok,nk->bon", m, T_eval)  # [B, out_ch, N]


class HeinnXConv1D(nn.Module):
    """
    Heinn-X indefinite sum + Chebyshev mixing.

    Pipeline: x → T_pinv → W → α*S_norm → T_eval_sum → y

    핵심 수정:
    1. S를 최대 singular value로 정규화 (신호 폭발 방지)
    2. α: 학습 가능한 스케일 파라미터 (실제 스케일 복원)
    3. 동적 기저 계산 (임의 해상도 지원)

    FNO SpectralConv 대비: 적분 연산자(S)를 inductive bias로 내장
    """

    def __init__(self, in_ch: int, out_ch: int, degree: int):
        super().__init__()
        self.degree = degree
        scale = (2.0 / (in_ch * (degree + 1))) ** 0.5
        self.W = nn.Parameter(scale * torch.randn(in_ch, out_ch, degree + 1))

        print(f"    S 행렬 계산 중 (degree={degree})...", end="", flush=True)
        S_norm, sv0 = build_S_matrix(degree)
        print(f" 완료 (max_sv={sv0:.1f}, 정규화 적용)")
        self.register_buffer("S", S_norm)  # [d+2, d+1], 정규화됨

        # α: 학습 가능한 스케일 (초기값 = 정규화된 sv0의 역수로 세팅)
        # 초기에는 S 효과를 최소화하고, 학습을 통해 적절한 스케일 찾기
        self.alpha = nn.Parameter(torch.tensor(1.0 / sv0))
        self._cache_pinv = {}
        self._cache_eval = {}

    def _get_matrices(self, N, device):
        if N not in self._cache_pinv:
            T_pinv, _ = chebyshev_matrices(N, self.degree, device)
            T_sum = chebyshev_sum_matrix(N, self.degree, device)
            self._cache_pinv[N] = T_pinv
            self._cache_eval[N] = T_sum
        return self._cache_pinv[N], self._cache_eval[N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, N] → [B, out_ch, N]"""
        B, C, N = x.shape
        T_pinv, T_sum = self._get_matrices(N, x.device)

        c = torch.einsum("bcn,kn->bck", x, T_pinv)  # [B, in_ch, d+1]
        m = torch.einsum("bck,cok->bok", c, self.W)  # [B, out_ch, d+1]
        s = torch.einsum("bok,lk->bol", m, self.S) * self.alpha  # [B, out_ch, d+2]
        return torch.einsum("bol,nl->bon", s, T_sum)  # [B, out_ch, N]


class ChebConv2D(nn.Module):
    """분리 가능한 2D Chebyshev mixing"""

    def __init__(self, in_ch: int, out_ch: int, degree: int):
        super().__init__()
        self.degree = degree
        self.out_ch = out_ch
        scale_w = (2.0 / (in_ch * (degree + 1))) ** 0.5
        scale_h = (2.0 / (out_ch * (degree + 1))) ** 0.5
        self.W_w = nn.Parameter(scale_w * torch.randn(in_ch, out_ch, degree + 1))
        self.W_h = nn.Parameter(scale_h * torch.randn(out_ch, out_ch, degree + 1))
        self._cache = {}

    def _get_matrices(self, N, device):
        if N not in self._cache:
            T_pinv, T_eval = chebyshev_matrices(N, self.degree, device)
            self._cache[N] = (T_pinv, T_eval)
        return self._cache[N]

    def _apply1d(self, x, W, N):
        T_pinv, T_eval = self._get_matrices(N, x.device)
        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        m = torch.einsum("bck,cok->bok", c, W)
        return torch.einsum("bok,nk->bon", m, T_eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_w = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        x_w = self._apply1d(x_w, self.W_w, W)
        x_w = x_w.reshape(B, H, self.out_ch, W).permute(0, 2, 1, 3)
        x_h = x_w.permute(0, 3, 1, 2).reshape(B * W, self.out_ch, H)
        x_h = self._apply1d(x_h, self.W_h, H)
        return x_h.reshape(B, W, self.out_ch, H).permute(0, 2, 3, 1)


class HeinnXConv2D(nn.Module):
    """분리 가능한 2D Heinn-X conv (S 정규화 적용)"""

    def __init__(self, in_ch: int, out_ch: int, degree: int):
        super().__init__()
        self.degree = degree
        self.out_ch = out_ch
        scale_w = (2.0 / (in_ch * (degree + 1))) ** 0.5
        scale_h = (2.0 / (out_ch * (degree + 1))) ** 0.5
        self.W_w = nn.Parameter(scale_w * torch.randn(in_ch, out_ch, degree + 1))
        self.W_h = nn.Parameter(scale_h * torch.randn(out_ch, out_ch, degree + 1))
        print(f"    S 행렬 계산 중 (degree={degree})...", end="", flush=True)
        S_norm, sv0 = build_S_matrix(degree)
        print(f" 완료 (max_sv={sv0:.1f})")
        self.register_buffer("S", S_norm)
        self.alpha = nn.Parameter(torch.tensor(1.0 / sv0))
        self._cache_pinv = {}
        self._cache_eval = {}

    def _get_matrices(self, N, device):
        if N not in self._cache_pinv:
            T_pinv, _ = chebyshev_matrices(N, self.degree, device)
            T_sum = chebyshev_sum_matrix(N, self.degree, device)
            self._cache_pinv[N] = T_pinv
            self._cache_eval[N] = T_sum
        return self._cache_pinv[N], self._cache_eval[N]

    def _apply1d(self, x, W, N):
        T_pinv, T_sum = self._get_matrices(N, x.device)
        c = torch.einsum("bcn,kn->bck", x, T_pinv)
        m = torch.einsum("bck,cok->bok", c, W)
        s = torch.einsum("bok,lk->bol", m, self.S) * self.alpha
        return torch.einsum("bol,nl->bon", s, T_sum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_w = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        x_w = self._apply1d(x_w, self.W_w, W)
        x_w = x_w.reshape(B, H, self.out_ch, W).permute(0, 2, 1, 3)
        x_h = x_w.permute(0, 3, 1, 2).reshape(B * W, self.out_ch, H)
        x_h = self._apply1d(x_h, self.W_h, H)
        return x_h.reshape(B, W, self.out_ch, H).permute(0, 2, 3, 1)


# ════════════════════════════════════════════════════════════════════
#  FNO (변경 없음)
# ════════════════════════════════════════════════════════════════════


class SpectralConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, 2))

    def _cmul(self, x_ft, w):
        wr, wi = w[..., 0], w[..., 1]
        xr, xi = x_ft.real, x_ft.imag
        return torch.complex(
            torch.einsum("bim,iom->bom", xr, wr) - torch.einsum("bim,iom->bom", xi, wi),
            torch.einsum("bim,iom->bom", xr, wi) + torch.einsum("bim,iom->bom", xi, wr),
        )

    def forward(self, x):
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_m = min(self.modes, N // 2 + 1)
        out_ft = torch.zeros(
            B, self.weights.shape[1], N // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :n_m] = self._cmul(x_ft[:, :, :n_m], self.weights[:, :, :n_m])
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class SpectralConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    def _cmul2d(self, x_ft, w):
        wr, wi = w[..., 0], w[..., 1]
        xr, xi = x_ft.real, x_ft.imag
        return torch.complex(
            torch.einsum("bixy,ioxy->boxy", xr, wr)
            - torch.einsum("bixy,ioxy->boxy", xi, wi),
            torch.einsum("bixy,ioxy->boxy", xr, wi)
            + torch.einsum("bixy,ioxy->boxy", xi, wr),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self._cmul2d(
            x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._cmul2d(
            x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


# ════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ════════════════════════════════════════════════════════════════════


class FNO1D(nn.Module):
    def __init__(self, modes=16, width=32, depth=4, in_ch=2, out_ch=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList(
            [SpectralConv1D(width, width, modes) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
        )

    def forward(self, x):
        x = self.lift(x).permute(0, 2, 1)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        return self.proj(x.permute(0, 2, 1))


class ChebNO1D(nn.Module):
    """
    Chebyshev Neural Operator 1D.
    FNO와 동일 구조, 기저만 Chebyshev로 교체.
    degree = 2*modes 로 설정하면 파라미터 수 동일 (실수 vs 복소수).
    """

    def __init__(self, degree=32, width=32, depth=4, in_ch=2, out_ch=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList(
            [ChebConv1D(width, width, degree) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(min(8, width), width) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
        )

    def forward(self, x):
        x = self.lift(x).permute(0, 2, 1)
        for conv, w, norm in zip(self.convs, self.ws, self.norms):
            x = F.gelu(norm(conv(x) + w(x)))
        return self.proj(x.permute(0, 2, 1))


class HeinnXNO1D(nn.Module):
    """
    Heinn-X Neural Operator 1D v6.
    S 정규화 + 학습 가능한 α 스케일.
    """

    def __init__(self, degree=16, width=32, depth=4, in_ch=2, out_ch=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        print(f"\n[HeinnXNO1D] 초기화 (degree={degree}, depth={depth})")
        self.hx = nn.ModuleList(
            [HeinnXConv1D(width, width, degree) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(min(8, width), width) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
        )

    def forward(self, x):
        x = self.lift(x).permute(0, 2, 1)
        for hx, w, norm in zip(self.hx, self.ws, self.norms):
            x = F.gelu(norm(hx(x) + w(x)))
        return self.proj(x.permute(0, 2, 1))


class FNO2D(nn.Module):
    def __init__(self, modes=12, width=32, depth=4, in_ch=3, out_ch=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList(
            [SpectralConv2D(width, width, modes, modes) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
        )

    def forward(self, x):
        x = self.lift(x).permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        return self.proj(x.permute(0, 2, 3, 1))


class ChebNO2D(nn.Module):
    def __init__(self, degree=24, width=32, depth=4, in_ch=3, out_ch=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList(
            [ChebConv2D(width, width, degree) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(min(8, width), width) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
        )

    def forward(self, x):
        x = self.lift(x).permute(0, 3, 1, 2)
        for conv, w, norm in zip(self.convs, self.ws, self.norms):
            x = F.gelu(norm(conv(x) + w(x)))
        return self.proj(x.permute(0, 2, 3, 1))


class HeinnXNO2D(nn.Module):
    def __init__(self, degree=10, width=32, depth=4, in_ch=3, out_ch=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, width)
        print(f"\n[HeinnXNO2D] 초기화 (degree={degree}, depth={depth})")
        self.hx = nn.ModuleList(
            [HeinnXConv2D(width, width, degree) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList(
            [nn.GroupNorm(min(8, width), width) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch)
        )

    def forward(self, x):
        x = self.lift(x).permute(0, 3, 1, 2)
        for hx, w, norm in zip(self.hx, self.ws, self.norms):
            x = F.gelu(norm(hx(x) + w(x)))
        return self.proj(x.permute(0, 2, 3, 1))


# ════════════════════════════════════════════════════════════════════
#  데이터셋
# ════════════════════════════════════════════════════════════════════


def make_burgers_1d(n_samples, grid_size, nu=0.01):
    x = torch.linspace(0, 2 * math.pi, grid_size)
    us, vs = [], []
    for _ in range(n_samples):
        u = torch.zeros(grid_size)
        for k in range(1, random.randint(2, 5) + 1):
            amp = (torch.rand(1).item() - 0.5) * 2
            phase = torch.rand(1).item() * 2 * math.pi
            u += amp * torch.sin(k * x + phase)
        u_ft = torch.fft.rfft(u)
        k_vec = torch.arange(u_ft.shape[0], dtype=torch.float32)
        u_ft = u_ft * torch.exp(-nu * k_vec**2 * 0.5)
        u_real = torch.fft.irfft(u_ft, n=grid_size)
        u_sq_ft = torch.fft.rfft(u_real**2 / 2)
        u_ft = u_ft - 0.1 * (1j * k_vec * u_sq_ft)
        us.append(u)
        vs.append(torch.fft.irfft(u_ft, n=grid_size).real)
    return torch.stack(us), torch.stack(vs)


def make_darcy_2d(n_samples, grid_size):
    As, Us = [], []
    for _ in range(n_samples):
        raw = torch.randn(grid_size, grid_size)
        raw_ft = torch.fft.rfft2(raw)
        kx = torch.fft.fftfreq(grid_size).reshape(-1, 1)
        ky = torch.fft.rfftfreq(grid_size).reshape(1, -1)
        a = (
            torch.clamp(
                torch.fft.irfft2(
                    raw_ft * torch.exp(-8 * (kx**2 + ky**2)), s=(grid_size, grid_size)
                ),
                min=0.1,
            )
            + 0.5
        )
        freq_x = torch.fft.fftfreq(grid_size) * grid_size
        freq_y = torch.fft.rfftfreq(grid_size) * grid_size
        Kx, Ky = torch.meshgrid(freq_x, freq_y, indexing="ij")
        K2 = Kx**2 + Ky**2
        K2[0, 0] = 1.0
        rhs_ft = (
            torch.ones(grid_size, grid_size // 2 + 1, dtype=torch.cfloat)
            / a.mean().item()
        )
        rhs_ft[0, 0] = 0
        u = torch.fft.irfft2(rhs_ft / K2, s=(grid_size, grid_size))
        As.append(a)
        Us.append(u)
    return torch.stack(As), torch.stack(Us)


def make_input_1d(U):
    B, N = U.shape
    grid = torch.linspace(0, 1, N).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
    return torch.cat([U.unsqueeze(-1), grid], dim=-1)


def make_input_2d(A):
    B, H, W = A.shape
    g1, g2 = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    grid = torch.stack([g1, g2], -1).unsqueeze(0).expand(B, -1, -1, -1)
    return torch.cat([A.unsqueeze(-1), grid], dim=-1)


# ════════════════════════════════════════════════════════════════════
#  학습 유틸
# ════════════════════════════════════════════════════════════════════


def relative_l2(pred, target):
    return ((pred - target).norm() / (target.norm() + 1e-8)).item()


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu())
        targets.append(yb)
    return relative_l2(torch.cat(preds), torch.cat(targets))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════
#  1D 벤치마크
# ════════════════════════════════════════════════════════════════════


def run_1d_benchmark():
    print("\n" + "=" * 65)
    print("  1D 벤치마크: Burgers-like Equation")
    print("  Train res=64  |  Test: 64 → 32 → 16 → 8")
    print("=" * 65)

    TRAIN_RES = 64
    N_TRAIN, N_TEST = 800, 200
    BATCH, EPOCHS, LR = 32, 60, 3e-3
    MODES = 16  # FNO
    # ChebConv: degree=2*MODES 로 파라미터 수 맞춤 (실수 vs 복소수)
    CHEB_DEGREE = 2 * MODES  # = 32  → params: 32*32*33 ≈ 33,792/layer
    # FNO params/layer: 32*32*16*2 = 32,768/layer — 거의 동일
    HEINNX_DEGREE = MODES  # = 16  → params: 32*32*17 ≈ 17,408/layer (+ α×depth)
    WIDTH, DEPTH = 32, 4

    print(f"\n데이터 생성 (N={N_TRAIN+N_TEST}, res={TRAIN_RES})...", flush=True)
    U_all, V_all = make_burgers_1d(N_TRAIN + N_TEST, TRAIN_RES)
    X_tr = make_input_1d(U_all[:N_TRAIN])
    Y_tr = V_all[:N_TRAIN].unsqueeze(-1)
    X_te = make_input_1d(U_all[N_TRAIN:])
    Y_te = V_all[N_TRAIN:].unsqueeze(-1)
    tr_loader = DataLoader(TensorDataset(X_tr, Y_tr), BATCH, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, Y_te), BATCH)

    fno = FNO1D(modes=MODES, width=WIDTH, depth=DEPTH).to(DEVICE)
    chebno = ChebNO1D(degree=CHEB_DEGREE, width=WIDTH, depth=DEPTH).to(DEVICE)
    hxno = HeinnXNO1D(degree=HEINNX_DEGREE, width=WIDTH, depth=DEPTH).to(DEVICE)

    print(f"\n파라미터 수:")
    print(f"  FNO (modes={MODES}):           {count_params(fno):>10,}")
    print(f"  ChebNO (degree={CHEB_DEGREE}):      {count_params(chebno):>10,}")
    print(f"  Heinn-X (degree={HEINNX_DEGREE}):    {count_params(hxno):>10,}")
    fno_pp = WIDTH * WIDTH * MODES * 2
    cheb_pp = WIDTH * WIDTH * (CHEB_DEGREE + 1)
    hx_pp = WIDTH * WIDTH * (HEINNX_DEGREE + 1)
    print(
        f"\n  params/layer: FNO={fno_pp:,} | ChebNO={cheb_pp:,} | Heinn-X={hx_pp:,}+α"
    )

    results = {}
    for name, model in [("FNO", fno), ("ChebNO", chebno), ("Heinn-X v6", hxno)]:
        print(f"\n── {name} 학습 ────────────────────────────────────────")
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            loss = train_epoch(model, tr_loader, opt, DEVICE)
            sched.step()
            if ep % 10 == 0:
                err = evaluate(model, te_loader, DEVICE)
                # α 값 출력 (Heinn-X만)
                if hasattr(model, "hx"):
                    alphas = [f"{l.alpha.item():.4f}" for l in model.hx]
                    print(
                        f"  Epoch {ep:3d}  loss={loss:.4e}  rel-L2={err:.4f}  α={alphas}"
                    )
                else:
                    print(f"  Epoch {ep:3d}  loss={loss:.4e}  rel-L2={err:.4f}")
        train_time = time.time() - t0

        print(f"\n  해상도 일반화 ({name}):")
        res_errors = {}
        for test_res in [64, 32, 16, 8]:
            U_t, V_t = make_burgers_1d(N_TEST, test_res)
            loader_t = DataLoader(
                TensorDataset(make_input_1d(U_t), V_t.unsqueeze(-1)), BATCH
            )
            err = evaluate(model, loader_t, DEVICE)
            res_errors[test_res] = err
            print(f"    res={test_res:3d}: rel-L2 = {err:.4f}")
        results[name] = {"train_time": train_time, "res_errors": res_errors}

    print(f"\n{'─'*65}")
    print(f"  1D 요약")
    print(f"{'─'*65}")
    print(f"  {'res':>6}  {'FNO':>8}  {'ChebNO':>8}  {'Heinn-X':>8}  winner")
    for res in [64, 32, 16, 8]:
        e_f = results["FNO"]["res_errors"][res]
        e_c = results["ChebNO"]["res_errors"][res]
        e_h = results["Heinn-X v6"]["res_errors"][res]
        best = min(e_f, e_c, e_h)
        w = "FNO" if best == e_f else ("Cheb" if best == e_c else "HX ✓")
        print(f"  {res:>6}  {e_f:>8.4f}  {e_c:>8.4f}  {e_h:>8.4f}  {w}")
    return results


# ════════════════════════════════════════════════════════════════════
#  2D 벤치마크
# ════════════════════════════════════════════════════════════════════


def run_2d_benchmark():
    print("\n" + "=" * 65)
    print("  2D 벤치마크: Darcy-Flow-like PDE")
    print("  Train res=32×32  |  Test: 32 → 16 → 8")
    print("=" * 65)

    TRAIN_RES = 32
    N_TRAIN, N_TEST = 400, 100
    BATCH, EPOCHS, LR = 16, 50, 3e-3
    MODES, WIDTH, DEPTH = 12, 32, 4
    CHEB_DEGREE = 2 * MODES  # 24
    HEINNX_DEGREE = MODES  # 12

    print(f"\n데이터 생성 (N={N_TRAIN+N_TEST}, res={TRAIN_RES})...", flush=True)
    A_all, U_all = make_darcy_2d(N_TRAIN + N_TEST, TRAIN_RES)
    X_tr = make_input_2d(A_all[:N_TRAIN])
    Y_tr = U_all[:N_TRAIN].unsqueeze(-1)
    X_te = make_input_2d(A_all[N_TRAIN:])
    Y_te = U_all[N_TRAIN:].unsqueeze(-1)
    tr_loader = DataLoader(TensorDataset(X_tr, Y_tr), BATCH, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, Y_te), BATCH)

    fno2 = FNO2D(modes=MODES, width=WIDTH, depth=DEPTH).to(DEVICE)
    cheb2 = ChebNO2D(degree=CHEB_DEGREE, width=WIDTH, depth=DEPTH).to(DEVICE)
    hxno2 = HeinnXNO2D(degree=HEINNX_DEGREE, width=WIDTH, depth=DEPTH).to(DEVICE)

    print(f"\n파라미터 수:")
    print(f"  FNO2D:         {count_params(fno2):>10,}")
    print(f"  ChebNO2D:      {count_params(cheb2):>10,}")
    print(f"  Heinn-X 2D:    {count_params(hxno2):>10,}")

    results = {}
    for name, model in [("FNO2D", fno2), ("ChebNO2D", cheb2), ("Heinn-X 2D v6", hxno2)]:
        print(f"\n── {name} 학습 ────────────────────────────────────────")
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            loss = train_epoch(model, tr_loader, opt, DEVICE)
            sched.step()
            if ep % 10 == 0:
                err = evaluate(model, te_loader, DEVICE)
                if hasattr(model, "hx"):
                    alphas = [f"{l.alpha.item():.4f}" for l in model.hx]
                    print(
                        f"  Epoch {ep:3d}  loss={loss:.4e}  rel-L2={err:.4f}  α={alphas}"
                    )
                else:
                    print(f"  Epoch {ep:3d}  loss={loss:.4e}  rel-L2={err:.4f}")
        train_time = time.time() - t0

        print(f"\n  해상도 일반화 ({name}):")
        res_errors = {}
        for test_res in [32, 16, 8]:
            A_t, U_t = make_darcy_2d(N_TEST, test_res)
            loader_t = DataLoader(
                TensorDataset(make_input_2d(A_t), U_t.unsqueeze(-1)), BATCH
            )
            err = evaluate(model, loader_t, DEVICE)
            res_errors[test_res] = err
            print(f"    res={test_res}×{test_res}: rel-L2 = {err:.4f}")
        results[name] = {"train_time": train_time, "res_errors": res_errors}

    print(f"\n{'─'*65}")
    print(f"  2D 요약")
    print(f"{'─'*65}")
    print(f"  {'res':>8}  {'FNO2D':>8}  {'ChebNO':>8}  {'Heinn-X':>8}  winner")
    for res in [32, 16, 8]:
        e_f = results["FNO2D"]["res_errors"][res]
        e_c = results["ChebNO2D"]["res_errors"][res]
        e_h = results["Heinn-X 2D v6"]["res_errors"][res]
        best = min(e_f, e_c, e_h)
        w = "FNO" if best == e_f else ("Cheb" if best == e_c else "HX ✓")
        print(
            f"  {str(res)+'×'+str(res):>8}  {e_f:>8.4f}  {e_c:>8.4f}  {e_h:>8.4f}  {w}"
        )
    return results


# ════════════════════════════════════════════════════════════════════
#  결과 플롯
# ════════════════════════════════════════════════════════════════════


def plot_results(r1d, r2d):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#0d0d1a")
        BG, PANEL = "#0d0d1a", "#13132b"
        C = {"FNO": "#00e5ff", "ChebNO": "#69f0ae", "HX": "#ffd54f"}

        for ax in axes:
            ax.set_facecolor(PANEL)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#333355")
            ax.tick_params(colors="#aaaacc", labelsize=9)
            ax.grid(True, color="#1e1e3a", linewidth=0.8, linestyle="--")

        ax = axes[0]
        res1 = [64, 32, 16, 8]
        ax.plot(
            res1,
            [r1d["FNO"]["res_errors"][r] for r in res1],
            "o-",
            color=C["FNO"],
            lw=2,
            ms=7,
            label="FNO",
        )
        ax.plot(
            res1,
            [r1d["ChebNO"]["res_errors"][r] for r in res1],
            "^-",
            color=C["ChebNO"],
            lw=2,
            ms=7,
            label="ChebNO",
        )
        ax.plot(
            res1,
            [r1d["Heinn-X v6"]["res_errors"][r] for r in res1],
            "s-",
            color=C["HX"],
            lw=2,
            ms=7,
            label="Heinn-X v6",
        )
        ax.set_xlabel("Test Resolution N", color="#aaaacc")
        ax.set_ylabel("Relative L² Error ↓", color="#aaaacc")
        ax.set_title(
            "1D Burgers: Resolution Generalization", color="white", fontweight="bold"
        )
        ax.invert_xaxis()
        ax.legend(
            framealpha=0.2, facecolor="#111133", labelcolor="white", edgecolor="#333355"
        )

        ax = axes[1]
        res2 = [32, 16, 8]
        ax.plot(
            res2,
            [r2d["FNO2D"]["res_errors"][r] for r in res2],
            "o-",
            color=C["FNO"],
            lw=2,
            ms=7,
            label="FNO2D",
        )
        ax.plot(
            res2,
            [r2d["ChebNO2D"]["res_errors"][r] for r in res2],
            "^-",
            color=C["ChebNO"],
            lw=2,
            ms=7,
            label="ChebNO2D",
        )
        ax.plot(
            res2,
            [r2d["Heinn-X 2D v6"]["res_errors"][r] for r in res2],
            "s-",
            color=C["HX"],
            lw=2,
            ms=7,
            label="Heinn-X v6",
        )
        ax.set_xlabel("Test Resolution N×N", color="#aaaacc")
        ax.set_ylabel("Relative L² Error ↓", color="#aaaacc")
        ax.set_title(
            "2D Darcy Flow: Resolution Generalization", color="white", fontweight="bold"
        )
        ax.invert_xaxis()
        ax.legend(
            framealpha=0.2, facecolor="#111133", labelcolor="white", edgecolor="#333355"
        )

        fig.suptitle(
            "FNO vs ChebNO vs Heinn-X v6 (S 정규화 + α 스케일)",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out = "benchmark_v6_results.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"\n  결과 그래프 저장됨: {out}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib 없음)")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 63 + "╗")
    print("║  FNO vs ChebNO vs Heinn-X v6                            ║")
    print("║  수정: 해상도 불일치 제거 + S 정규화 + α 스케일          ║")
    print("╚" + "═" * 63 + "╝\n")

    r1d = run_1d_benchmark()
    r2d = run_2d_benchmark()

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print("\n  1D Burgers:")
    for res in [64, 32, 16, 8]:
        e_f = r1d["FNO"]["res_errors"][res]
        e_c = r1d["ChebNO"]["res_errors"][res]
        e_h = r1d["Heinn-X v6"]["res_errors"][res]
        best = min(e_f, e_c, e_h)
        tag = "FNO" if best == e_f else ("Cheb✓" if best == e_c else "HX✓")
        print(f"    res={res:3d}  FNO={e_f:.4f}  Cheb={e_c:.4f}  HX={e_h:.4f}  [{tag}]")

    print("\n  2D Darcy:")
    for res in [32, 16, 8]:
        e_f = r2d["FNO2D"]["res_errors"][res]
        e_c = r2d["ChebNO2D"]["res_errors"][res]
        e_h = r2d["Heinn-X 2D v6"]["res_errors"][res]
        best = min(e_f, e_c, e_h)
        tag = "FNO" if best == e_f else ("Cheb✓" if best == e_c else "HX✓")
        print(f"    {res}×{res}  FNO={e_f:.4f}  Cheb={e_c:.4f}  HX={e_h:.4f}  [{tag}]")

    plot_results(r1d, r2d)
    print("\n  Done ✓")
