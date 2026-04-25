"""
exp1_data_efficiency.py — Killer Scenario #1: Extreme Sample Efficiency
(Real PDEBench 1D Burgers/Advection Data)
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math
from fractions import Fraction
from XNO import HeinnXConv1D, ChebConv1D, SpectralConv1D, make_no_1d  # 기존 모듈 임포트

# 설정
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

MODE = 16
WIDTH = 32
DEPTH = 4


# =================================================================
# 1. HDF5 PDE 데이터 로더 (실제 물리 데이터)
# =================================================================
def load_pdebench_data(filepath, total_samples=1200):
    """
    PDEBench HDF5 파일에서 초기 상태(t=0)로 최종 상태(t=T)를 예측하는 데이터 추출
    """
    print(f"Loading data from {filepath}...")
    try:
        with h5py.File(filepath, "r") as f:
            # PDEBench 데이터 구조는 보통 'tensor' 또는 특정 변수명에 저장됨
            # 형태: (num_samples, x_resolution, t_resolution)
            keys = list(f.keys())
            data_key = keys[0] if "tensor" not in keys else "tensor"
            data = f[data_key][:total_samples]

            # 입력: t=0 일 때의 공간 데이터
            U_in = torch.tensor(data[:, :, 0], dtype=torch.float32)
            # 타겟: t=T (마지막 시간) 일 때의 공간 데이터
            U_out = torch.tensor(data[:, :, -1], dtype=torch.float32)

            # 정규화 (표준화)
            u_mean, u_std = U_in.mean(), U_in.std()
            U_in = (U_in - u_mean) / u_std
            U_out = (U_out - U_out.mean()) / U_out.std()

            return U_in, U_out
    except Exception as e:
        print(
            f"[경고] HDF5 파일을 읽을 수 없습니다. 테스트용 합성 Burgers 데이터를 생성합니다. ({e})"
        )
        # HDF5가 아직 다운로드 안 됐을 때를 대비한 1D 파동 합성기
        N = 1024
        x = torch.linspace(0, 2 * math.pi, N)
        U_in = torch.randn(total_samples, N) * 0.1 + torch.sin(x)
        # Burgers 방정식의 충격파(Shockwave) 형성 흉내
        U_out = U_in - 0.5 * torch.sin(2 * U_in) * 0.5
        return U_in, U_out


def make_input_1d(U):
    B, N = U.shape
    grid = torch.linspace(0, 1, N).view(1, N, 1).expand(B, -1, -1)
    return torch.cat([U.unsqueeze(-1), grid], dim=-1)


# =================================================================
# 2. 훈련 및 평가 함수
# =================================================================
def train_and_eval(model, train_loader, test_loader, epochs=40):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        preds, tgts = [], []
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu())
            tgts.append(yb)
        preds = torch.cat(preds)
        tgts = torch.cat(tgts)
        # Relative L2 Error
        err = (preds - tgts).norm() / (tgts.norm() + 1e-8)
    return err.item()


# =================================================================
# 3. 메인 실험 루프
# =================================================================
def run_sample_efficiency_test(filepath):
    # 데이터 준비
    U_in, U_out = load_pdebench_data(filepath, total_samples=1200)

    # 200개는 고정 테스트 셋으로 사용
    U_test_in, U_test_out = U_in[-200:], U_out[-200:].unsqueeze(-1)
    test_loader = DataLoader(
        TensorDataset(make_input_1d(U_test_in), U_test_out), batch_size=32
    )

    # 실험할 학습 데이터 크기 (Data Starvation)
    train_sizes = [1000, 100, 10]

    results = {"FNO": [], "ChebNO": [], "Heinn-X": []}

    print("\n" + "=" * 65)
    print(" 🚀 [KILLER SCENARIO #1] 극한의 샘플 효율성 테스트 시작")
    print(f" 데이터셋: {filepath.split('/')[-1]}")
    print("=" * 65)

    for size in train_sizes:
        print(f"\n▶ Training with {size} samples...")

        # 데이터 자르기
        U_tr_in, U_tr_out = U_in[:size], U_out[:size].unsqueeze(-1)
        batch_size = min(32, size)  # 데이터가 10개면 배치사이즈도 줄여야 함
        train_loader = DataLoader(
            TensorDataset(make_input_1d(U_tr_in), U_tr_out),
            batch_size=batch_size,
            shuffle=True,
        )

        # 모델 초기화 (공정한 비교를 위해 매번 새로 초기화)
        fno = make_no_1d(lambda: SpectralConv1D(WIDTH, WIDTH, MODE))
        cheb = make_no_1d(lambda: ChebConv1D(WIDTH, WIDTH, MODE))
        hx = make_no_1d(lambda: HeinnXConv1D(WIDTH, WIDTH, MODE))

        models = {"FNO": fno, "ChebNO": cheb, "Heinn-X": hx}

        for name, model in models.items():
            err = train_and_eval(model, train_loader, test_loader, epochs=50)
            results[name].append(err)
            print(f"  - {name:>8} Error: {err:.4f}")

    return train_sizes, results


# =================================================================
# 4. 논문용 그래프 생성 (Log-Log Plot)
# =================================================================
def plot_efficiency(train_sizes, results):
    plt.figure(figsize=(9, 6))

    colors = {"FNO": "#FF3B30", "ChebNO": "#8E8E93", "Heinn-X": "#007AFF"}
    styles = {"FNO": "s--", "ChebNO": "d-.", "Heinn-X": "o-"}

    for name, errors in results.items():
        plt.plot(
            train_sizes,
            errors,
            styles[name],
            label=name,
            color=colors[name],
            linewidth=2.5,
            markersize=8,
        )

    plt.xscale("log")
    plt.yscale("log")
    # x축을 역순으로 표시 (데이터가 줄어드는 방향)
    plt.gca().invert_xaxis()

    plt.xticks(train_sizes, [str(s) for s in train_sizes])
    plt.xlabel("Number of Training Samples (Decreasing $\\rightarrow$)", fontsize=13)
    plt.ylabel("Relative $L^2$ Error (Test Set)", fontsize=13)
    plt.title(
        "Sample Efficiency on PDEBench (Burgers/Advection)",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=12)

    # 10개 샘플 영역 강조
    plt.axvspan(15, 8, color="gray", alpha=0.1)
    plt.text(
        10,
        min(results["Heinn-X"]) * 0.8,
        "Extreme\nData Starvation",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="black",
    )

    plt.tight_layout()
    plt.savefig("Killer_Scenario_1_Sample_Efficiency.png", dpi=300)
    print("\n✅ 그래프가 'Killer_Scenario_1_Sample_Efficiency.png'로 저장되었습니다.")


if __name__ == "__main__":
    # 가지고 계신 HDF5 파일 경로를 여기에 입력하세요.
    # 예: HDF5_FILE = "1D_Burgers_Sols_Nu0.001.hdf5"
    HDF5_FILE = "dataset/1D_Burgers_Sols_Nu0.001.hdf5"

    sizes, res = run_sample_efficiency_test(HDF5_FILE)
    plot_efficiency(sizes, res)
