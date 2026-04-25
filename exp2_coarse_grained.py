"""
exp2_coarse_grained_optimized.py
"""

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from XNO import HeinnXConv1D, ChebConv1D, SpectralConv1D, make_no_1d
import time

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_pdebench_advection(filepath, num_samples=1000):
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())
        data_key = keys[0] if "tensor" not in keys else "tensor"
        data = f[data_key][:num_samples]
        U_in = torch.tensor(data[:, :, 0], dtype=torch.float32)
        U_out = torch.tensor(data[:, :, -1], dtype=torch.float32)
        U_in = (U_in - U_in.mean()) / U_in.std()
        U_out = (U_out - U_out.mean()) / U_out.std()
        return U_in, U_out


def make_input(U):
    B, N = U.shape
    grid = torch.linspace(0, 1, N).view(1, N, 1).expand(B, -1, -1)
    return torch.cat([U.unsqueeze(-1), grid], dim=-1)


def run_coarse_grained_test(filepath):
    print("▶ 데이터 로딩 중...")
    U_in, U_out = load_pdebench_advection(filepath, 1200)
    U_tr_in, U_tr_out = U_in[:1000], U_out[:1000].unsqueeze(-1)
    U_te_in, U_te_out = U_in[1000:], U_out[1000:].unsqueeze(-1)

    # 배치 사이즈를 64로 늘려 GPU/MPS 연산 효율 극대화
    train_loader = DataLoader(
        TensorDataset(make_input(U_tr_in), U_tr_out), batch_size=64, shuffle=True
    )

    models = {
        "FNO": make_no_1d(lambda: SpectralConv1D(32, 32, 16)).to(DEVICE),
        "ChebNO": make_no_1d(lambda: ChebConv1D(32, 32, 16)).to(DEVICE),
        "Heinn-X": make_no_1d(lambda: HeinnXConv1D(32, 32, 16)).to(DEVICE),
    }

    print(f"\n▶ 고해상도(Master Resolution) 학습 진행 중... (Using Device: {DEVICE})")
    for name, model in models.items():
        print(f"\n[{name}] 모델 훈련 시작...")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()

        # 빠른 검증을 위해 Epoch를 20으로 단축
        for ep in range(1, 101):
            start_t = time.time()
            tot_loss = 0
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = F.mse_loss(model(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward()
                opt.step()
                tot_loss += loss.item()

            # 매 5 에포크마다 진행 상황 출력
            if ep % 5 == 0 or ep == 1:
                elapsed = time.time() - start_t
                print(
                    f"  - Epoch {ep:2d}/100 | Avg Loss: {tot_loss/len(train_loader):.6f} | {elapsed:.2f}s/ep"
                )

    resolutions = [1, 2, 4, 8, 16]
    results = {m: [] for m in models.keys()}

    print("\n▶ 제로샷 평가 결과 (Resolution Downsampling)")
    for step in resolutions:
        sub_in = U_te_in[:, ::step]
        sub_out = U_te_out[:, ::step, :]
        N_pts = sub_in.shape[1]
        test_loader = DataLoader(
            TensorDataset(make_input(sub_in), sub_out), batch_size=64
        )

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                preds = torch.cat([model(xb.to(DEVICE)).cpu() for xb, _ in test_loader])
                tgts = torch.cat([yb for _, yb in test_loader])
                err = ((preds - tgts).norm() / (tgts.norm() + 1e-8)).item()
                results[name].append(err)
        print(
            f"  - Points: {N_pts:3d} | FNO: {results['FNO'][-1]:.4f} | ChebNO: {results['ChebNO'][-1]:.4f} | HX: {results['Heinn-X'][-1]:.4f}"
        )

    plt.figure(figsize=(8, 5))
    x_labels = [str(U_te_in.shape[1] // s) for s in resolutions]
    for m, c, s in zip(
        models.keys(), ["#FF3B30", "#8E8E93", "#007AFF"], ["s--", "d-.", "o-"]
    ):
        plt.plot(x_labels, results[m], s, label=m, color=c, lw=2)
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Sensor Resolution (Grid Points)")
    plt.ylabel("Relative L2 Error")
    plt.title("Zero-Shot Coarse-Grained Simulation (Advection)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exp2_coarse_grained.png", dpi=300)
    print("\n✅ 그래프 저장 완료: exp2_coarse_grained.png")


if __name__ == "__main__":
    FILE_PATH = "dataset/1D_Advection_Sols_beta1.0.hdf5"
    run_coarse_grained_test(FILE_PATH)
