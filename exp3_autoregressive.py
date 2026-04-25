"""
exp3_autoregressive_100ep.py — Killer Scenario #3: Long-term Stability (100 Epochs & Beta 1.0)
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


def load_temporal_data(filepath, num_samples=200):
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())
        data_key = keys[0] if "tensor" not in keys else "tensor"
        data = f[data_key][:num_samples]
        data = torch.tensor(data, dtype=torch.float32)
        data = (data - data.mean()) / data.std()

        U_in, U_out = [], []
        T_steps = min(data.shape[2], 50)
        for t in range(T_steps - 1):
            U_in.append(data[:, :, t])
            U_out.append(data[:, :, t + 1])

        U_in = torch.cat(U_in, dim=0)
        U_out = torch.cat(U_out, dim=0)
        return U_in, U_out, data


def make_input(U):
    B, N = U.shape
    grid = torch.linspace(0, 1, N).view(1, N, 1).expand(B, -1, -1)
    return torch.cat([U.unsqueeze(-1), grid], dim=-1)


def run_autoregressive_test(filepath):
    print(f"▶ 데이터 로딩 중... ({filepath.split('/')[-1]})")
    U_in, U_out, raw_data = load_temporal_data(filepath, num_samples=200)
    print(f"  - 생성된 학습 페어 수: {len(U_in)} 개")

    train_loader = DataLoader(
        TensorDataset(make_input(U_in), U_out.unsqueeze(-1)),
        batch_size=128,
        shuffle=True,
    )

    models = {
        "FNO": make_no_1d(lambda: SpectralConv1D(32, 32, 16)).to(DEVICE),
        "ChebNO": make_no_1d(lambda: ChebConv1D(32, 32, 16)).to(DEVICE),
        "Heinn-X": make_no_1d(lambda: HeinnXConv1D(32, 32, 16)).to(DEVICE),
    }

    print(f"\n▶ 1-Step 예측 학습 진행 중 (100 Epochs)... (Using Device: {DEVICE})")

    for name, model in models.items():
        print(f"\n[{name}] 모델 훈련 시작...")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()

        # 에포크를 100으로 증가시켜 공정한 비교 수행
        for ep in range(1, 101):
            start_t = time.time()
            total_loss = 0
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = F.mse_loss(model(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward()
                opt.step()
                total_loss += loss.item()

            # 터미널 창 가독성을 위해 1번째와 매 10번째 에포크만 출력
            if ep == 1 or ep % 10 == 0:
                elapsed = time.time() - start_t
                print(
                    f"  - Epoch {ep:3d}/100 | Avg Loss: {total_loss/len(train_loader):.6f} | {elapsed:.1f} sec/ep"
                )

    print("\n▶ 50-Step Autoregressive Rollout 테스트 진행 중... (약 10초 소요)")
    T_max = 50
    test_idx = 0
    u_current = {m: raw_data[test_idx, :, 0].unsqueeze(0) for m in models.keys()}
    gt_trajectory = [raw_data[test_idx, :, 0].cpu()]

    rollout_preds = {m: [] for m in models.keys()}

    with torch.no_grad():
        for t in range(1, T_max + 1):
            gt_trajectory.append(raw_data[test_idx, :, t].cpu())
            for name, model in models.items():
                model.eval()
                inp = make_input(u_current[name]).to(DEVICE)
                pred = model(inp).squeeze(-1).cpu()
                rollout_preds[name].append(pred.squeeze())
                u_current[name] = pred

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    x_grid = torch.linspace(0, 1, raw_data.shape[1])

    # t=0 시점의 초기 파동 위치 추가 (이동 확인용)
    plt.plot(x_grid, gt_trajectory[0], "g:", label="Initial State (T=0)", alpha=0.3)

    plt.plot(
        x_grid, gt_trajectory[T_max], "k--", label="Ground Truth (T=50)", alpha=0.5
    )
    plt.plot(
        x_grid,
        rollout_preds["FNO"][-1],
        color="#FF3B30",
        label="FNO (Blow-up)",
        alpha=0.8,
    )
    plt.plot(
        x_grid, rollout_preds["ChebNO"][-1], color="#8E8E93", label="ChebNO", alpha=0.8
    )
    plt.plot(
        x_grid,
        rollout_preds["Heinn-X"][-1],
        color="#007AFF",
        label="Heinn-X (Stable)",
        linewidth=2.5,
    )

    plt.title("Wave Shape after 50 Autoregressive Steps (beta=1.0)")
    plt.xlabel("Spatial Grid (x)")
    plt.ylabel("u(x, t=50)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exp3_autoregressive_shape_beta1.png", dpi=300)
    print("\n✅ 파형 결과 저장 완료: exp3_autoregressive_shape_beta1.png")


if __name__ == "__main__":
    # ⚠️ 파동이 실제로 이동하는 beta1.0 데이터셋 적용
    FILE_PATH = "dataset/1D_Advection_Sols_beta1.0.hdf5"

    run_autoregressive_test(FILE_PATH)
