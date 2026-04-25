import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from torch.utils.data import Dataset, DataLoader

# 현원님이 지정하신 임포트 방식 적용
from XNO import HeinnXConv1D, ChebConv1D, SpectralConv1D, make_no_1d

# 전역 설정
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FILE_PATH = "dataset/1D_Burgers_Sols_Nu0.001.hdf5"
WIDTH = 64
MODES = 32  # S-행렬 차원 및 Spectral 모드 수
EPOCHS = 150
BATCH_SIZE = 32

print(f"🚀 M4 Pro Research Mode | Device: {DEVICE}")

# =================================================================
# 1. DATASET: PDEBench Burgers 1D (3D/4D 대응 수정본)
# =================================================================


class BurgersDataset(Dataset):
    def __init__(self, path, n_samples=1000, t_idx=150, step=1):
        with h5py.File(path, "r") as f:
            u_raw = f["tensor"]
            # PDEBench 1D 데이터의 3D([N,T,X]) vs 4D([N,T,X,V]) 대응
            if len(u_raw.shape) == 4:
                u = torch.tensor(u_raw[:n_samples, :, ::step, 0], dtype=torch.float32)
            else:
                u = torch.tensor(u_raw[:n_samples, :, ::step], dtype=torch.float32)

            self.x = torch.tensor(f["x-coordinate"][::step], dtype=torch.float32)

        self.input = u[:, 0, :]  # t=0 (Initial Condition)
        self.target = u[:, t_idx, :]  # t=150 (Shock point)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        return self.input[i].unsqueeze(0), self.target[i].unsqueeze(0)


# =================================================================
# 2. UTILS: Metric & Training
# =================================================================


def rel_l2(pred, target):
    return (torch.norm(pred - target) / (torch.norm(target) + 1e-8)).item()


def train_model(name, model, loader):
    print(f"\n── {name} Training Start ────────────────")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    model.train()

    start_time = time.time()
    for ep in range(1, EPOCHS + 1):
        tot_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()
        if ep % 10 == 0:
            print(f"  Ep {ep:2d} | Avg Loss: {tot_loss/len(loader):.6f}")

    duration = time.time() - start_time
    print(f"✅ {name} Training Done. ({duration:.1f}s)")
    return model


# =================================================================
# 3. MAIN BENCHMARK
# =================================================================


def run_benchmark():
    # 1. 데이터 로드 (학습용: 1024 points)
    print(f"1. Loading Burgers Dataset: {FILE_PATH}")
    full_ds = BurgersDataset(FILE_PATH, n_samples=1200, t_idx=150)
    train_size = 1000
    train_ds, test_ds_full = torch.utils.data.random_split(
        full_ds, [train_size, len(full_ds) - train_size]
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 모델 빌드
    models = {
        "FNO": make_no_1d(lambda: SpectralConv1D(WIDTH, WIDTH, MODES), width=WIDTH),
        "ChebNO": make_no_1d(lambda: ChebConv1D(WIDTH, WIDTH, MODES), width=WIDTH),
        "Heinn-X": make_no_1d(lambda: HeinnXConv1D(WIDTH, WIDTH, MODES), width=WIDTH),
    }

    # 3. 모델 학습
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_model(name, model.to(DEVICE), train_loader)

    # 4. EXP 1: Zero-Shot Resolution Transfer (1024 -> 512, 256, 128)
    print("\n" + "=" * 50)
    print(f"{'Resolution':>12} | {'FNO':>8} | {'ChebNO':>8} | {'Heinn-X':>8}")
    print("-" * 50)

    res_list = [1024, 512, 256, 128]
    summary_results = {res: {} for res in res_list}

    for res in res_list:
        step = 1024 // res
        # 테스트용 데이터셋 (해당 해상도로 서브샘플링)
        eval_ds = BurgersDataset(FILE_PATH, n_samples=1200, t_idx=150, step=step)
        # 앞에서 쓴 test_ds_full의 인덱스 유지 (공정한 비교)
        _, eval_test_ds = torch.utils.data.random_split(
            eval_ds, [train_size, len(eval_ds) - train_size]
        )
        eval_loader = DataLoader(eval_test_ds, batch_size=1)

        row_errors = []
        for name, model in trained_models.items():
            model.eval()
            errors = []
            with torch.no_grad():
                for tx, ty in eval_loader:
                    tp = model(tx.to(DEVICE)).cpu()
                    errors.append(rel_l2(tp, ty))
            summary_results[res][name] = np.mean(errors)

        print(
            f"{res:>12d} | {summary_results[res]['FNO']:>8.4f} | "
            f"{summary_results[res]['ChebNO']:>8.4f} | {summary_results[res]['Heinn-X']:>8.4f}"
        )

    # 5. 시각화 (Res=128에서의 Shock Capture)
    print("\n🎨 Saving Visualization for Resolution 128...")
    res_viz = 128
    step_viz = 1024 // res_viz
    viz_ds = BurgersDataset(FILE_PATH, n_samples=10, t_idx=150, step=step_viz)

    tx, ty = viz_ds[0]
    plt.figure(figsize=(12, 6))
    plt.plot(viz_ds.x, ty.squeeze(), "k", label="Ground Truth", linewidth=3, alpha=0.6)

    colors = {"FNO": "red", "ChebNO": "gray", "Heinn-X": "blue"}
    for name, model in trained_models.items():
        tp = model(tx.unsqueeze(0).to(DEVICE)).detach().cpu().squeeze().numpy()
        plt.plot(
            viz_ds.x, tp, label=f"{name} Prediction", color=colors[name], linestyle="--"
        )

    plt.title(
        f"1D Burgers Shock Capture (Zero-shot Transfer to {res_viz} pts)", fontsize=14
    )
    plt.xlabel("Space (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Burgers_Final_Comparison.png", dpi=300)
    print("✅ Visualization saved as 'Burgers_Final_Comparison.png'")
    plt.show()


if __name__ == "__main__":
    run_benchmark()
