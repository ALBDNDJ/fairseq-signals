# eval.py  —— 多导联可视化增强版
import os
import json
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# =========================
# 可调参数（不改代码也能跑）
# =========================
# 默认只画导联 II（节省图片）
PLOT_LEAD_NAMES = ["II"]              # 覆盖为 ["II","V1","V5"] 会多画三条叠图
# 是否额外输出 12 导联 3×4 网格图
PLOT_FULL_GRID = False                # 如需强制保存全导联图，改为 True
# 每类图最大样本数
MAX_SAMPLES_OVERLAY = 5               # 单导联/三导联叠图最多保存的样本数
MAX_SAMPLES_GRID = 2                  # 12导联网格图最多保存的样本数

# =========================
# 基本工具
# =========================
def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_split(processed_dir: str, split: str):
    d = np.load(os.path.join(processed_dir, f"{split}.npz"), allow_pickle=True)
    X = d["X"].astype(np.float32)              # [N, C(=12), T]
    y = d["y"] if "y" in d else None
    names = d["names"] if "names" in d else None
    return X, y, names

def _lead_indices(lead_names_want, lead_order):
    """返回想要的导联在 lead_order 中的索引（缺失则跳过）"""
    idx = []
    for name in lead_names_want:
        if name in lead_order:
            idx.append(lead_order.index(name))
    return idx

def _plot_overlay_single(ax, x, title):
    ax.plot(x)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("amp")

def _plot_overlay_pair(ax, x, y, title):
    ax.plot(x, label="orig")
    ax.plot(y, label="recon", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("amp")
    ax.legend()

def _plot_grid_12leads(sample_12xT, lead_order, suptitle, save_path):
    """3×4 小 multiples：12 导联波形"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.ravel()
    for i in range(12):
        axes[i].plot(sample_12xT[i])
        axes[i].set_title(lead_order[i])
        axes[i].tick_params(labelleft=False, labelbottom=False, length=0)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# =========================
# 主流程
# =========================
def run_eval():
    cfg = load_cfg()
    pdir = cfg["paths"]["processed_dir"]
    plots_dir = os.path.join(pdir, "plots")
    _ensure_dir(plots_dir)

    # 读取模型
    ckpt_path = os.path.join(pdir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型文件：{ckpt_path}，请先运行 train。")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mtype = ckpt.get("type", "clf")

    # 读取测试集
    Xte, yte, names = _load_split(pdir, "test")     # Xte:[N,12,T]
    N, C, T = Xte.shape
    lead_order = cfg["parse"]["lead_order"]
    want_idx = _lead_indices(PLOT_LEAD_NAMES, lead_order)  # 例如 [1] 代表 II

    # 分类器：评估 + 可视化
    if mtype == "clf":
        num_classes = ckpt["num_classes"]
        # 构建模型（与训练时保持一致）
        from models.simple_cnn import SimpleCNN1D
        model = SimpleCNN1D(num_classes=num_classes, in_channels=C)
        model.load_state_dict(ckpt["state"])
        model.eval()

        with torch.no_grad():
            logits = model(torch.from_numpy(Xte))
            pred = logits.argmax(dim=1).numpy()

        if yte is not None:
            acc = accuracy_score(yte, pred)
            print(f"[eval][clf] test_acc={acc:.4f}")
            # 混淆矩阵
            cm = confusion_matrix(yte, pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.title("Confusion Matrix")
            cm_path = os.path.join(plots_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=200)
            plt.close()

            # 保存一份指标
            with open(os.path.join(pdir, "metrics_eval.json"), "w", encoding="utf-8") as f:
                json.dump({"test_acc": float(acc)}, f, ensure_ascii=False, indent=2)

        # 可视化：默认画导联 II（或你在上面设定的导联）
        max_n = min(MAX_SAMPLES_OVERLAY, N)
        for i in range(max_n):
            for j, li in enumerate(want_idx):
                title = f"{names[i] if names is not None else i} | Lead {lead_order[li]} | pred={pred[i]}"
                if yte is not None:
                    title += f" | gt={yte[i]}"
                fig, ax = plt.subplots(figsize=(8, 3))
                _plot_overlay_single(ax, Xte[i, li], title)
                out = os.path.join(plots_dir, f"clf_{i}_lead-{lead_order[li]}.png")
                fig.savefig(out, dpi=200)
                plt.close(fig)

        # 可选：12 导联网格图
        if PLOT_FULL_GRID:
            max_g = min(MAX_SAMPLES_GRID, N)
            for i in range(max_g):
                suptitle = f"{names[i] if names is not None else i} | pred={pred[i]}"
                if yte is not None:
                    suptitle += f" | gt={yte[i]}"
                out = os.path.join(plots_dir, f"clf_{i}_grid12.png")
                _plot_grid_12leads(Xte[i], lead_order, suptitle, out)

    # 自编码器：重构可视化
    else:
        from models.autoencoder import AE1D
        model = AE1D(latent_dim=64)
        model.load_state_dict(ckpt["state"])
        model.eval()

        with torch.no_grad():
            Xte_t = torch.from_numpy(Xte)          # [N,12,T]
            recon = model(Xte_t)                    # [N,12,T'] 可能略偏
            if recon.size(-1) != Xte_t.size(-1):
                recon = recon[..., : Xte_t.size(-1)]
            recon = recon.numpy()

        # 叠图：默认 II（或设成 ["II","V1","V5"]）
        max_n = min(MAX_SAMPLES_OVERLAY, N)
        for i in range(max_n):
            for li in want_idx:
                fig, ax = plt.subplots(figsize=(8, 3))
                title = f"{names[i] if names is not None else i} | Lead {lead_order[li]}"
                _plot_overlay_pair(ax, Xte[i, li], recon[i, li], title)
                out = os.path.join(plots_dir, f"ae_{i}_lead-{lead_order[li]}_overlay.png")
                fig.savefig(out, dpi=200)
                plt.close(fig)

        # 可选：12 导联网格（原始与重构各一张）
        if PLOT_FULL_GRID:
            max_g = min(MAX_SAMPLES_GRID, N)
            for i in range(max_g):
                out1 = os.path.join(plots_dir, f"ae_{i}_grid12_orig.png")
                _plot_grid_12leads(Xte[i], lead_order, f"{names[i] if names is not None else i} | ORIGINAL", out1)
                out2 = os.path.join(plots_dir, f"ae_{i}_grid12_recon.png")
                _plot_grid_12leads(recon[i], lead_order, f"{names[i] if names is not None else i} | RECON", out2)

        # 也保存一份重构误差的统计（MSE）
        mse = ((Xte - recon) ** 2).mean(axis=(1, 2))  # 每个样本的平均 MSE
        with open(os.path.join(pdir, "metrics_eval.json"), "w", encoding="utf-8") as f:
            json.dump({"recon_mse_mean": float(mse.mean()),
                       "recon_mse_std": float(mse.std())}, f, ensure_ascii=False, indent=2)

    # 标记文件
    with open(os.path.join(pdir, "eval_done.txt"), "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[eval] 图像与指标输出：{plots_dir}")


if __name__ == "__main__":
    run_eval()