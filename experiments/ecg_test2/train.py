# train.py
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 你的模型文件名是 simple_cnn.py
from simple_cnn import SimpleCNN1D

def load_cfg(path="guitu_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_split(processed_dir: str, split: str):
    d = np.load(os.path.join(processed_dir, f"{split}.npz"), allow_pickle=True)
    X = d["X"].astype(np.float32)     # [N, 12, T]
    y = d["y"] if "y" in d else None
    return X, y

def _make_loader(X, y, batch_size, shuffle=True):
    X_t = torch.from_numpy(X)             # [N, 12, T]
    if y is None:
        ds = TensorDataset(X_t)
    else:
        y_t = torch.from_numpy(y.astype(np.int64))
        ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def run_train():
    cfg = load_cfg()
    pdir = cfg["paths"]["processed_dir"]
    _ensure_dir(pdir)

    Xtr, ytr = _load_split(pdir, "train")
    Xva, yva = _load_split(pdir, "val")

    if ytr is None:
        raise RuntimeError("未发现标签 y。请提供 data/labels.csv 并先运行 preprocess。")

    C = Xtr.shape[1]                  # 一般为 12
    num_classes = int(cfg["labels"]["num_classes"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN1D(num_classes=num_classes, in_channels=C).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    crit = nn.CrossEntropyLoss()

    dl_tr = _make_loader(Xtr, ytr, cfg["train"]["batch_size"], shuffle=True)
    dl_va = _make_loader(Xva, yva, cfg["train"]["batch_size"], shuffle=False)

    best_va, best_state = -1.0, None
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(len(dl_tr.dataset), 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        va_acc = correct / max(total, 1)
        print(f"[train] epoch {epoch+1}: train_loss={tr_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_va:
            best_va = va_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"type": "clf", "state": model.state_dict(), "num_classes": num_classes},
               os.path.join(pdir, "model.pt"))

    with open(os.path.join(pdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": float(best_va)}, f, ensure_ascii=False, indent=2)

    print(f"[train] 最佳模型已保存到 {os.path.join(pdir, 'model.pt')}")

if __name__ == "__main__":
    run_train()