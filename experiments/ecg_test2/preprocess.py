# preprocess.py — robust, logged, must-save version
from __future__ import annotations
import sys, json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# 可选依赖
try:
    import yaml  # 用于读取本地 config.yaml（如果没有 ecg_convert.load_cfg）
except Exception:
    yaml = None

try:
    import pandas as pd  # 用于读取 labels.csv（若存在）
except Exception:
    pd = None

# 优先沿用转换阶段已有的 load_cfg（若存在）
try:
    from ecg_convert import load_cfg as _load_cfg  # type: ignore
except Exception:
    _load_cfg = None


# =============== 小工具 ===============

def _log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)

def _standardize_per_lead(x: np.ndarray) -> np.ndarray:
    """
    逐导联标准化： (x - mean) / std
    x: (C, T) 或 (T,)；输出 (C, T)
    """
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(np.float32, copy=True)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x - mu) / sd

def _pack_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    将任意长度的 (C, T) 或 (T,) 裁剪/补零为 (C, target_len)
    """
    if arr.ndim == 1:
        arr = arr[None, :]
    C, T = arr.shape[0], arr.shape[-1]
    out = np.zeros((C, target_len), dtype=np.float32)
    L = min(T, target_len)
    out[:, :L] = arr[:, :L]
    return out

def _read_labels_csv(path: Path) -> Optional[Dict[str, int]]:
    """
    读取 labels.csv（两列：filename,label）。filename 为不带扩展名的基名。
    若不存在或无法读取，返回 None。
    """
    if not path.exists():
        _log(f"INFO  labels.csv not found at {path} → NO-LABEL mode")
        return None
    if pd is None:
        raise RuntimeError("pandas 未安装，无法读取 labels.csv（请安装 pandas 或移除标签模式）")
    df = pd.read_csv(path)
    need = {"filename", "label"}
    if not need.issubset(df.columns):
        raise KeyError(f"labels.csv 必须包含列 {need}，实际为 {df.columns.tolist()}")
    m = {str(r.filename).strip(): int(r.label) for r in df.itertuples(index=False)}
    _log(f"OK    labels loaded = {len(m)}")
    return m

def _load_config() -> dict:
    """
    配置优先级：
    1) ecg_convert.load_cfg()（若存在）
    2) 本地同目录 config.yaml（若存在）
    3) 内置默认值
    """
    # 1) 复用转换阶段配置
    if _load_cfg is not None:
        try:
            cfg = _load_cfg() or {}
            _log("OK    using ecg_convert.load_cfg()")
            return cfg
        except Exception as e:
            _log(f"WARN  ecg_convert.load_cfg() failed: {e}")

    # 2) 读本地 config.yaml
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if cfg_path.exists() and yaml is not None:
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            _log(f"OK    loaded {cfg_path.name}")
            return cfg
        except Exception as e:
            _log(f"WARN  failed to read {cfg_path.name}: {e}")

    # 3) 默认值
    _log("INFO  using built-in defaults")
    return {
        "paths": {
            "interim_npy_dir": "data/interim_npy",
            "processed_dir": "data/processed",
            "labels_csv": "data/labels.csv",
        },
        "parse": {
            "sampling_rate": 500,          # Hz
            "fixed_length_sec": 10,        # s → 5000 点
        },
        "preprocess": {
            "normalize": True
        }
    }

def _infer_npz_key(p: Path) -> Tuple[str, Tuple[str, ...]]:
    """
    推断 .npz 中波形数组的键名。优先 'X' → 'signals' → 'data' → 第一个键。
    返回 (选中的键, 全部键列表)
    """
    with np.load(p, allow_pickle=True) as d0:
        keys = tuple(d0.keys())
        if "X" in keys:
            return "X", keys
        if "signals" in keys:
            return "signals", keys
        if "data" in keys:
            return "data", keys
        return keys[0], keys


# =============== 主流程 ===============

def run_preprocess() -> None:
    _log("START")
    cfg = _load_config()

    # 路径与参数（带默认值兜底）
    paths = cfg.get("paths", {})
    parse = cfg.get("parse", {})
    in_dir  = Path(paths.get("interim_npy_dir", "data/interim_npy"))
    out_dir = Path(paths.get("processed_dir",   "data/processed"))
    labels_csv = Path("data/labels.csv")

    sr   = int(parse.get("sampling_rate", 500))
    secs = int(parse.get("fixed_length_sec", 10))
    target_len = int(sr * secs)
    do_norm = bool(cfg.get("preprocess", {}).get("normalize", True))

    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"in={in_dir}")
    _log(f"out={out_dir}")
    _log(f"labels={labels_csv}")
    _log(f"target_len={target_len}  normalize={do_norm}")

    # 列出中间产物
    files = sorted(in_dir.glob("*.npz"))
    _log(f"found npz = {len(files)}")
    if not files:
        _log(f"ERROR no .npz under {in_dir}. Did you run --convert ?")
        sys.exit(2)

    # 推断键名
    use_key, all_keys = _infer_npz_key(files[0])
    _log(f"npz keys0={list(all_keys)} → use key='{use_key}'")

    # 读标签（可无）
    label_map = _read_labels_csv(labels_csv)

    # 主循环
    X_list, y_list, names = [], [], []
    for i, f in enumerate(files, 1):
        try:
            with np.load(f, allow_pickle=True) as d:
                arr = d[use_key] if use_key in d else d[list(d.keys())[0]]
            arr = _pack_length(arr, target_len)   # (C, target_len)
            if do_norm:
                arr = _standardize_per_lead(arr)  # (C, target_len)
            X_list.append(arr.astype(np.float32))
            names.append(f.stem)
            if label_map is not None:
                y_list.append(int(label_map.get(f.stem, -1)))
        except Exception as e:
            _log(f"WARN  skip {f.name} due to: {e}")
            continue

        if i % 100 == 0 or i == len(files):
            _log(f"processed {i}/{len(files)}")

    if not X_list:
        _log("ERROR nothing processed — check npz content / key names.")
        sys.exit(3)

    # 堆叠与落盘
    X = np.stack(X_list, axis=0)  # [N, C, T]
    np.save(out_dir / "X.npy", X)
    _log(f"OK    saved X.npy shape={X.shape}")

    meta = {"names": names, "target_len": target_len}
    if y_list:
        y = np.array(y_list, dtype=np.int64)
        np.save(out_dir / "y.npy", y)
        meta["label_set"] = sorted(set(y.tolist()))
        np.savez(out_dir / "dataset.npz", X=X, y=y, names=np.array(names))
        _log(f"OK    saved y.npy shape={y.shape}, labels={meta['label_set']}")
    else:
        np.savez(out_dir / "dataset.npz", X=X, names=np.array(names))
        _log("OK    saved dataset.npz (no labels)")

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _log("DONE")


if __name__ == "__main__":
    # 允许直接：python -u -X dev preprocess.py
    run_preprocess()