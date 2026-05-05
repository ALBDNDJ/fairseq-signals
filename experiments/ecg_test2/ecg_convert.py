# convert.py (Guitu-specific)
import os, glob, json
import numpy as np
import xml.etree.ElementTree as ET
import yaml

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _read_text(node, tag, default=None):
    x = node.findtext(tag)
    return x if x is not None else default

def _to_int(x, default=0):
    try: return int(x)
    except: return default

def _to_float(x, default=0.0):
    try: return float(x)
    except: return default

def _scale_units(raw_values: np.ndarray, v_per_bit: float, units: str, target: str):
    """
    raw_values: int16 -> physical units.
    VoltageValuePerBit=4.88, VoltageUnits='uV'  (归途)
    target: "uV" | "mV"
    """
    # int → uV
    uv = raw_values.astype(np.float32) * float(v_per_bit)
    if target.lower() == "uv":
        return uv
    elif target.lower() == "mv":
        return uv / 1000.0
    else:
        return uv  # fallback

def _parse_guitu_xml(xml_path: str, lead_order):
    """
    返回:
      wave: np.ndarray, shape (12, 5000)  # 缺失导联用0填充，多余则裁剪
      meta: dict  # patient & measurements 元数据
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1) RhythmWaveform
    rw = root.find(".//RhythmWaveform")
    if rw is None:
        raise ValueError(f"No RhythmWaveform in {xml_path}")

    n_leads = _to_int(_read_text(rw, "NumberofLeads"), 12)
    fs = _to_int(_read_text(rw, "SampleRate"), 500)

    # 2) 逐导联读取
    # 预置字典：每个导联名 → 数组
    lead_map = {}

    for ld in rw.findall(".//LeadData"):
        lead_id = _read_text(ld, "LeadID", "").strip()
        if not lead_id:
            continue

        # 点数、标尺、单位、样本位宽
        cnt = _to_int(_read_text(ld, "LeadSampleCountTotal"), 0)
        v_per_bit = _to_float(_read_text(ld, "VoltageValuePerBit"), 1.0)
        units = (_read_text(ld, "VoltageUnits", "uV") or "uV").strip()
        # 原始采样：空格分隔的 int16
        wf_text = _read_text(ld, "WaveFormData", "") or ""
        if not wf_text:
            continue

        # 解析整型序列
        try:
            ints = np.fromstring(wf_text, sep=" ", dtype=np.int16)  # 更快更准
        except Exception:
            # 兜底：慢一点的 split 解析
            ints = np.array([int(t) for t in wf_text.split()], dtype=np.int16)

        # 如果 XML 声称 cnt=5000，核对一下
        if cnt > 0 and ints.size != cnt:
            # 容忍轻微不一致：以实际为准，并在后续裁剪/填充
            pass

        # 量化到物理电压，默认输出 mV（配置可改）
        target_unit = "mV"
        scaled = _scale_units(ints, v_per_bit, units, target_unit)

        lead_map[lead_id] = scaled

    # 3) 组装为固定顺序的 12 导联 × T
    T = 5000  # 10s@500Hz（文档默认）
    waves = []
    for name in lead_order:
        arr = lead_map.get(name, None)
        if arr is None or arr.size == 0:
            waves.append(np.zeros(T, dtype=np.float32))
        else:
            if arr.size > T:
                waves.append(arr[:T].astype(np.float32))
            elif arr.size < T:
                pad = np.zeros(T, dtype=np.float32); pad[:arr.size] = arr.astype(np.float32)
                waves.append(pad)
            else:
                waves.append(arr.astype(np.float32))
    wave_12xT = np.stack(waves, axis=0)  # (12, 5000)

    # 4) 元数据：Patient & ECGMeasurements（可选）
    meta = {}
    pt = root.find(".//Patient")
    if pt is not None:
        meta["patient"] = {
            "PatientID": _read_text(pt, "PatientID"),
            "PatientLastName": _read_text(pt, "PatientLastName"),
            "Gender": _read_text(pt, "Gender"),
            "DateofBirth": _read_text(pt, "DateofBirth"),
            "PatientAge": _read_text(pt, "PatientAge"),
            "AgeUnits": _read_text(pt, "AgeUnits"),
        }

    meas = root.find(".//ECGMeasurements")
    if meas is not None:
        fields = ["VentricularRate","AtrialRate","PRInterval","QRSDuration",
                  "QTInterval","QTc","PAxis","RAxis","TAxis","QRSNum","RRInterval"]
        meta["measurements"] = {k: _read_text(meas, k) for k in fields}

    meta["fs"] = fs
    meta["n_leads"] = n_leads
    meta["lead_order"] = list(lead_order)

    return wave_12xT, meta

def run_convert():
    cfg = load_cfg()
    raw_dir = cfg["paths"]["raw_xml_dir"]
    out_dir = cfg["paths"]["interim_npy_dir"]
    lead_order = cfg["parse"]["lead_order"]
    _ensure_dir(out_dir)

    xml_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.xml"), recursive=True))
    if not xml_files:
        print(f"[convert] 未发现XML：{raw_dir}")
        return

    ok, bad = 0, 0
    for xp in xml_files:
        try:
            arr12, meta = _parse_guitu_xml(xp, lead_order)
            print(meta)
            base = os.path.splitext(os.path.basename(xp))[0]
            # 保存为 .npz：同时带 wave 与 meta
            np.savez_compressed(os.path.join(out_dir, base + ".npz"), X=arr12, meta=json.dumps(meta, ensure_ascii=False))
            ok += 1
        except Exception as e:
            print(f"[convert][ERR] {xp}: {e}")
            bad += 1

    print(f"[convert] 完成。成功 {ok}，失败 {bad} → {out_dir}")