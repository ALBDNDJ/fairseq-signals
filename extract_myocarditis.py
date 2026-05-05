import pandas as pd
import ast

# =========================
# 1. 路径与字段
# =========================
CSV_PATH = "records_w_diag_icd10.csv"
OUT_PATH = "mimic_myocarditis_records.csv"
ICD_COL = "all_diag_all"

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print("Total rows:", len(df))

# =========================
# 2. 解析 ICD 列（字符串 -> list）
# =========================
def parse_icd(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

print("Parsing ICD lists...")
df["icd_list"] = df[ICD_COL].apply(parse_icd)

# =========================
# 3. 心肌炎 ICD 规则（✔ 已适配 MIMIC 实际格式）
# =========================
# I40*  : Acute myo:
    return (
        any(code.startswith(MYOCARDITIS_PREFIX) for code in icds)
        or any(code == "I514" for code in icds)
    )

# =========================
# 4. 打标 & 统计
# =========================
print("Building myocarditis label...")
df["MYOCARDITIS"] = df["icd_list"].apply(is_myocarditis)

print("Myocarditis counts:")
print(df["MYOCARDITIS"].value_counts())

df_myo = df[df["MYOCARDITIS"]].copy()
print("Extracted myocarditis records:", len(df_myo))

# =========================
# 5. 导出
# =========================
df_myo.to_csv(OUT_PATH, index=False)
print("Saved to:", OUT_PATH)
print("Done.")
