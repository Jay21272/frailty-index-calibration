# -*- coding: utf-8 -*-
"""HRS(RAND) v3：修正 ADL 变量名(R1=ADLW, R2+=ADL5A)，补 IADL5A/移动MOBILA/认知SER7。"""
import pyreadstat
import pandas as pd
import numpy as np

PATH = r"G:\hrs data\randhrs1992_2022v1.sas7bdat"
_, meta = pyreadstat.read_sas7bdat(PATH, metadataonly=True)
avail = set(meta.column_names)
WAVES = list(range(1, 16))

want = ['HHIDPN', 'RABYEAR', 'RAGENDER', 'RAEDYRS', 'RADYEAR', 'RADMONTH']
for w in WAVES:
    want += ['INW%d' % w, 'R%dAGEY_B' % w, 'R%dMSTAT' % w,
             'R%dADL5A' % w, 'R%dIADL5A' % w, 'R%dMOBILA' % w, 'R%dSER7' % w,
             'R%dDEPRES' % w, 'R%dCESD' % w, 'R%dBMI' % w, 'R%dSHLT' % w,
             'R%dHIBPE' % w, 'R%dDIABE' % w, 'R%dCANCRE' % w, 'R%dLUNGE' % w,
             'R%dHEARTE' % w, 'R%dSTROKE' % w, 'R%dARTHRE' % w,
             'H%dATOTF' % w, 'H%dATOTB' % w, 'H%dITOT' % w, 'H%dCHILD' % w]
want += ['R1ADLW', 'R1MOBILW', 'R1DEPREX']
usecols = [c for c in want if c in avail]

df, _ = pyreadstat.read_sas7bdat(PATH, usecols=usecols)
df.columns = [c.lower() for c in df.columns]

YEAR = {w: 1990 + 2 * w for w in WAVES}


def num(col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan


age_at_w = {w: num('r%dagey_b' % w) for w in WAVES}
inw = {w: num('inw%d' % w) for w in WAVES}
base_wave = np.full(len(df), np.nan)
for w in WAVES:
    eligible = (inw[w] == 1) & (age_at_w[w] >= 60)
    base_wave = np.where(np.isnan(base_wave) & eligible, w, base_wave)

D = pd.DataFrame({"pid": df["hhidpn"].astype(str), "base_wave": base_wave})
D["base_year"] = D["base_wave"].map(YEAR)


def gather_alt(colmap):
    """按波次用不同列名抽基线值。"""
    out = np.full(len(df), np.nan)
    for w, c in colmap.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            out = np.where(base_wave == w, v, out)
    return out


def gather(pattern):
    out = np.full(len(df), np.nan)
    for w in WAVES:
        c = pattern % w
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            out = np.where(base_wave == w, v, out)
    return out


# ADL: w1 -> R1ADLW(Wallace), w>=2 -> R%dADL5A
adl_map = {1: 'r1adlw'}
adl_map.update({w: 'r%dadl5a' % w for w in WAVES if w >= 2})
D["adl_count"] = gather_alt(adl_map)
# 移动: w1 -> R1MOBILW, w>=2 -> R%dMOBILA
mob_map = {1: 'r1mobilw'}
mob_map.update({w: 'r%dmobila' % w for w in WAVES if w >= 2})
D["mobil_count"] = gather_alt(mob_map)
# IADL: w>=2 -> R%dIADL5A (w1 无)
D["iadl_count"] = gather_alt({w: 'r%diadl5a' % w for w in WAVES if w >= 2})
# 认知 SER7: w>=2 -> R%dSER7 (w1 无)
D["ser7"] = gather_alt({w: 'r%dser7' % w for w in WAVES if w >= 2})

D["age"] = gather('r%dagey_b')
D["gender"] = num("ragender")
D["educ"] = num("raedyrs")
D["marital"] = gather('r%dmstat')
# 抑郁: w1 -> R1DEPREX, w>=2 -> R%dDEPRES(单条 Felt depressed)
dep_map = {1: 'r1deprex'}
dep_map.update({w: 'r%ddepres' % w for w in WAVES if w >= 2})
D["depress"] = gather_alt(dep_map)
# 完整 CESD 得分: w>=2 -> R%dCESD (w1 无)
D["cesd"] = gather_alt({w: 'r%dcesd' % w for w in WAVES if w >= 2})
D["bmi"] = gather('r%dbmi')
D["self_rated_health"] = gather('r%dshlt')
for name, h in {"hibp": "hibpe", "diab": "diabe", "cancr": "cancre", "lung": "lunge",
                "heart": "hearte", "strok": "stroke", "arth": "arthre"}.items():
    D[name] = gather('r%d' + h)
D["wealth_fin"] = gather('h%datotf')
D["wealth_total"] = gather('h%datotb')
D["income_total"] = gather('h%ditot')
D["n_children"] = gather('h%dchild')

D["death_year"] = num("radyear")
D["death_month"] = num("radmonth")
D["died"] = D["death_year"].notna()
D["death_5y"] = D["died"] & (D["death_year"] <= D["base_year"] + 5)
D["death_10y"] = D["died"] & (D["death_year"] <= D["base_year"] + 10)

D = D[D["base_wave"].notna()].reset_index(drop=True)

print(f"60+ 基线样本 {len(D)} 人")
print(f"base_wave 分布: {dict(D['base_wave'].value_counts().sort_index())}")
print()
for c in ["adl_count", "iadl_count", "mobil_count", "ser7", "depress", "cesd", "bmi"]:
    vals = D[c].dropna()
    print(f"{c}: 非空 {vals.shape[0]}/{len(D)} ({vals.shape[0]/len(D)*100:.1f}%), "
          f"取值 {sorted(vals.unique())[:15]}")

out = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\hrs_60plus_baseline_v3.csv"
D.to_csv(out, index=False)
print(f"\n已保存 {out}（{D.shape[1]} 列，{len(D)} 行）")
