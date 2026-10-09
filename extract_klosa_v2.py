# -*- coding: utf-8 -*-
"""KLoSA v2：补 BMI(w01bmi) + IADL9项(C209-217) + 握力(C507-510) + 认知K-MMSE原始值(C401-412)。"""
import pandas as pd
import numpy as np

MAIN = r"G:\kLoSA\e_imp_w01.dta"
EXIT = r"G:\kLoSA\Exit09_e.dta"

class _LatinReader(pd.io.stata.StataReader):
    def _set_encoding(self, *a, **k):
        self._encoding = "latin-1"

def read_latin(path):
    r = _LatinReader(path, convert_categoricals=False)
    return r.read()

df = read_latin(MAIN)
if "vimputation_" in df.columns:
    df = df[df["vimputation_"] == 1].reset_index(drop=True)

def num(col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

D = pd.DataFrame({
    "pid": num("pid"), "hhid": num("hhid"),
    "age": num("w01A001_age"), "gender": num("w01gender1"),
    "educ": num("w01A003"), "marital": num("w01A006"),
    "self_rated_health": num("w01C001"),
    "n_children": num("w01Ba003"),
    "hh_size": num("w01hhsize"),
    "wealth_house": num("w01F058"),
    "wealth_total": num("w01cassets"),
    "income_total": num("w01hhinc"),
    "bmi": num("w01bmi"),
})

# 10 慢病（w01chronic_a-j，1=是 5=否）
for k, name in [("a","hibp"),("b","diab"),("c","cancr"),("d","lung"),("e","liver"),
                ("f","heart"),("g","strok"),("h","psyche"),("i","arth"),("j","prost")]:
    D[name] = num(f"w01chronic_{k}")

# 抑郁（CESD 条目 w01C142-148，负性条目；w01C146 happy 反向）
cesd_items = ["w01C142","w01C143","w01C144","w01C145","w01C147","w01C148"]
cesd = pd.DataFrame({c: num(c) for c in cesd_items})
D["depress"] = cesd.sum(axis=1)

# 死亡（Exit 模块）
try:
    ex = read_latin(EXIT)
    ex_death = pd.DataFrame({
        "pid": pd.to_numeric(ex["pid"], errors="coerce") if "pid" in ex.columns else np.nan,
        "death_year": pd.to_numeric(ex["w09X_A010Y"], errors="coerce") if "w09X_A010Y" in ex.columns else np.nan,
    }).dropna(subset=["death_year"])
    D = D.merge(ex_death, on="pid", how="left")
    D["died"] = D["death_year"].notna()
except Exception as e:
    print(f"死亡 merge 失败: {e}")
    D["death_year"] = np.nan; D["died"] = False

# ADL 8 项（C201-208，1=独立 3=部分 5=不能）
adl_items = ["w01C201","w01C202","w01C203","w01C204","w01C205","w01C206","w01C207","w01C208"]
adl = pd.DataFrame({c: (num(c).isin([3,5])).where(num(c).notna()) for c in adl_items})
D["adl_count"] = adl.sum(axis=1)
D["adl_answered"] = adl.notna().sum(axis=1)

# IADL 9 项（C209-217，1=独立 3=部分 5=不能）
iadl_items = ["w01C209","w01C210","w01C211","w01C212","w01C213","w01C214","w01C215","w01C216","w01C217"]
iadl = pd.DataFrame({c: (num(c).isin([3,5])).where(num(c).notna()) for c in iadl_items})
D["iadl_count"] = iadl.sum(axis=1)

# 握力（C507-510 取最大值 kg）
grip_cols = ["w01C507","w01C508","w01C509","w01C510"]
grip = pd.DataFrame({c: num(c) for c in grip_cols})
D["grip"] = grip.max(axis=1)

# 认知 K-MMSE 原始值（C401-412，阈值后续统一处理）
for c in ["w01C401","w01C402","w01C403","w01C404","w01C405","w01C406",
          "w01C407","w01C408","w01C409","w01C410","w01C411","w01C412"]:
    D[c] = num(c)

# 同住 + 社会参与 + 空巢（沿用 v1）
cohab_cols = [c for c in df.columns if c.startswith("w01Ba013_")]
D["cohab_child"] = pd.DataFrame({c: (num(c)==1) for c in cohab_cols}).any(axis=1)
social_cols = [f"w01A017m{i:02d}" for i in range(1,8) if f"w01A017m{i:02d}" in df.columns]
D["social_particip"] = pd.DataFrame({c: (num(c)==1) for c in social_cols}).any(axis=1)
has_child = D["n_children"].fillna(0) > 0
cohab = (D["cohab_child"] == 1)
D["cohabiting"] = cohab
D["empty_nest"] = has_child & ~cohab
D["living_alone"] = (D["hh_size"] == 1)
D["no_spouse"] = D["marital"].isin([2,3,4,5]).where(D["marital"].notna())
D["no_social"] = (D["social_particip"] == 0).where(D["social_particip"].notna())

age60 = D["age"] >= 60
print(f"样本 {len(D)} 人；60+ {age60.sum()}")
print(f"死亡 {D['died'].sum()}（{D['died'].mean()*100:.1f}%）")
print(f"BMI 非空 {D['bmi'].notna().sum()}；握力非空 {D['grip'].notna().sum()}；IADL 非空 {D['iadl_count'].notna().sum()}")
print(f"ADL(≥1困难) {D['adl_count'].ge(1).sum()}；IADL(≥1困难) {D['iadl_count'].ge(1).sum()}")
print(f"认知 C401 非空 {D['w01C401'].notna().sum()}；握力中位 {D['grip'].median():.0f} kg")

out = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\klosa_w1_core_v2.csv"
D.to_csv(out, index=False)
print(f"\n已保存 {out}（{D.shape[1]} 列）")
