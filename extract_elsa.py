# -*- coding: utf-8 -*-
"""ELSA 抽取：复用 CHARLS v5 模板 + ELSA 变量映射（10波，ADL=Wallace，无握力，认知=词语回忆）。"""
import pandas as pd
import numpy as np

PATH = r"G:\ELSA-5050\gh_elsa_h.dta"
df = pd.read_stata(PATH, convert_categoricals=False)
w1 = df["r1agey"].notna()

def num(col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

D = pd.DataFrame({
    "pid": df["idauniq"].astype(float) if "idauniq" in df.columns else df.index.astype(float),
    "age": num("r1agey"), "gender": num("ragender"), "educ": num("raeducl"),
    "marital": num("r1mstat"), "n_marriages": num("r1mrct"),
    "income_earn": num("r1itearn"), "income_pension": num("r1itpena"),
    "wealth_fin": num("h1atotf"), "wealth_total": num("h1atotb"), "wealth_house": num("h1ahous"),
    "social_particip": num("r1socyr"), "n_children": num("r1child"),
    "cohab_child": num("h1coresd"), "hh_size": num("h1hhres"),
    "self_rated_health": num("r1shlt"), "lonely": num("r1lonela"),
})

# 疾病 12 项
for name, h in {"hibp":"hibpe","diab":"diabe","cancr":"cancre","lung":"lunge","heart":"hearte",
                "strok":"stroke","arth":"arthre","asthma":"asthmae","kidney":"kidneye",
                "digest":"digeste","liver":"livere","psyche":"psyche"}.items():
    D[name] = num("r1"+h)

# 功能/移动/认知/心理/测量
D["adl_count"] = num("r1adlwa")
D["walk_room"] = num("r1walkra"); D["walk100"] = num("r1walk100a"); D["chair"] = num("r1chaira")
D["immediate_recall"] = num("r1imrc"); D["delayed_recall"] = num("r1dlrc")
D["depress"] = num("r1cesd"); D["bmi"] = num("r1bmi")
for c in ["r1moneya","r1medsa","r1shopa","r1phonea","r1mealsa","r1housewka"]:
    D["iadl_"+c[2:]] = num(c)

# 死亡（10波 iwstat）
iw_cols = [f"r{k}iwstat" for k in range(2, 11) if f"r{k}iwstat" in df.columns]
def died(col):
    return num(col).isin([5, 6])
D["died_by_w3"] = np.logical_or.reduce([died(c) for c in ["r2iwstat","r3iwstat"]])
D["died_by_w4"] = np.logical_or.reduce([died(c) for c in ["r2iwstat","r3iwstat","r4iwstat"]])
D["died_by_w6"] = np.logical_or.reduce([died(c) for c in ["r2iwstat","r3iwstat","r4iwstat","r5iwstat","r6iwstat"]])
D["lost_unknown"] = (num("r10iwstat") == 9) if "r10iwstat" in df.columns else False

# FI（NaN 保留）
def di(name, s):
    D[name] = s.astype(float).where(s.notna())
for c in ["hibp","diab","cancr","lung","heart","strok","arth","asthma","kidney","digest","liver","psyche"]:
    di("d_"+c, D[c].where(D[c].isin([0,1])))
di("d_adl", (D["adl_count"] >= 1).where(D["adl_count"].notna()))
for c in ["iadl_moneya","iadl_medsa","iadl_shopa","iadl_phonea","iadl_mealsa","iadl_housewka"]:
    if c in D.columns:
        di("d_"+c, D[c].where(D[c].isin([0,1])))
di("d_walk_room", (D["walk_room"] == 1).where(D["walk_room"].notna()))
di("d_walk100", (D["walk100"] == 1).where(D["walk100"].notna()))
di("d_chair", (D["chair"] == 1).where(D["chair"].notna()))
# 词语回忆：最低四分位=缺陷
for c, nm in [("immediate_recall","d_imrc"), ("delayed_recall","d_dlrc")]:
    if D[c].notna().sum() > 0:
        q = D[c].quantile(0.25)
        di(nm, (D[c] <= q).where(D[c].notna()))
di("d_depress", (D["depress"] >= 10).where(D["depress"].notna()))  # 阈值待核
di("d_bmi_low", (D["bmi"] < 18.5).where(D["bmi"].notna()))
di("d_bmi_high", (D["bmi"] >= 30).where(D["bmi"].notna()))
di("d_srh_poor", (D["self_rated_health"] == 5).where(D["self_rated_health"].notna()))

dc = [c for c in D.columns if c.startswith("d_")]
D["n_deficits"] = D[dc].sum(axis=1)
D["n_answered"] = D[dc].notna().sum(axis=1)
D["fi"] = D["n_deficits"] / D["n_answered"]
D["fi"] = D["fi"].where(D["n_answered"] >= len(dc) * 0.8)

# 空巢 / 结构隔离 / 二婚
has_child = D["n_children"].fillna(0) > 0
cohab = (D["cohab_child"] == 1)
D["cohabiting"] = cohab
D["empty_nest"] = has_child & ~cohab
D["remarried"] = (D["n_marriages"] >= 2).where(D["n_marriages"].notna())
D["living_alone"] = (D["hh_size"] == 1)
D["no_spouse"] = D["marital"].isin([4,5,7,8]).where(D["marital"].notna())
D["no_social"] = (D["social_particip"] == 0).where(D["social_particip"].notna())
D["child_far_or_none"] = ~cohab
D["low_contact"] = np.nan  # ELSA w1 无 kcntf 对应？留空
iso = D[["no_spouse","living_alone","no_social","child_far_or_none"]].astype(float)
D["structural_isolation"] = iso.sum(axis=1, min_count=3)

# 报告
D = D[w1].reset_index(drop=True)
age60 = D["age"] >= 60
print(f"wave1 {len(D)} 人；60+ {age60.sum()}")
print(f"FI(60+) 中位 {D.loc[age60,'fi'].median():.3f}")
print(f"死亡 ~6yr(w4) 60+ {D.loc[age60,'died_by_w4'].sum()}/{age60.sum()}={D.loc[age60,'died_by_w4'].mean()*100:.1f}%")
print(f"死亡 ~10yr(w6) 60+ {D.loc[age60,'died_by_w6'].sum()}/{age60.sum()}={D.loc[age60,'died_by_w6'].mean()*100:.1f}%")
print(f"空巢 {D['empty_nest'].sum()} / 无配偶 {D['no_spouse'].sum()} / 独居 {D['living_alone'].sum()} / 高隔离 {D['structural_isolation'].ge(3).sum()}")
print(f"再婚 {D['remarried'].sum()}（缺失 {D['remarried'].isna().sum()}）")
print(f"\n缺失较多的列:")
print(D.isna().mean().sort_values(ascending=False).head(15).round(3).to_string())

out = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\elsa_w1_corrected.csv"
D.to_csv(out, index=False)
print(f"\n已保存 {out}（{D.shape[1]} 列）")
