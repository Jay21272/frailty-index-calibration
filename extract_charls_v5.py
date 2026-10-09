# -*- coding: utf-8 -*-
"""CHARLS v5：按四专家审核修正——死亡用iwstat、FI重做31项、社会变量编码修正。"""
import pandas as pd
import numpy as np

PATH = r"G:\CHARLS\H_CHARLS_D_Data.dta"
df = pd.read_stata(PATH, convert_categoricals=False)
w1 = df["r1agey"].notna()

def num(col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

D = pd.DataFrame({
    "pid": df["ID"].astype(float),
    "age": num("r1agey"), "gender": num("ragender"), "educ": num("raeducl"),
    "marital": num("r1mstat"), "n_marriages": num("r1mrct"),
    # 经济
    "income_earn": num("r1itearn"), "income_pension": num("r1ipeni"),
    "wealth_fin": num("hh1atotf"), "wealth_total": num("hh1atotb"), "wealth_house": num("hh1ahousa"),
    # 社会
    "social_particip": num("r1socwk"), "n_children": num("h1child"),
    "cohab_child": num("h1coresd"), "live_near_child": num("h1lvnear"),
    "weekly_child_contact": num("h1kcntf"), "hh_size": num("h1hhres"),
    # 生活质量（课题2，编码方向修正）
    "self_rated_health": num("r1shlt"), "life_satisfaction": num("r1satlife"), "lonely": num("r1flonel"),
})

# ---- 慢病 12 项 ----
for name, h in {"hibp":"hibpe","diab":"diabe","cancr":"cancre","lung":"lunge","heart":"hearte",
                "strok":"stroke","arth":"arthre","asthma":"asthmae","kidney":"kidneye",
                "digest":"digeste","liver":"livere","psyche":"psyche"}.items():
    D[name] = num("r1"+h)

# ---- 功能/移动/认知/握力 ----
D["adl_count"] = num("r1adlfive")
D["grip"] = num("r1gripsum")
D["walk100"] = num("r1walk100a"); D["walk1km"] = num("r1walk1kma"); D["chair"] = num("r1chaira")
D["draw"] = num("r1draw"); D["depress"] = num("r1cesd10")
D["bmi"] = num("r1mbmi")
for c in ["r1moneya","r1medsa","r1shopa","r1mealsa","r1housewka"]:
    D["iadl_"+c[2:]] = num(c)  # iadl_moneya/medsa/shopa/mealsa/housewka

# ---- 死亡结局（用 iwstat 访谈状态）----
D["iw2"] = num("r2iwstat"); D["iw3"] = num("r3iwstat"); D["iw4"] = num("r4iwstat")
def died_by(iw):
    return iw.isin([5, 6])
D["died_by_w3"] = died_by(D["iw2"]) | died_by(D["iw3"])   # ~4年(2015)
D["died_by_w4"] = died_by(D["iw2"]) | died_by(D["iw3"]) | died_by(D["iw4"])  # ~7年(2018)
D["lost_unknown"] = (D["iw4"] == 9)  # 生死不明，删失

# ---- FI（31项，NaN 保留，分母=已答项）----
def di(name, s):
    # 保留 NaN，0/1 计缺陷
    D[name] = s.astype(float).where(s.notna())

for c in ["hibp","diab","cancr","lung","heart","strok","arth","asthma","kidney","digest","liver","psyche"]:
    di("d_"+c, D[c].where(D[c].isin([0,1])))
di("d_adl", (D["adl_count"] >= 1).where(D["adl_count"].notna()))
for c in ["iadl_moneya","iadl_medsa","iadl_shopa","iadl_mealsa","iadl_housewka"]:
    di("d_"+c, D[c].where(D[c].isin([0,1])))
di("d_walk100", (D["walk100"] == 1).where(D["walk100"].notna()))
di("d_walk1km", (D["walk1km"] == 1).where(D["walk1km"].notna()))
di("d_chair", (D["chair"] == 1).where(D["chair"].notna()))
di("d_draw", (D["draw"] == 1).where(D["draw"].notna()))
di("d_depress", (D["depress"] >= 10).where(D["depress"].notna()))
di("d_bmi_low", (D["bmi"] < 18.5).where(D["bmi"].notna()))
di("d_bmi_high", (D["bmi"] >= 30).where(D["bmi"].notna()))
di("d_srh_poor", (D["self_rated_health"] == 5).where(D["self_rated_health"].notna()))  # 严格"差"=5

# 握力低 = 分性别最低五分位
grip_lo = pd.Series(np.nan, index=D.index)
for g in [1, 2]:
    sub = D["gender"] == g
    q = D.loc[sub, "grip"].quantile(0.2)
    grip_lo[sub] = (D.loc[sub, "grip"] <= q)
di("d_grip_low", grip_lo.where(D["grip"].notna()))

dc = [c for c in D.columns if c.startswith("d_")]
D["n_deficits"] = D[dc].sum(axis=1)
D["n_answered"] = D[dc].notna().sum(axis=1)
D["fi"] = D["n_deficits"] / D["n_answered"]
D["fi"] = D["fi"].where(D["n_answered"] >= len(dc) * 0.8)  # ≥80% 已答才计

# ---- 空巢 / 二婚 / 结构隔离（编码修正）----
has_child = D["n_children"].fillna(0) > 0
cohab = (D["cohab_child"] == 1); near = (D["live_near_child"] == 1)
D["cohabiting"] = cohab
D["empty_nest"] = has_child & ~cohab
D["far_empty_nest"] = has_child & ~cohab & ~near
D["remarried"] = (D["n_marriages"] >= 2).where(D["n_marriages"].notna())
D["living_alone"] = (D["hh_size"] == 1)
D["no_spouse"] = D["marital"].isin([4,5,7,8]).where(D["marital"].notna())  # 分居/离异/丧偶/未婚
D["no_social"] = (D["social_particip"] == 0).where(D["social_particip"].notna())  # NaN保留(代理受访)
D["child_far_or_none"] = (~cohab) & (~near)
D["low_contact"] = (D["weekly_child_contact"] == 0).where(D["weekly_child_contact"].notna())
iso = D[["no_spouse","living_alone","no_social","child_far_or_none","low_contact"]].astype(float)
D["structural_isolation"] = iso.sum(axis=1, min_count=3)

# 生活质量（课题2，方向修正）
D["dissatisfied_life"] = D["life_satisfaction"].isin([1,2]).where(D["life_satisfaction"].notna())

# ---- 只保留 wave1 + 报告 ----
D = D[w1].reset_index(drop=True)
age60 = D["age"] >= 60
print(f"wave1 {len(D)} 人；60+ {age60.sum()}")
print(f"FI(60+) 中位 {D.loc[age60,'fi'].median():.3f}（修正前0.235）")
print(f"死亡: ~4年(w3) 60+ {(D.loc[age60,'died_by_w3']==True).sum()}/{age60.sum()}={(D.loc[age60,'died_by_w3']==True).mean()*100:.1f}%")
print(f"死亡: ~7年(w4) 60+ {(D.loc[age60,'died_by_w4']==True).sum()}/{age60.sum()}={(D.loc[age60,'died_by_w4']==True).mean()*100:.1f}%")
print(f"生死不明(删失) {(D['lost_unknown']==True).sum()}")
print(f"无配偶 {D['no_spouse'].sum()} / 远巢空巢 {D['far_empty_nest'].sum()} / 再婚 {D['remarried'].sum()} / 独居 {D['living_alone'].sum()}")
print(f"高隔离(SI≥3) {D['structural_isolation'].ge(3).sum()} / 生活不满意 {D['dissatisfied_life'].sum()}")

out = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\charls_w1_corrected.csv"
D.to_csv(out, index=False)
print(f"\n已保存 {out}（{D.shape[1]} 列）")
