# -*- coding: utf-8 -*-
"""offset 框架重跑失能结局 θ（新发ADL失能，基线ADL=0→随访≥1，随访死亡排除）。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import DISEASES, clean_disease


def build_fi12_manual(D):
    it = {}
    for c in DISEASES:
        if c in D.columns:
            it[c] = clean_disease(D[c])
    it["adl"] = (D["adl_count"] >= 1).where(D["adl_count"].notna()).astype(float)
    it["iadl"] = D["iadl_any"].astype(float).where(D["iadl_any"].notna())
    it["bmi_low"] = (D["bmi"] < 18.5).where(D["bmi"].notna()).astype(float)
    it["bmi_high"] = (D["bmi"] >= 30).where(D["bmi"].notna()).astype(float)
    it["srh"] = (D["self_rated_health"].isin([4, 5])).where(D["self_rated_health"].isin([1, 2, 3, 4, 5])).astype(float)
    dd = pd.DataFrame(it); denom = dd.notna().sum(axis=1)
    n_possible = int((dd.notna().sum(axis=0) > 0).sum())
    return (dd.sum(axis=1) / denom).where(denom >= int(n_possible * 0.8))


def run_disability_off(name, D, wc, base_adl, fu_adl, fu_death):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12_manual(D)
    b0 = pd.to_numeric(D[base_adl], errors="coerce")
    fu = pd.to_numeric(D[fu_adl], errors="coerce")
    fd = pd.to_numeric(D[fu_death], errors="coerce")
    at_risk = (b0 == 0)
    y = ((fu >= 1).astype(float)).where(at_risk & fu.notna())
    y = y.where(~(fd.isin([5, 6]) & at_risk))
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    D = D[m].reset_index(drop=True); y = y[m].values.astype(float)
    # offset：参考模型(FI+age+sex) 固定为 offset，只估 disadv
    Xr = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    mr = sm.Logit(y, Xr).fit(disp=0); eta = Xr @ mr.params
    Xo = sm.add_constant(pd.DataFrame({"disadv": D["disadv"]}))
    mo = sm.Logit(y, Xo, offset=eta).fit(disp=0)
    return mo.params["disadv"], mo.bse["disadv"], mo.pvalues["disadv"], len(D), int(y.sum())


# CHARLS
ch = pd.read_stata(r"G:\CHARLS\H_CHARLS_D_Data.dta",
                   columns=["ID", "r1agey", "ragender", "r1adlwa", "r3adlwa", "r3iwstat",
                            "r1hibpe", "r1diabe", "r1cancre", "r1lunge", "r1hearte", "r1stroke", "r1arthre",
                            "r1moneya", "r1medsa", "r1shopa", "r1mealsa", "r1housewka", "r1mbmi", "r1shlt",
                            "hh1atotb"], convert_categoricals=False)
ch = ch.rename(columns={"ID": "pid", "r1agey": "age", "ragender": "gender", "r1mbmi": "bmi",
                        "r1shlt": "self_rated_health", "hh1atotb": "wealth_total"})
for k, n in [("hibpe", "hibp"), ("diabe", "diab"), ("cancre", "cancr"), ("lunge", "lung"),
             ("hearte", "heart"), ("stroke", "strok"), ("arthre", "arth")]:
    ch[n] = pd.to_numeric(ch["r1" + k], errors="coerce")
iadl_cols = ["r1moneya", "r1medsa", "r1shopa", "r1mealsa", "r1housewka"]
ch["adl_count"] = pd.to_numeric(ch["r1adlwa"], errors="coerce")
ch["iadl_any"] = pd.DataFrame({c: pd.to_numeric(ch[c], errors="coerce") for c in iadl_cols}).isin([1]).any(axis=1)

# ELSA
el = pd.read_stata(r"G:\ELSA-5050\gh_elsa_h.dta",
                   columns=["idauniq", "r1agey", "ragender", "r1adlwa", "r3adlwa", "r3iwstat",
                            "r1hibpe", "r1diabe", "r1cancre", "r1lunge", "r1hearte", "r1stroke", "r1arthre",
                            "r1moneya", "r1medsa", "r1shopa", "r1phonea", "r1mealsa", "r1housewka", "r1shlt",
                            "h1atotb"], convert_categoricals=False)
el = el.rename(columns={"idauniq": "pid", "r1agey": "age", "ragender": "gender",
                        "r1shlt": "self_rated_health", "h1atotb": "wealth_total"})
el["bmi"] = np.nan
for k, n in [("hibpe", "hibp"), ("diabe", "diab"), ("cancre", "cancr"), ("lunge", "lung"),
             ("hearte", "heart"), ("stroke", "strok"), ("arthre", "arth")]:
    el[n] = pd.to_numeric(el["r1" + k], errors="coerce")
iadl_cols = ["r1moneya", "r1medsa", "r1shopa", "r1phonea", "r1mealsa", "r1housewka"]
el["adl_count"] = pd.to_numeric(el["r1adlwa"], errors="coerce")
el["iadl_any"] = pd.DataFrame({c: pd.to_numeric(el[c], errors="coerce") for c in iadl_cols}).isin([1]).any(axis=1)

# SHARE
sh = pd.read_stata(r"G:\SHARE\GH_SHARE_g.dta",
                   columns=["mergeid", "r1agey", "ragender", "r1adlwa", "r4adlwa", "r4iwstat",
                            "r1hibpe", "r1diabe", "r1cancre", "r1lunge", "r1hearte", "r1stroke", "r1arthre",
                            "r1moneya", "r1medsa", "r1shopa", "r1phonea", "r1mealsa", "r1housewka", "r1bmi", "r1shlt",
                            "h1atotb"], convert_categoricals=False)
sh = sh.rename(columns={"mergeid": "pid", "r1agey": "age", "ragender": "gender", "r1bmi": "bmi",
                        "r1shlt": "self_rated_health", "h1atotb": "wealth_total"})
for k, n in [("hibpe", "hibp"), ("diabe", "diab"), ("cancre", "cancr"), ("lunge", "lung"),
             ("hearte", "heart"), ("stroke", "strok"), ("arthre", "arth")]:
    sh[n] = pd.to_numeric(sh["r1" + k], errors="coerce")
iadl_cols = ["r1moneya", "r1medsa", "r1shopa", "r1phonea", "r1mealsa", "r1housewka"]
sh["adl_count"] = pd.to_numeric(sh["r1adlwa"], errors="coerce")
sh["iadl_any"] = pd.DataFrame({c: pd.to_numeric(sh[c], errors="coerce") for c in iadl_cols}).isin([1]).any(axis=1)

print("=== 新发ADL失能 θ（offset）===")
for name, D, ba, fa, fd in [
    ("CHARLS", ch, "r1adlwa", "r3adlwa", "r3iwstat"),
    ("ELSA", el, "r1adlwa", "r3adlwa", "r3iwstat"),
    ("SHARE", sh, "r1adlwa", "r4adlwa", "r4iwstat"),
]:
    th, se, p, n, ev = run_disability_off(name, D, "wealth_total", ba, fa, fd)
    print(f"{name}: n={n} 事件={ev} θ={th:.4f} SE={se:.4f} p={p:.4f}")
