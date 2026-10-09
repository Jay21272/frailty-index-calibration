# -*- coding: utf-8 -*-
"""DIF 稳健性在 offset 框架下重跑：剔除DIF条目后θ是否仍增强。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (DISEASES, load_charls, load_elsa, load_share, load_hrs, clean_disease)


def build_items(D):
    it = {}
    for c in DISEASES:
        it[c] = clean_disease(D[c])
    it["adl"] = D["adl_any"].astype(float).where(D["adl_any"].notna())
    it["iadl"] = D["iadl_any"].astype(float).where(D["iadl_any"].notna())
    it["bmi_low"] = (D["bmi"] < 18.5).where(D["bmi"].notna()).astype(float)
    it["bmi_high"] = (D["bmi"] >= 30).where(D["bmi"].notna()).astype(float)
    it["srh"] = (D["self_rated_health"].isin([4, 5])).where(D["self_rated_health"].isin([1, 2, 3, 4, 5])).astype(float)
    return pd.DataFrame(it)


def offset_theta(D, wc, items):
    denom = items.notna().sum(axis=1)
    n_possible = int((items.notna().sum(axis=0) > 0).sum())
    fi = (items.sum(axis=1) / denom).where(denom >= int(n_possible * 0.8))
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D = D.copy(); D["fi"] = fi
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    D = D[m].reset_index(drop=True); y = y[m].values.astype(float)
    Xr = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    mr = sm.Logit(y, Xr).fit(disp=0); eta = Xr @ mr.params
    Xo = sm.add_constant(pd.DataFrame({"disadv": D["disadv"]}))
    mo = sm.Logit(y, Xo, offset=eta).fit(disp=0)
    return mo.params["disadv"], mo.bse["disadv"], mo.pvalues["disadv"]


def dif_items(D, items, wc):
    D = D[D["age"] >= 60].copy()
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    poor = (D["wq"] == 1); rich = (D["wq"] == 5)
    base = (poor | rich)
    dif = []
    for col in items.columns:
        purified = items.drop(col, axis=1).sum(axis=1)
        sub_col = base & items[col].notna() & purified.notna()
        if sub_col.sum() < 100:
            continue
        y = items.loc[sub_col, col].values
        pur = purified[sub_col].values
        poorv = (D.loc[sub_col, "wq"] == 1).astype(float).values
        X0 = sm.add_constant(pd.DataFrame({"purified": pur}))
        X1 = sm.add_constant(pd.DataFrame({"purified": pur, "poor": poorv}))
        m0 = sm.Logit(y, X0).fit(disp=0); m1 = sm.Logit(y, X1).fit(disp=0)
        r2_0 = 1 - m0.llf / m0.llnull; r2_1 = 1 - m1.llf / m1.llnull
        p = m1.pvalues["poor"]
        if p < 0.05 and (r2_1 - r2_0) >= 0.02:
            dif.append(col)
    return dif


cohorts = [("CHARLS", load_charls()), ("ELSA", load_elsa()), ("SHARE", load_share()), ("HRS", load_hrs())]
print("=== offset 框架下 DIF 稳健性：全12项θ vs 剔除DIF条目后θ ===")
for name, D in cohorts:
    D = D[D["age"] >= 60].copy()
    items = build_items(D)
    dif = dif_items(D, items, "wealth_total")
    t_all = offset_theta(D, "wealth_total", items)[0]
    keep = [c for c in items.columns if c not in dif]
    t_keep = offset_theta(D, "wealth_total", items[keep])[0]
    print(f"{name}: 有DIF={dif} | 全12项θ={t_all:.3f} → 剔除后θ={t_keep:.3f} (Δ={t_keep-t_all:+.3f})")
