# -*- coding: utf-8 -*-
"""offset 框架重跑稳健性：10条共同条目FI(S8) / 五分位哑变量(S9) / 完整版FI(S2)。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (DISEASES, load_charls, load_elsa, load_share, load_hrs,
                               build_fi12, clean_disease)


def build_fi10(D):
    it = {c: clean_disease(D[c]) for c in DISEASES if c in D.columns}
    it["adl"] = D["adl_any"].astype(float).where(D["adl_any"].notna())
    it["iadl"] = D["iadl_any"].astype(float).where(D["iadl_any"].notna())
    it["srh"] = (D["self_rated_health"].isin([4, 5])).where(D["self_rated_health"].isin([1, 2, 3, 4, 5])).astype(float)
    dd = pd.DataFrame(it); denom = dd.notna().sum(axis=1)
    n_possible = int((dd.notna().sum(axis=0) > 0).sum())
    return (dd.sum(axis=1) / denom).where(denom >= int(n_possible * 0.8))


def build_full_fi(D):
    it = {c: clean_disease(D[c]) for c in DISEASES if c in D.columns}
    for c in ["asthma", "kidney", "digest", "liver", "psyche"]:
        if c in D.columns and D[c].notna().sum() > 500:
            it[c] = clean_disease(D[c])
    it["adl"] = D["adl_any"].astype(float).where(D["adl_any"].notna())
    it["iadl"] = D["iadl_any"].astype(float).where(D["iadl_any"].notna())
    for c in D.columns:
        if c.startswith("iadl_") and D[c].notna().sum() > 500:
            it[c] = D[c].where(D[c].isin([0, 1])).astype(float)
    if "mobil_count" in D.columns and D["mobil_count"].notna().sum() > 500:
        it["mobil"] = (D["mobil_count"] >= 1).where(D["mobil_count"].notna()).astype(float)
    for c in ["walk_room", "walk100", "walk1km", "chair"]:
        if c in D.columns and D[c].notna().sum() > 500:
            it[c] = (D[c] == 1).where(D[c].notna()).astype(float)
    for c in ["immediate_recall", "delayed_recall", "ser7"]:
        if c in D.columns and D[c].notna().sum() > 500:
            q = D[c].quantile(0.25); it["cog_" + c] = (D[c] <= q).where(D[c].notna()).astype(float)
    if "draw" in D.columns and D["draw"].notna().sum() > 500:
        it["cog_draw"] = (D["draw"] == 1).where(D["draw"].notna()).astype(float)
    for c in ["depress", "cesd"]:
        if c in D.columns and D[c].notna().sum() > 500:
            q = D[c].quantile(0.75); it["dep_" + c] = (D[c] >= q).where(D[c].notna()).astype(float)
    it["bmi_low"] = (D["bmi"] < 18.5).where(D["bmi"].notna()).astype(float)
    it["bmi_high"] = (D["bmi"] >= 30).where(D["bmi"].notna()).astype(float)
    it["srh"] = (D["self_rated_health"].isin([4, 5])).where(D["self_rated_health"].isin([1, 2, 3, 4, 5])).astype(float)
    dd = pd.DataFrame(it); denom = dd.notna().sum(axis=1)
    n_possible = int((dd.notna().sum(axis=0) > 0).sum())
    return (dd.sum(axis=1) / denom).where(denom >= int(n_possible * 0.8)), n_possible


def prep(D):
    D = D[D["age"] >= 60].copy()
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D["wealth_total"].where(D["wealth_total"] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    return D, y


def off_theta(D, y, fi, extra=()):
    m = fi.notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    for c in extra:
        m &= D[c].notna()
    D2 = D[m].reset_index(drop=True); y2 = y[m].values.astype(float)
    Xr = sm.add_constant(pd.DataFrame({"FI": fi[m].values, "age": D2["age"], "sex": D2["gender"]}))
    mr = sm.Logit(y2, Xr).fit(disp=0); eta = Xr @ mr.params
    Xo = pd.DataFrame({"disadv": D2["disadv"]})
    for c in extra:
        Xo[c] = D2[c].astype(float).values
    Xo = sm.add_constant(Xo)
    mo = sm.Logit(y2, Xo, offset=eta).fit(disp=0)
    return mo.params["disadv"], mo.bse["disadv"], len(y2)


cohorts = [("CHARLS", load_charls()), ("ELSA", load_elsa()), ("SHARE", load_share()), ("HRS", load_hrs())]

print("=== S8 十条目FI θ（offset）===")
for name, D in cohorts:
    D, y = prep(D)
    fi12 = build_fi12(D); fi10 = build_fi10(D)
    t12 = off_theta(D, y, fi12)[0]; t10 = off_theta(D, y, fi10)[0]
    print(f"{name}: 12项θ={t12:.4f} 10项θ={t10:.4f} Δθ={t10-t12:+.4f}")

print("\n=== S9 五分位哑变量 θ（offset，参考最富 disadv=0）===")
for name, D in cohorts:
    D, y = prep(D); fi = build_fi12(D)
    m = fi.notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    D2 = D[m].reset_index(drop=True); y2 = y[m].values.astype(float)
    Xr = sm.add_constant(pd.DataFrame({"FI": fi[m].values, "age": D2["age"], "sex": D2["gender"]}))
    mr = sm.Logit(y2, Xr).fit(disp=0); eta = Xr @ mr.params
    Xd = sm.add_constant(pd.DataFrame({
        "d1": (D2["disadv"] == 1).astype(float), "d2": (D2["disadv"] == 2).astype(float),
        "d3": (D2["disadv"] == 3).astype(float), "d4": (D2["disadv"] == 4).astype(float)}))
    md = sm.Logit(y2, Xd, offset=eta).fit(disp=0)
    Xl = sm.add_constant(pd.DataFrame({"disadv": D2["disadv"]}))
    ml = sm.Logit(y2, Xl, offset=eta).fit(disp=0)
    from scipy.stats import chi2
    lr = 2 * (md.llf - ml.llf); p_lr = chi2.sf(lr, df=3)
    ds = [f"{md.params['d%d' % k]:+.3f}" for k in range(1, 5)]
    print(f"{name}: " + " ".join(ds) + f"  LR线性p={p_lr:.3f}")

print("\n=== S2 完整版FI θ（offset）===")
for name, D in cohorts:
    D, y = prep(D)
    fi12 = build_fi12(D); fifull, npos = build_full_fi(D)
    t12 = off_theta(D, y, fi12)[0]; tfull = off_theta(D, y, fifull)[0]
    print(f"{name}: 12项θ={t12:.4f} 完整版({npos}项)θ={tfull:.4f} Δθ={tfull-t12:+.4f}")
