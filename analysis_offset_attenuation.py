# -*- coding: utf-8 -*-
"""offset-calibration 框架下的 θ 衰减重算：握力 / 社会因素 / 种族(HRS)。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (load_charls, load_elsa, load_share, load_hrs, build_fi12)


def clean_key(s):
    return pd.to_numeric(s.astype(str).str.lstrip("0"), errors="coerce")


def load_grip(path, idcol, gripcol, agecol, str_key=False):
    df = pd.read_stata(path, columns=[idcol, gripcol, agecol], convert_categoricals=False)
    df = df[df[agecol].notna() & (df[agecol] >= 60)]
    key = df[idcol].astype(str) if str_key else clean_key(df[idcol])
    s = pd.to_numeric(df[gripcol], errors="coerce")
    s = s.where(s.between(5, 60))
    out = pd.Series(s.values, index=key.values)
    return out[~out.index.duplicated(keep="first")]


grip = {
    "CHARLS": load_grip(r"G:\CHARLS\H_CHARLS_D_Data.dta", "ID", "r1gripsum", "r1agey"),
    "SHARE": load_grip(r"G:\SHARE\GH_SHARE_g.dta", "mergeid", "r1gripsum", "r1agey", str_key=True),
    "ELSA": load_grip(r"G:\ELSA-5050\gh_elsa_h.dta", "idauniq", "r2gripsum", "r1agey"),
}


def off_theta(D, y, extra=()):
    base = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    for c in extra:
        base &= D[c].notna()
    D = D[base].reset_index(drop=True); y = y[base].values.astype(float)
    if len(D) < 100:
        return np.nan, np.nan, len(D)
    Xr = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    mr = sm.Logit(y, Xr).fit(disp=0)
    eta = Xr @ mr.params
    Xo = pd.DataFrame({"disadv": D["disadv"]})
    for c in extra:
        Xo[c] = D[c].astype(float)
    Xo = sm.add_constant(Xo)
    mo = sm.Logit(y, Xo, offset=eta).fit(disp=0)
    return mo.params["disadv"], mo.bse["disadv"], len(D)


def prep(D, name):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D["wealth_total"].where(D["wealth_total"] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    pid = D["pid"].astype(str) if name == "SHARE" else pd.to_numeric(D["pid"], errors="coerce")
    if name in grip:
        D["grip"] = pid.map(grip[name])
    return D, y


print("=== 握力衰减（offset）===")
for name in ["CHARLS", "ELSA", "SHARE"]:
    D, y = prep(load_charls() if name == "CHARLS" else load_elsa() if name == "ELSA" else load_share(), name)
    t0, s0, n0 = off_theta(D, y)
    t1, s1, n1 = off_theta(D, y, extra=["grip"])
    print(f"{name}: θ={t0:.4f}→{t1:.4f} 衰减{(t0-t1)/t0*100:.1f}% (n={n1})")

print("\n=== 社会因素衰减（offset，全加）===")
social_cols = {
    "CHARLS": ["educ", "living_alone", "no_spouse", "no_social"],
    "ELSA": ["educ", "living_alone", "no_spouse", "no_social"],
    "SHARE": ["educ", "living_alone", "no_spouse", "no_social"],
    "HRS": ["educ"],
}
for name in ["CHARLS", "ELSA", "SHARE", "HRS"]:
    D, y = prep(load_charls() if name == "CHARLS" else load_elsa() if name == "ELSA" else load_share() if name == "SHARE" else load_hrs(), name)
    cols = [c for c in social_cols[name] if c in D.columns]
    t0, s0, n0 = off_theta(D, y)
    t1, s1, n1 = off_theta(D, y, extra=cols)
    print(f"{name}: θ={t0:.4f}→{t1:.4f} 衰减{(t0-t1)/t0*100:.1f}% (n={n1}, {cols})")

print("\n=== 种族调整（HRS，offset）===")
D, y = prep(load_hrs(), "HRS")
D["race_eth"] = pd.to_numeric(D["race_eth"], errors="coerce")
D["r_black"] = (D["race_eth"] == 1).astype(float)
D["r_hisp"] = (D["race_eth"] == 2).astype(float)
D["r_other"] = (D["race_eth"] == 3).astype(float)
t0, s0, n0 = off_theta(D, y)
t1, s1, n1 = off_theta(D, y, extra=["r_black", "r_hisp", "r_other"])
print(f"HRS: θ={t0:.4f}→{t1:.4f} (变化{(t1-t0)/t0*100:+.1f}%)")
