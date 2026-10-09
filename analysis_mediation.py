# -*- coding: utf-8 -*-
"""中介分析：disadv → 握力 → 死亡。
路径a: grip ~ disadv(控FI/age/sex)  [OLS]
路径b: death ~ grip(控disadv/FI/age/sex)  [Logit]
总效应c: death ~ disadv; 直接效应c': death ~ disadv+grip
中介比例 = (c - c')/c（logit 系数差近似），bootstrap 500 次。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import load_charls, load_elsa, load_share, build_fi12


def clean_key_num(s):
    return pd.to_numeric(s.astype(str).str.lstrip("0"), errors="coerce")


def load_grip(path, idcol, gripcol, agecol, str_key=False):
    df = pd.read_stata(path, columns=[idcol, gripcol, agecol], convert_categoricals=False)
    df = df[df[agecol].notna() & (df[agecol] >= 60)]
    key = df[idcol].astype(str) if str_key else clean_key_num(df[idcol])
    s = pd.to_numeric(df[gripcol], errors="coerce").where(pd.to_numeric(df[gripcol], errors="coerce").between(5, 60))
    out = pd.Series(s.values, index=key.values)
    return out[~out.index.duplicated(keep="first")]


grip = {
    "CHARLS": load_grip(r"G:\CHARLS\H_CHARLS_D_Data.dta", "ID", "r1gripsum", "r1agey"),
    "SHARE": load_grip(r"G:\SHARE\GH_SHARE_g.dta", "mergeid", "r1gripsum", "r1agey", str_key=True),
    "ELSA": load_grip(r"G:\ELSA-5050\gh_elsa_h.dta", "idauniq", "r2gripsum", "r1agey"),
}
str_id = {"SHARE"}


def mediation(name, D, rng):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    pid = D["pid"].astype(str) if name in str_id else pd.to_numeric(D["pid"], errors="coerce")
    D["grip"] = pid.map(grip[name])
    w = D["wealth_total"].where(D["wealth_total"] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    base = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna() & D["grip"].notna()
    dd = D[base].reset_index(drop=True); yy = y[base].reset_index(drop=True)

    conf = ["fi", "age", "gender"]
    # 路径 a
    ma = sm.OLS(dd["grip"], sm.add_constant(dd[conf + ["disadv"]])).fit()
    a = ma.params["disadv"]
    # 总效应 c
    mc = sm.Logit(yy, sm.add_constant(dd[conf + ["disadv"]])).fit(disp=0)
    c = mc.params["disadv"]
    # 直接效应 c' + 路径 b
    mbc = sm.Logit(yy, sm.add_constant(dd[conf + ["disadv", "grip"]])).fit(disp=0)
    b = mbc.params["grip"]
    cp = mbc.params["disadv"]
    prop = (c - cp) / c * 100 if c != 0 else np.nan
    return a, b, c, cp, prop


rng = np.random.default_rng(42)
print(f"{'库':<8}{'a(disadv→握力)':>13}{'b(握力→死亡)':>13}{'c(总效应)':>11}{'c\'(直接)':>11}{'中介%':>9}{'中介95%CI':>16}")
for name, load in [("CHARLS", load_charls), ("ELSA", load_elsa), ("SHARE", load_share)]:
    D = load()
    # 全样本准备
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    pid = D["pid"].astype(str) if name in str_id else pd.to_numeric(D["pid"], errors="coerce")
    D["grip"] = pid.map(grip[name])
    w = D["wealth_total"].where(D["wealth_total"] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    base = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna() & D["grip"].notna()
    dd = D[base].reset_index(drop=True); yy = y[base].reset_index(drop=True)
    conf = ["fi", "age", "gender"]

    def once():
        idx = rng.integers(0, len(dd), len(dd))
        d = dd.iloc[idx]; Y = yy.iloc[idx]
        ma = sm.OLS(d["grip"], sm.add_constant(d[conf + ["disadv"]])).fit()
        c = sm.Logit(Y, sm.add_constant(d[conf + ["disadv"]])).fit(disp=0).params["disadv"]
        mbc = sm.Logit(Y, sm.add_constant(d[conf + ["disadv", "grip"]])).fit(disp=0)
        cp = mbc.params["disadv"]
        return (c - cp) / c * 100 if c != 0 else np.nan

    a = sm.OLS(dd["grip"], sm.add_constant(dd[conf + ["disadv"]])).fit().params["disadv"]
    b = sm.Logit(yy, sm.add_constant(dd[conf + ["disadv", "grip"]])).fit(disp=0).params["grip"]
    c = sm.Logit(yy, sm.add_constant(dd[conf + ["disadv"]])).fit(disp=0).params["disadv"]
    cp = sm.Logit(yy, sm.add_constant(dd[conf + ["disadv", "grip"]])).fit(disp=0).params["disadv"]
    prop = (c - cp) / c * 100
    boots = np.array([once() for _ in range(500)])
    boots = boots[~np.isnan(boots)]
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    print(f"{name:<8}{a:>13.2f}{b:>13.3f}{c:>11.3f}{cp:>11.3f}{prop:>8.1f}%   [{ci[0]:.1f}, {ci[1]:.1f}]")
