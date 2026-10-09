# -*- coding: utf-8 -*-
"""θ 的机制分解：把「独立SES风险」具体拆成教育/独居/无配偶/社会参与等，量化每个因素衰减θ多少。
基准 logit(death)=FI+age+sex+θ·disadv；逐个加入社会因素，看 θ 衰减比例。描述性分解非因果中介。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import BASE, load_charls, load_elsa, load_share, load_hrs, build_fi12


def fit_theta(D, y, fi, age, sex, disadv, extra=None):
    cols = {"FI": fi, "age": age, "sex": sex, "disadv": disadv}
    if extra:
        for k, v in extra.items():
            cols[k] = v
    X = sm.add_constant(pd.DataFrame(cols))
    m = pd.DataFrame(cols).notna().all(axis=1) & y.notna()
    mdl = sm.Logit(y[m], X[m]).fit(disp=0)
    return mdl.params["disadv"], mdl.bse["disadv"]


def decompose(name, D, wc):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    base, base_se = fit_theta(D, y, D["fi"], D["age"], D["gender"], D["disadv"])
    out = {"name": name, "base": base}
    factors = {}
    if "educ" in D.columns and D["educ"].notna().sum() > 100:
        factors["教育"] = {"educ": D["educ"]}
    if "living_alone" in D.columns and D["living_alone"].notna().sum() > 100:
        factors["独居"] = {"living_alone": D["living_alone"].astype(float)}
    if "no_spouse" in D.columns and D["no_spouse"].notna().sum() > 100:
        factors["无配偶"] = {"no_spouse": D["no_spouse"].astype(float)}
    if "no_social" in D.columns and D["no_social"].notna().sum() > 100:
        factors["社会参与缺失"] = {"no_social": D["no_social"].astype(float)}
    for fname, extra in factors.items():
        t, s = fit_theta(D, y, D["fi"], D["age"], D["gender"], D["disadv"], extra)
        out[fname] = t
    # 全加
    all_extra = {}
    for extra in factors.values():
        all_extra.update(extra)
    t_all, s_all = fit_theta(D, y, D["fi"], D["age"], D["gender"], D["disadv"], all_extra)
    out["全加"] = t_all
    return out


cohorts = [
    ("CHARLS", load_charls(), "wealth_total"),
    ("ELSA", load_elsa(), "wealth_total"),
    ("SHARE", load_share(), "wealth_total"),
    ("HRS", load_hrs(), "wealth_total"),
]
rows = [decompose(n, D, wc) for n, D, wc in cohorts]

print("=== θ 衰减分解（基准=FI+age+sex+disadv，逐加社会因素看θ衰减）===")
print(f"{'库':<8}{'基准θ':>8}{'教育':>8}{'独居':>8}{'无配偶':>8}{'社参缺失':>9}{'全加':>8}{'残余%':>7}")
for r in rows:
    base = r["base"]
    def pct(k):
        return f"{r[k]:.3f}" if k in r else "  —"
    residual = r["全加"] / base * 100 if "全加" in r else np.nan
    print(f"{r['name']:<8}{base:>8.3f}{pct('教育'):>8}{pct('独居'):>8}{pct('无配偶'):>8}{pct('社会参与缺失'):>9}{pct('全加'):>8}{residual:>7.0f}")
