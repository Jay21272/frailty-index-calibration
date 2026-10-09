# -*- coding: utf-8 -*-
"""校准三件套(SES盲模型 FI+age+sex 不含disadv)：穷/富组各自的校准截距CITL + 校准斜率。
循环论证陷阱规避：审计对象必须是SES盲模型，在组内评估校准。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (BASE, load_charls, load_elsa, load_share, load_hrs, build_fi12)


def calibration(name, D, wc):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["wq"].notna()
    D = D[m]; y = y[m].astype(float)
    # SES盲模型(不含disadv)
    Xb = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    pred = sm.Logit(y, Xb).fit(disp=0).predict(Xb)
    lo = np.log(pred / (1 - pred))
    out = {}
    for grp, label in [(1, "穷Q1"), (5, "富Q5")]:
        sub = (D["wq"] == grp).values
        yy = y[sub]; loo = lo[sub]
        # 校准截距 CITL: logit(P) = a + offset(loo)
        Xa = np.ones((yy.shape[0], 1))
        ma = sm.GLM(yy, Xa, offset=loo, family=sm.families.Binomial()).fit()
        # 校准斜率: logit(P) = a + b*loo
        Xs = sm.add_constant(pd.DataFrame({"lo": loo}))
        ms = sm.Logit(yy, Xs).fit(disp=0)
        out[label] = {
            "截距a": ma.params[0], "截距SE": ma.bse[0], "截距p": ma.pvalues[0],
            "斜率b": ms.params["lo"], "斜率SE": ms.bse["lo"],
            "斜率<1_p": ms.pvalues["lo"],  # 双侧；<1 需单侧判断
            "n": int(sub.sum()), "events": int(yy.sum()),
        }
    return name, out


cohorts = [
    ("CHARLS", load_charls(), "wealth_total"),
    ("ELSA", load_elsa(), "wealth_total"),
    ("SHARE", load_share(), "wealth_total"),
    ("HRS", load_hrs(), "wealth_total"),
]
print("=== 校准截距CITL + 斜率（SES盲模型，穷Q1 vs 富Q5）===")
print("截距>0=该组被低估；斜率<1=预测区分度压缩(过度自信)")
for name, D, wc in cohorts:
    n, out = calibration(name, D, wc)
    print(f"\n{name}:")
    for grp in ["穷Q1", "富Q5"]:
        r = out[grp]
        print(f"  {grp}: 截距a={r['截距a']:+.3f}(p={r['截距p']:.3f}) 斜率b={r['斜率b']:.3f}(SE{r['斜率SE']:.3f}) n={r['n']}事件{r['events']}")
