# -*- coding: utf-8 -*-
"""部署后果反事实：原评分(FI+age+sex) vs SES重校准评分(FI+age+sex+θ·disadv)，
固定高风险阈值(top 20%)下，多少穷人被「升级」进高风险组。锚风险分层不锚配给。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (BASE, load_charls, load_elsa, load_share, load_hrs, build_fi12)


def recalibrate(name, D, wc):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    D = D[m]; y = y[m]
    # 原评分（不含 SES）
    X0 = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    pred0 = sm.Logit(y, X0).fit(disp=0).predict(X0)
    # 重校准评分（含 θ·disadv）
    X1 = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"], "disadv": D["disadv"]}))
    pred1 = sm.Logit(y, X1).fit(disp=0).predict(X1)
    # 固定阈值 = 原评分 top 20%
    thr = np.quantile(pred0, 0.80)
    hi0 = pred0 >= thr          # 原评分高风险
    hi1 = pred1 >= thr          # 重校准后高风险（同一阈值）
    poor = (D["wq"] == 1).values
    rich = (D["wq"] == 5).values
    n_poor_up = ((~hi0) & hi1 & poor).sum()   # 穷人被升级
    n_poor_all = poor.sum()
    n_rich_down = (hi0 & ~hi1 & rich).sum()   # 富人被降级
    n_rich_all = rich.sum()
    return {
        "name": name, "n": len(D),
        "穷升级": f"{n_poor_up}/{n_poor_all}={n_poor_up/n_poor_all*100:.1f}%",
        "富降级": f"{n_rich_down}/{n_rich_all}={n_rich_down/n_rich_all*100:.1f}%",
        "穷人低估量(死亡概率差)": pred1[poor].mean() - pred0[poor].mean(),
    }


cohorts = [
    ("CHARLS", load_charls(), "wealth_total"),
    ("ELSA", load_elsa(), "wealth_total"),
    ("SHARE", load_share(), "wealth_total"),
    ("HRS", load_hrs(), "wealth_total"),
]
print("=== 部署后果反事实（固定原评分top20%阈值，重校准=加θ·disadv）===")
print(f"{'库':<8}{'n':>7}{'穷升级进高风险':>16}{'富降级出高风险':>16}{'穷人死亡概率被低估':>16}")
for name, D, wc in cohorts:
    r = recalibrate(name, D, wc)
    print(f"{r['name']:<8}{r['n']:>7}{r['穷升级']:>16}{r['富降级']:>16}{r['穷人低估量(死亡概率差)']:>16.3f}")
