# -*- coding: utf-8 -*-
"""真衰弱锚点(握力)：裁决「偏差 vs 真相」。
在 FI 相同分档内，穷人是否同时握力更差(独立客观身体功能指标)。
grip = FI + age + sex + θ·disadv，θ显著负=穷人相同FI下更衰弱=FI少算穷人真实负荷(偏差坐实)。
3库有grip(CHARLS/SHARE/KLoSA)，ELSA/HRS无。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (BASE, load_charls, load_share, load_klosa, build_fi12)


def anchor(name, D, wc):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    if "grip" not in D.columns or D["grip"].notna().sum() < 200:
        return None
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & D["disadv"].notna() & D["grip"].notna()
    D = D[m]
    X = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"], "disadv": D["disadv"]}))
    mdl = sm.OLS(D["grip"].values, X).fit()
    # 也看：FI 分档内穷vs富握力差（直观）
    D["fi_q"] = pd.qcut(D["fi"].rank(method="first"), 3, labels=["低", "中", "高"])
    rows = []
    for fq in ["低", "中", "高"]:
        g_poor = D.loc[(D["fi_q"] == fq) & (D["wq"] == 1), "grip"].mean()
        g_rich = D.loc[(D["fi_q"] == fq) & (D["wq"] == 5), "grip"].mean()
        rows.append((fq, g_poor, g_rich, g_rich - g_poor))
    return {
        "name": name, "n": len(D),
        "θ_grip": mdl.params["disadv"], "θ_se": mdl.bse["disadv"], "θ_p": mdl.pvalues["disadv"],
        "fi_strata": rows,
    }


cohorts = [
    ("CHARLS", load_charls(), "wealth_total"),
    ("SHARE", load_share(), "wealth_total"),
    ("KLoSA", load_klosa(), "wealth_total"),
]
print("=== 真衰弱锚点(握力)：grip = FI+age+sex+θ·disadv ===")
print("（θ<0 = 相同FI下穷人握力更低 = FI少算穷人真实身体负荷 = 偏差坐实）")
print(f"{'库':<8}{'n':>7}{'θ_grip':>9}{'SE':>7}{'p':>8}")
for name, D, wc in cohorts:
    r = anchor(name, D, wc)
    if r is None:
        print(f"{name:<8} 无握力数据"); continue
    print(f"{r['name']:<8}{r['n']:>7}{r['θ_grip']:>9.2f}{r['θ_se']:>7.2f}{r['θ_p']:>8.3f}")

print("\n=== FI 三档内 穷vs富 握力(kg) ===")
for name, D, wc in cohorts:
    r = anchor(name, D, wc)
    if r is None: continue
    print(f"{r['name']}: " + " | ".join(f"{fq}:穷{gp:.1f}/富{gr:.1f}(差{gr-gp:+.1f})" for fq, gp, gr, diff in r["fi_strata"]))
