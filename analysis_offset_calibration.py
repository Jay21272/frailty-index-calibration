# -*- coding: utf-8 -*-
"""θ 的严格 offset-calibration 定义：
1) 参考模型 logit=FI+age+sex 拟合，取线性预测子 η（含截距）
2) offset 模型 logit=α + offset(η) + θ·disadv（η 系数锁 1，只估 α 和 θ）
θ = 每五分位校准偏移量 = calibration-in-the-large 跨财富五分位的斜率。
与联合估计(FI/age/sex一起重估)对比，并重跑 meta(HKSJ)。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from analysis_theta_v2 import (BASE, load_charls, load_elsa, load_share, load_hrs, build_fi12)


def offset_theta(D, wc):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    D = D[m].reset_index(drop=True); y = y[m].values.astype(float)
    # 参考模型（线性预测子）
    Xr = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    mr = sm.Logit(y, Xr).fit(disp=0)
    eta = Xr @ mr.params  # 线性预测子（含截距）
    # offset 模型
    Xo = sm.add_constant(pd.DataFrame({"disadv": D["disadv"]}))
    mo = sm.Logit(y, Xo, offset=eta).fit(disp=0)
    theta = mo.params["disadv"]; se = mo.bse["disadv"]; p = mo.pvalues["disadv"]
    alpha = mo.params["const"]
    return theta, se, p, alpha, len(D)


cohorts = [
    ("CHARLS", load_charls(), "wealth_total"),
    ("ELSA", load_elsa(), "wealth_total"),
    ("SHARE", load_share(), "wealth_total"),
    ("HRS", load_hrs(), "wealth_total"),
]
print("=== offset-calibration θ vs 联合估计 θ ===")
print(f"{'库':<8}{'offsetθ':>9}{'SE':>8}{'p':>8}{'截距α':>8}{'n':>7}")
rows = []
for name, D, wc in cohorts:
    th, se, p, a, n = offset_theta(D, wc)
    rows.append((name, th, se))
    print(f"{name:<8}{th:>9.4f}{se:>8.4f}{p:>8.4f}{a:>8.4f}{n:>7}")

# meta（随机效应 D-L + HKSJ）
ths = np.array([r[1] for r in rows])
ses = np.array([r[2] for r in rows])
k = len(ths)
w = 1 / ses**2
fe = np.sum(w * ths) / np.sum(w)
Q = np.sum(w * (ths - fe)**2)
tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
wre = 1 / (ses**2 + tau2)
re = np.sum(wre * ths) / np.sum(wre)
se_re = np.sqrt(1 / np.sum(wre))
I2 = max(0.0, (Q - (k - 1)) / Q) * 100
# HKSJ
se_hksj = np.sqrt(np.sum(wre * (ths - re)**2) / (k - 1) / np.sum(wre))
tcrit = stats.t.ppf(0.975, k - 1)
ci_hksj = (re - tcrit * se_hksj, re + tcrit * se_hksj)
ci_dl = (re - 1.96 * se_re, re + 1.96 * se_re)
print(f"\n=== meta ===")
print(f"固定效应 θ = {fe:.4f}")
print(f"随机效应 θ = {re:.4f} (SE {se_re:.4f})")
print(f"DL 95% CI = [{ci_dl[0]:.3f}, {ci_dl[1]:.3f}]")
print(f"HKSJ 95% CI = [{ci_hksj[0]:.3f}, {ci_hksj[1]:.3f}] (df={k-1})")
print(f"I² = {I2:.1f}%, τ² = {tau2:.5f}")
print(f"OR 四档(最穷vs最富) = exp(4θ) = {np.exp(4*re):.3f}")
