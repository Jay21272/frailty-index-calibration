# -*- coding: utf-8 -*-
"""生成 Fig5：重校准前后穷组校准曲线对比（SES盲 vs 加SES校正），展示曲线拉平。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from analysis_theta_v2 import load_hrs, load_share, build_fi12

OUT = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\figures"
os.makedirs(OUT, exist_ok=True)


def recal_curve(D, wc, ax, poor=True):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["wq"].notna()
    D = D[m]; y = y[m].astype(float)
    sub = (D["wq"] == 1).values if poor else (D["wq"] == 5).values
    # SES盲模型
    Xb = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    pb = sm.Logit(y, Xb).fit(disp=0).predict(Xb)
    # 重校准模型
    Xr = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"], "disadv": D["disadv"]}))
    pr = sm.Logit(y, Xr).fit(disp=0).predict(Xr)
    for pred, style, lab in [(pb, dict(color="black", linestyle="--", marker="o", mfc="white", mec="black"), "SES-blind"),
                             (pr, dict(color="black", linestyle="-", marker="o", mfc="black", mec="black"), "SES-adjusted")]:
        ps, ys = pred[sub], y[sub]
        dec = pd.qcut(pd.Series(ps).rank(method="first"), 8, labels=False)
        obs, prd = [], []
        for d in range(8):
            mk = dec == d
            if mk.sum() > 5:
                obs.append(ys[mk].mean()); prd.append(ps[mk].mean())
        ax.plot(prd, obs, **style, ms=5, label=lab)


fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, (name, D, wc) in zip(axes, [("HRS", load_hrs(), "wealth_total"), ("SHARE", load_share(), "wealth_total")]):
    recal_curve(D, wc, ax, poor=True)
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.set_xlabel("Predicted mortality (poorest quintile)"); ax.set_title(name)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Observed mortality")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig5_recalibration_curve.png"), dpi=300)
plt.close(fig)
print("已生成 Fig5")
