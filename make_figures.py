# -*- coding: utf-8 -*-
"""生成 4 张黑白图（J Gerontol A 可打印）：Fig1 θ森林图 / Fig2 校准曲线 / Fig3 握力锚点 / Fig4 修复+部署。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from analysis_theta_v2 import BASE, load_charls, load_elsa, load_share, load_hrs, load_klosa, build_fi12

OUT = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\figures"
os.makedirs(OUT, exist_ok=True)

# 穷=虚线空心, 富=实线实心（黑白可打印）
POOR = dict(color="black", linestyle="--", marker="o", mfc="white", mec="black")
RICH = dict(color="black", linestyle="-", marker="o", mfc="black", mec="black")


# ---------- Fig 1: θ 森林图 ----------
def fig1():
    cohorts = ["CHARLS", "ELSA", "SHARE", "HRS"]
    theta = np.array([0.090, 0.135, 0.121, 0.139])
    se = np.array([0.034, 0.028, 0.020, 0.015])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ys = np.arange(len(cohorts))[::-1]
    for i, (t, s) in enumerate(zip(theta, se)):
        ax.plot([t - 1.96 * s, t + 1.96 * s], [ys[i], ys[i]], color="black", lw=2)
        ax.plot(t, ys[i], "s", color="black", ms=7)
    # meta 菱形
    ax.plot(0.129, -0.6, "D", color="black", ms=9)
    ax.plot([0.102, 0.155], [-0.6, -0.6], color="black", lw=3)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_yticks(list(ys) + [-0.6])
    ax.set_yticklabels(cohorts + ["Pooled (RE)"])
    ax.set_xlabel("Socioeconomic calibration offset θ (per wealth quintile)")
    ax.set_xlim(-0.05, 0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig1_forest.png"), dpi=300)
    plt.close(fig)


# ---------- Fig 2: 校准曲线（核心） ----------
def calibration_curve(D, wc, ax, label, style):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["wq"].notna()
    D = D[m]; y = y[m].astype(float)
    Xb = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"]}))
    pred = sm.Logit(y, Xb).fit(disp=0).predict(Xb)
    sub = (D["wq"] == (1 if "poor" in label else 5)).values
    pred_s, y_s = pred[sub], y[sub]
    # 十分位 observed vs predicted
    dec = pd.qcut(pd.Series(pred_s).rank(method="first"), 10, labels=False)
    obs, prd = [], []
    for d in range(10):
        msk = dec == d
        if msk.sum() > 5:
            obs.append(y_s[msk].mean()); prd.append(pred_s[msk].mean())
    ax.plot(prd, obs, **style, ms=5, label=label)


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (name, D, wc) in zip(axes, [("HRS", load_hrs(), "wealth_total"), ("ELSA", load_elsa(), "wealth_total")]):
        calibration_curve(D, wc, ax, "Poorest Q1", POOR)
        calibration_curve(D, wc, ax, "Wealthiest Q5", RICH)
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
        ax.set_xlabel("Predicted mortality"); ax.set_title(name)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed mortality")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig2_calibration.png"), dpi=300)
    plt.close(fig)


# ---------- Fig 3: 握力锚点 ----------
def fig3():
    # FI 三档内 穷vs富 握力(kg)
    data = {
        "CHARLS": [(27.4, 32.8), (26.5, 29.5), (25.2, 28.1)],
        "SHARE": [(32.5, 35.6), (28.9, 33.9), (25.9, 31.2)],
        "KLoSA": [(24.4, 27.4), (20.9, 25.7), (18.4, 23.5)],
    }
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharey=True)
    for ax, (name, rows) in zip(axes, data.items()):
        x = np.arange(3)
        poor = [r[0] for r in rows]; rich = [r[1] for r in rows]
        ax.plot(x, poor, **POOR, label="Poorest Q1")
        ax.plot(x, rich, **RICH, label="Wealthiest Q5")
        ax.set_xticks(x); ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("FI tertile"); ax.set_title(name)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Grip strength (kg)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig3_grip.png"), dpi=300)
    plt.close(fig)


# ---------- Fig 4: 修复 + 部署（双面板） ----------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cohorts = ["CHARLS", "ELSA", "SHARE", "HRS"]
    # A: 缺口闭合
    blind = [0.098, 0.196, 0.234, 0.218]
    fixed = [-0.068, -0.026, 0.027, -0.004]
    x = np.arange(4)
    axes[0].bar(x - 0.18, blind, width=0.36, color="white", edgecolor="black", hatch="//", label="SES-blind")
    axes[0].bar(x + 0.18, fixed, width=0.36, color="black", label="Recalibrated")
    axes[0].axhline(0, color="gray", ls=":", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cohorts)
    axes[0].set_ylabel("Poor-quintile calibration intercept")
    axes[0].set_title("A. Recalibration closes the gap")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    # B: 再分类
    up = [6.1, 7.8, 7.2, 5.5]; down = [3.9, 4.6, 4.5, 3.1]
    axes[1].bar(x - 0.18, up, width=0.36, color="black", label="Poor upgraded")
    axes[1].bar(x + 0.18, down, width=0.36, color="white", edgecolor="black", hatch="//", label="Rich downgraded")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cohorts)
    axes[1].set_ylabel("Reclassified (%)")
    axes[1].set_title("B. Reclassification at fixed threshold")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig4_recalibration.png"), dpi=300)
    plt.close(fig)


fig1(); fig2(); fig3(); fig4()
print("已生成 Fig1-4 到", OUT)
