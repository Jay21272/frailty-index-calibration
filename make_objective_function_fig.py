# -*- coding: utf-8 -*-
"""Fig6: 握力衰减 θ（%）vs FI→握力 β（kg）跨库散点，展示「FI 捕获握力越强→握力衰减 θ 越小」。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\figures"
os.makedirs(OUT, exist_ok=True)

# (FI→握力 β, 握力衰减 θ%, 队列名, 是否 w2 滞后)
pts = [
    (-7.3, 48.4, "CHARLS", False),
    (-9.7, 20.6, "ELSA", True),
    (-13.4, 7.4, "SHARE", False),
]

fig, ax = plt.subplots(figsize=(5, 3.8))
for x, y, name, lag in pts:
    ax.plot(x, y, "s", color="black", ms=8, mfc="white" if lag else "black", mec="black")
    dy = -3 if name != "SHARE" else 3
    ax.annotate(name + ("†" if lag else ""), (x, y), textcoords="offset points",
                xytext=(6, dy), fontsize=9)
# 趋势参考线（描述性，非统计）
x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
coef = np.polyfit(x, y, 1)
xs = np.linspace(-14, -6, 20)
ax.plot(xs, coef[0] * xs + coef[1], color="gray", ls=":", lw=1)

ax.set_xlabel("FI → grip strength β (kg per FI unit, more negative = FI captures grip more)")
ax.set_ylabel("Attenuation of θ after adding grip strength (%)")
ax.set_xlim(-14.5, -5.5)
ax.set_ylim(0, 55)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.03, 0.03, "† ELSA grip measured at wave 2", transform=ax.transAxes, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig6_grip_attenuation.png"), dpi=300)
plt.close(fig)
print("已生成 Fig6_grip_attenuation.png")
