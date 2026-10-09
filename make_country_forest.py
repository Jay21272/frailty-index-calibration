# -*- coding: utf-8 -*-
"""生成 Supplementary 图：SHARE 12 国 θ 森林图（标注国家名）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\figures"
os.makedirs(OUT, exist_ok=True)

# country 编码 -> 国家名（UN ISO 3166-1 numeric）
COUNTRY = {11: "Austria", 12: "Germany", 13: "Sweden", 14: "Netherlands", 15: "Spain",
           16: "Italy", 17: "France", 18: "Denmark", 19: "Greece", 20: "Switzerland",
           23: "Belgium", 25: "Israel"}
# (country, θ, SE)
rows = [(18, 0.257, 0.083), (19, 0.174, 0.068), (13, 0.165, 0.065), (14, 0.128, 0.068),
        (15, 0.123, 0.067), (20, 0.102, 0.144), (25, 0.100, 0.072), (17, 0.089, 0.063),
        (11, 0.071, 0.078), (16, 0.011, 0.076), (12, -0.004, 0.078), (23, -0.040, 0.069)]
rows = sorted(rows, key=lambda x: x[1], reverse=True)

names = [COUNTRY[c] for c, t, s in rows]
theta = np.array([t for c, t, s in rows])
se = np.array([s for c, t, s in rows])

fig, ax = plt.subplots(figsize=(5, 5))
ys = np.arange(len(rows))[::-1]
for i, (t, s) in enumerate(zip(theta, se)):
    ax.plot([t - 1.96 * s, t + 1.96 * s], [ys[i], ys[i]], color="black", lw=2)
    ax.plot(t, ys[i], "s", color="black", ms=6)
ax.axvline(0, color="gray", ls=":", lw=1)
# 合并 θ = 0.098
ax.plot(0.098, -0.8, "D", color="black", ms=8)
ax.plot([0.098 - 1.96 * 0.02, 0.098 + 1.96 * 0.02], [-0.8, -0.8], color="black", lw=3)
ax.set_yticks(list(ys) + [-0.8])
ax.set_yticklabels(names + ["Pooled"])
ax.set_xlabel("Disadvantage gradient θ (within-country wealth quintiles)")
ax.set_xlim(-0.2, 0.45)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "FigS1_country_forest.png"), dpi=300)
plt.close(fig)
print("已生成 FigS1_country_forest")
