# -*- coding: utf-8 -*-
"""修复(SES分层重校准) + 留出验证：加disadv校正项能否闭合穷组的校准缺口(截距回到0)。
发现偏差→修复→修复后更准。5折交叉验证估计校准缺口闭合量。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import (BASE, load_charls, load_elsa, load_share, load_hrs, build_fi12)


def fix_cohort(name, D, wc):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    w = D[wc].where(D[wc] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
    D = D[m].reset_index(drop=True); y = y[m].values.astype(float)
    # 5折：训练集fit，测试集评估穷组校准缺口(截距)在「SES盲 vs 重校准」下的变化
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 5, len(D))
    gaps_blind, gaps_fix = [], []
    for k in range(5):
        tr, te = fold != k, fold == k
        # SES盲模型(训练)
        Xb_tr = sm.add_constant(pd.DataFrame({"FI": D["fi"][tr], "age": D["age"][tr], "sex": D["gender"][tr]}))
        mb = sm.Logit(y[tr], Xb_tr).fit(disp=0)
        # 重校准模型(训练,含disadv)
        Xf_tr = sm.add_constant(pd.DataFrame({"FI": D["fi"][tr], "age": D["age"][tr], "sex": D["gender"][tr], "disadv": D["disadv"][tr]}))
        mf = sm.Logit(y[tr], Xf_tr).fit(disp=0)
        # 测试集预测(两个模型)
        Xb_te = sm.add_constant(pd.DataFrame({"FI": D["fi"][te], "age": D["age"][te], "sex": D["gender"][te]}))
        pb = mb.predict(Xb_te)
        Xf_te = sm.add_constant(pd.DataFrame({"FI": D["fi"][te], "age": D["age"][te], "sex": D["gender"][te], "disadv": D["disadv"][te]}))
        pf = mf.predict(Xf_te)
        # 测试集穷组截距(两个模型)
        poor_te = (D["wq"][te] == 1).values
        yt = y[te][poor_te]
        for pred_te, store in [(pb, gaps_blind), (pf, gaps_fix)]:
            lob = np.log(pred_te[poor_te] / (1 - pred_te[poor_te]))
            Xa = np.ones((yt.shape[0], 1))
            a = sm.GLM(yt, Xa, offset=lob, family=sm.families.Binomial()).fit().params[0]
            store.append(a)
    return name, {
        "穷组截距(SES盲)": np.mean(gaps_blind),
        "穷组截距(重校准)": np.mean(gaps_fix),
        "缺口闭合": np.mean(gaps_blind) - np.mean(gaps_fix),
    }


cohorts = [
    ("CHARLS", load_charls(), "wealth_total"),
    ("ELSA", load_elsa(), "wealth_total"),
    ("SHARE", load_share(), "wealth_total"),
    ("HRS", load_hrs(), "wealth_total"),
]
print("=== 修复验证：5折CV，穷组校准截距在SES盲 vs 重校准(加disadv)下的变化 ===")
print("（重校准后截距接近0 = 缺口闭合 = 修复有效）")
print(f"{'库':<8}{'盲模型穷组截距':>14}{'重校准后截距':>14}{'缺口闭合':>10}")
for name, D, wc in cohorts:
    n, r = fix_cohort(name, D, wc)
    print(f"{n:<8}{r['穷组截距(SES盲)']:>14.3f}{r['穷组截距(重校准)']:>14.3f}{r['缺口闭合']:>10.3f}")
