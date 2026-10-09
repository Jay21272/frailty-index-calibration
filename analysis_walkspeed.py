# -*- coding: utf-8 -*-
"""步速(客观身体功能)作为独立变量加入死亡模型，检验 θ 是否衰减。
步速=走固定距离计时(秒)，清洗 0.5-20 秒。复用现有 CSV 的 FI 逻辑。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_theta_v2 import BASE, load_charls, load_elsa, load_share, build_fi12

# 从原始数据提取步速
def get_wspeed(path, cols):
    df = pd.read_stata(path, columns=cols + ["ID"] if "ID" in cols else cols, convert_categoricals=False)
    return df


def run_ws(name, D, ws):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    D["wspeed"] = ws
    # 清洗：0.5-20 秒保留
    D["wspeed"] = D["wspeed"].where(D["wspeed"].between(0.5, 20))
    w = D["wealth_total"].where(D["wealth_total"] >= 0)
    D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    D["disadv"] = 5 - D["wq"]
    m = (D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna()
         & D["disadv"].notna() & D["wspeed"].notna())
    D = D[m]; y = y[m]
    X0 = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"], "disadv": D["disadv"]}))
    m0 = sm.Logit(y, X0).fit(disp=0)
    X1 = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"], "disadv": D["disadv"], "wspeed": D["wspeed"]}))
    m1 = sm.Logit(y, X1).fit(disp=0)
    t0, t1 = m0.params["disadv"], m1.params["disadv"]
    return {
        "name": name, "n": len(D), "ws_mean": D["wspeed"].mean(),
        "θ_ref": t0, "θ_adj": t1, "Δθ": t1 - t0,
        "衰减%": (t0 - t1) / t0 * 100 if t0 != 0 else None,
        "ws_coef": m1.params["wspeed"], "ws_p": m1.pvalues["wspeed"],
    }


# 步速原始数据（ID 去前导 0 转 float 对齐 CSV 的 pid）
def clean_key(s):
    return pd.to_numeric(s.astype(str).str.lstrip("0"), errors="coerce")

charls_raw = pd.read_stata(r"G:\CHARLS\H_CHARLS_D_Data.dta",
                           columns=["ID", "r1wspeed"], convert_categoricals=False)
charls_raw["key"] = clean_key(charls_raw["ID"])
charls_raw = charls_raw.set_index("key")["r1wspeed"]

share_raw = pd.read_stata(r"G:\SHARE\GH_SHARE_g.dta",
                          columns=["mergeid", "r1wspeed"], convert_categoricals=False)
share_raw["key"] = clean_key(share_raw["mergeid"])
share_raw = share_raw.set_index("key")["r1wspeed"]

elsa_raw = pd.read_stata(r"G:\ELSA-5050\gh_elsa_h.dta",
                         columns=["idauniq", "r1wspeed"], convert_categoricals=False)
elsa_raw["key"] = clean_key(elsa_raw["idauniq"])
elsa_raw = elsa_raw.set_index("key")["r1wspeed"]

# CSV（有 pid 列）
for name, load, raw, idcol in [
    ("CHARLS", load_charls, charls_raw, "pid"),
    ("ELSA", load_elsa, elsa_raw, "pid"),
    ("SHARE", load_share, share_raw, "pid"),
]:
    D = load()
    pid = pd.to_numeric(D[idcol], errors="coerce")
    ws = pid.map(raw)
    print(f"[{name}] 步速对齐非空: {ws.notna().sum()}/{len(ws)}")
    r = run_ws(name, D, ws)
    print(f"  {name}: n={r['n']}, 步速均值={r['ws_mean']:.2f}s, θ_ref={r['θ_ref']:.3f}, "
          f"θ_adj={r['θ_adj']:.3f}, Δθ={r['Δθ']:.3f}, 衰减{r['衰减%']:.1f}%, 步速β={r['ws_coef']:.3f} p={r['ws_p']:.3f}")
