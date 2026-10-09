# -*- coding: utf-8 -*-
"""统一 12 项 FI(慢病7+ADL+IADL+BMI低+BMI高+SRH差) → θ 参考模型 logit=FI+age+sex+θ·wealth。
先诊断各库编码方向，再跑 θ。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测"
DISEASES = ["hibp", "diab", "cancr", "lung", "heart", "strok", "arth"]


def load_charls():
    D = pd.read_csv(BASE + r"\charls_w1_corrected.csv", low_memory=False)
    D["adl_any"] = (D["adl_count"] >= 1)
    iadl = [c for c in ["iadl_moneya", "iadl_medsa", "iadl_shopa", "iadl_mealsa", "iadl_housewka"] if c in D.columns]
    D["iadl_any"] = D[iadl].isin([1]).any(axis=1)
    D["death"] = D["died_by_w3"]  # ~4年
    return D


def load_elsa():
    D = pd.read_csv(BASE + r"\elsa_w1_corrected.csv", low_memory=False)
    D["adl_any"] = (D["adl_count"] >= 1)
    iadl = [c for c in ["iadl_moneya", "iadl_medsa", "iadl_shopa", "iadl_phonea", "iadl_mealsa", "iadl_housewka"] if c in D.columns]
    D["iadl_any"] = D[iadl].isin([1]).any(axis=1)
    D["death"] = D["died_by_w3"]  # ~4年（统一窗口）
    return D


def load_share():
    D = pd.read_csv(BASE + r"\share_w1_corrected.csv", low_memory=False)
    D["adl_any"] = (D["adl_count"] >= 1)
    iadl = [c for c in ["iadl_moneya", "iadl_medsa", "iadl_shopa", "iadl_phonea", "iadl_mealsa", "iadl_housewka"] if c in D.columns]
    D["iadl_any"] = D[iadl].isin([1]).any(axis=1)
    D["death"] = D["died_by_w3"]  # ~4年（统一窗口）
    return D


def load_klosa():
    D = pd.read_csv(BASE + r"\klosa_w1_core_v2.csv", low_memory=False)
    D["adl_any"] = (D["adl_count"] >= 1)
    D["iadl_any"] = (D["iadl_count"] >= 1)
    # 精确 5 年：基线 2006，death_year <= 2011
    D["death"] = D["died"] & (D["death_year"] <= 2011)
    return D


def load_hrs():
    D = pd.read_csv(BASE + r"\hrs_60plus_baseline_v3.csv", low_memory=False)
    D["adl_any"] = (D["adl_count"] >= 1)
    D["iadl_any"] = (D["iadl_count"] >= 1)
    D["death_4y"] = D["died"] & (D["death_year"] <= D["base_year"] + 4)
    D["death"] = D["death_4y"]  # 精确4年（统一窗口）
    return D


def clean_disease(s):
    # 1=是(缺陷1), 0或5=否(缺陷0), 负值/其他=NaN
    out = pd.Series(np.nan, index=s.index)
    out[s == 1] = 1.0
    out[(s == 0) | (s == 5)] = 0.0
    return out


def build_fi12(D):
    it = {}
    for c in DISEASES:
        if c in D.columns:
            it[c] = clean_disease(D[c])
    it["adl"] = D["adl_any"].astype(float).where(D["adl_any"].notna())
    it["iadl"] = D["iadl_any"].astype(float).where(D["iadl_any"].notna())
    it["bmi_low"] = (D["bmi"] < 18.5).where(D["bmi"].notna()).astype(float)
    it["bmi_high"] = (D["bmi"] >= 30).where(D["bmi"].notna()).astype(float)
    it["srh"] = (D["self_rated_health"].isin([4, 5])).where(D["self_rated_health"].isin([1, 2, 3, 4, 5])).astype(float)
    dd = pd.DataFrame(it)
    denom = dd.notna().sum(axis=1)
    n_possible = int((dd.notna().sum(axis=0) > 0).sum())  # 非全缺失的列数(ELSA缺BMI=10)
    fi = (dd.sum(axis=1) / denom).where(denom >= int(n_possible * 0.8))
    return fi


def run(name, D, wealth_col):
    D = D[D["age"] >= 60].copy()
    D["fi"] = build_fi12(D)
    y = pd.to_numeric(D["death"], errors="coerce")
    has_w = wealth_col in D.columns and D[wealth_col].notna().sum() > 200
    r = {"n": len(D), "FI": D["fi"].median(), "财富非空": D[wealth_col].notna().sum() if has_w else 0}
    if has_w:
        w = D[wealth_col].where(D[wealth_col] >= 0)  # 排除 Gateway 负值缺失码(-18/-16/-4等)
        D["wq"] = pd.qcut(w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(float)
        D["disadv"] = 5 - D["wq"]  # 0=最富 4=最穷
        m = D["fi"].notna() & D["age"].notna() & D["gender"].notna() & y.notna() & D["disadv"].notna()
        D = D[m]; y = y[m]
        X = sm.add_constant(pd.DataFrame({"FI": D["fi"], "age": D["age"], "sex": D["gender"], "disadv": D["disadv"]}))
        mdl = sm.Logit(y, X).fit(disp=0)
        r["θ(disadv)"] = mdl.params["disadv"]
        r["θ_SE"] = mdl.bse["disadv"]
        r["θ_p"] = mdl.pvalues["disadv"]
        # O/E 穷(Q1) vs 富(Q5)
        pred = mdl.predict(X)
        poor = D["wq"] == 1; rich = D["wq"] == 5
        r["O/E穷"] = y[poor.values].mean() / pred[poor.values].mean()
        r["O/E富"] = y[rich.values].mean() / pred[rich.values].mean()
    return r


cohorts = [
    ("CHARLS", load_charls(), "wealth_total", "self_rated_health"),
    ("ELSA", load_elsa(), "wealth_total", "self_rated_health"),
    ("SHARE", load_share(), "wealth_total", "self_rated_health"),
    ("KLoSA", load_klosa(), "wealth_total", "self_rated_health"),
    ("HRS", load_hrs(), "wealth_total", "self_rated_health"),
]

print("=== SRH 方向诊断 ===")
for name, D, wc, srh in cohorts:
    print(f"{name}: SRH 取值 {sorted(D[srh].dropna().unique())[:8]}, 慢病hibp取值 {sorted(D['hibp'].dropna().unique())[:6]}")

print("\n=== θ 结果（12项FI + 参考模型 logit=FI+age+sex+θ·disadv）===")
print(f"{'库':<8}{'n':>7}{'FI':>6}{'财富非空':>9}{'θ':>9}{'SE':>8}{'p':>8}{'O/E穷':>8}{'O/E富':>8}")
for name, D, wc, srh in cohorts:
    try:
        r = run(name, D, wc)
        def f(x, nd=2):
            return "  --" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:>8.{nd}f}"
        print(f"{name:<8}{r['n']:>7}{r['FI']:>6.3f}{r['财富非空']:>9}{f(r['θ(disadv)'],3):>9}{f(r['θ_SE'],3):>8}{f(r['θ_p'],3):>8}{f(r['O/E穷']):>8}{f(r['O/E富']):>8}")
    except Exception as e:
        print(f"{name:<8} 失败: {str(e)[:80]}")
