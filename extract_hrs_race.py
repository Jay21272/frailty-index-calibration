# -*- coding: utf-8 -*-
"""补抽 HRS 种族/族裔（时不变），合并进 hrs_60plus_baseline_v3.csv，构建四分类 race_eth。"""
import pyreadstat
import pandas as pd
import numpy as np
import re

PATH = r"G:\hrs data\randhrs1992_2022v1.sas7bdat"
CSV = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\hrs_60plus_baseline_v3.csv"

# 1) 找种族相关列名
_, meta = pyreadstat.read_sas7bdat(PATH, metadataonly=True)
cols = list(meta.column_names)
race_cols = [c for c in cols if re.search(r"RACE|HISPAN|ETHNIC|ORIGIN", c, re.I)]
print("RACE/HISPAN/ETHNIC 候选列:", race_cols)

# 2) 抽取时不变种族变量（HHIDPN 作主键）
want = [c for c in ["HHIDPN", "RARACEM", "RAHISPAN"] if c in cols]
print("实际抽取列:", want)
df, _ = pyreadstat.read_sas7bdat(PATH, usecols=want)
df.columns = [c.lower() for c in df.columns]

pid = df["hhidpn"].astype(str)
race = pd.to_numeric(df["raracem"], errors="coerce") if "raracem" in df.columns else pd.Series(np.nan, index=df.index)
hispan = pd.to_numeric(df["rahispan"], errors="coerce") if "rahispan" in df.columns else pd.Series(np.nan, index=df.index)

print("\nRARACEM 取值分布:")
print(race.value_counts(dropna=False).sort_index())
print("\nRAHISPAN 取值分布:")
print(hispan.value_counts(dropna=False).sort_index())

# 3) 四分类 race_eth：0=非西裔白人 1=非西裔黑人 2=西裔(任何种族) 3=非西裔其他
race_eth = pd.Series(np.nan, index=df.index)
race_eth[(race == 1) & (hispan == 0)] = 0
race_eth[(race == 2) & (hispan == 0)] = 1
race_eth[hispan == 1] = 2
race_eth[(race == 3) & (hispan == 0)] = 3
race_eth[(race == 1) & (hispan.isna())] = np.nan  # hispan 缺失不硬判

# 4) 合并
D = pd.read_csv(CSV, low_memory=False)
D["pid"] = D["pid"].astype(str)
race_map = dict(zip(pid, race_eth))
D["race_eth"] = D["pid"].map(race_map)

print("\n合并后 race_eth 分布（60+基线样本）:")
print(D["race_eth"].value_counts(dropna=False).sort_index())

D.to_csv(CSV, index=False)
print(f"\n已保存回 {CSV}（新增列 race_eth，共 {D.shape[1]} 列）")
