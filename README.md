# Analysis code for "The frailty index under-predicts mortality in socioeconomically disadvantaged older adults"

This repository contains the analysis code for a cross-national audit of whether a 12-item frailty index (FI) is calibrated equitably across socioeconomic strata.

## Overview

We audited a fixed 12-item FI as a predictor of mortality (and, secondarily, incident disability) across four harmonized population cohorts of adults aged 60 and older: CHARLS (China), ELSA (England), SHARE (continental Europe), and HRS (United States). The primary quantity of interest is the **socioeconomic calibration offset θ** — the per-quintile shift in predicted mortality risk that must be added to the reference model (FI + age + sex) to correct the under-prediction of mortality among wealth-disadvantaged older adults.

- θ is estimated by **offset calibration**: the reference model's linear predictor η is fixed (entered as an offset with coefficient 1), and only the intercept α and θ·disadvantage are estimated. This makes θ a *calibration quantity* (the slope of calibration-in-the-large across wealth quintiles), not an association coefficient.
- Attenuation of θ by grip strength, walking speed, and social factors is reported from **jointly re-fitted** models (the conventional "does adding covariate X change θ" question).

## Dependencies

Python 3.12 with:

```
numpy
pandas
statsmodels
scipy
matplotlib
pyreadstat      # for reading Stata (.dta) and SAS (.sas7bdat) files
scikit-learn    # for AUC
python-docx      # for generating manuscript/supplementary .docx
```

Install with `pip install -r requirements.txt`.

## Data requirements

The raw data are **not** included in this repository. They must be obtained from the respective cohort consortia:

| Cohort | Source | URL |
|---|---|---|
| CHARLS | China Health and Retirement Longitudinal Study | http://charls.pku.edu.cn |
| ELSA | English Longitudinal Study of Ageing | https://www.elsa-project.ac.uk |
| SHARE | Survey of Health, Ageing and Retirement in Europe | https://share-eric.eu |
| HRS | Health and Retirement Study (RAND version) | https://hrs.isr.umich.edu |
| KLoSA | Korean Longitudinal Study of Ageing | https://survey.keis.or.kr |
| NHANES | National Health and Nutrition Examination Survey | https://www.cdc.gov/nchs/nhanes |

**Raw data paths are hardcoded** in the scripts (e.g., `G:\CHARLS\H_CHARLS_D_Data.dta`, `G:\hrs data\randhrs1992_2022v1.sas7bdat`, `G:\NHANES\...`). Edit these paths before running.

## Analysis pipeline

1. **Extraction** (`extract_*.py`) — reads the raw cohort files and writes harmonized CSV files (`*_w1_corrected.csv`, `hrs_60plus_baseline_v3.csv`).
2. **Analysis** (`analysis_*.py`) — computes θ, the calibration audit, attenuation, and robustness checks from the CSVs.
3. **Figures** (`make_*.py`) — renders the publication figures (black-and-white, print-ready PNG).
4. **Manuscript** (`make_docx.py`, `make_supplementary_docx.py`) — assembles the manuscript and supplementary material as `.docx`.

## Script index

### Extraction

| Script | Produces |
|---|---|
| `extract_charls_v5.py` | `charls_w1_corrected.csv` |
| `extract_elsa.py` | `elsa_w1_corrected.csv` |
| `extract_share.py` | `share_w1_corrected.csv` |
| `extract_hrs_v3.py` | `hrs_60plus_baseline_v3.csv` |
| `extract_hrs_race.py` | adds the `race_eth` column to the HRS CSV |
| `extract_klosa_v2.py` | `klosa_w1_core_v2.csv` |

### Core analysis

| Script | Result |
|---|---|
| `analysis_theta_v2.py` | shared load functions, 12-item FI construction, and the joint-estimation θ (used as a comparison) |
| `analysis_offset_calibration.py` | **primary θ** (offset calibration) per cohort + pooled meta-analysis (Figure 1, Table 2) |
| `analysis_calibration.py` | calibration-in-the-large (intercept) and slope within poorest/wealthiest quintiles (Figure 2, Table 3) |
| `analysis_fix.py` | 5-fold cross-validated recalibration (gap closure; Figure 4, Table 3) |
| `analysis_anchor.py` | grip strength as an objective physical-function anchor (Figure 3) |
| `analysis_recalibration.py` | reclassification at a fixed risk threshold (Figure 4) |

### Attenuation of θ

| Script | Covariate | Framework |
|---|---|---|
| `analysis_mediation.py` | grip strength | joint (path c vs c′); the attenuation reported in the main text |
| `analysis_walkspeed.py` | walking speed | joint |
| `analysis_decomposition.py` | social factors (education, living alone, marital status, social participation) | joint |
| `analysis_offset_attenuation.py` | grip / social / race | offset (see Note below) |

### Robustness / sensitivity

| Script | Result |
|---|---|
| `analysis_offset_disability.py` | θ for incident ADL disability (Table S1) |
| `analysis_offset_supp.py` | full-item FI, 10-common-item FI, and wealth-quintile indicators (Tables S2, S8, S9) |
| `analysis_offset_dif.py` | differential item functioning (DIF) robustness |

### Figures and manuscript

| Script | Output |
|---|---|
| `make_figures.py` | Figure 1–4 |
| `make_recalibration_fig.py` | Figure 5 |
| `make_objective_function_fig.py` | Figure 6 |
| `make_country_forest.py` | Supplementary Figure S1 (SHARE 12-country θ forest plot) |
| `make_docx.py` | `manuscript.docx` (prose + Tables 1–4 + Figures 1–6) |
| `make_supplementary_docx.py` | `supplementary.docx` (Tables S1–S9 + Figure S1) |

## Notes

- **Offset vs. joint estimation.** The headline θ uses offset calibration (`analysis_offset_calibration.py`), which yields a slightly smaller and more homogeneous estimate (pooled θ = 0.129, I² = 0%) than the jointly re-fitted coefficient (0.137, I² = 8%), because the two are distinct estimands. The joint coefficient is reported in the manuscript as the *adjusted association* of disadvantage with mortality, not as a calibration quantity.
- **Attenuation uses joint estimation.** For grip strength, walking speed, and social factors, attenuation is quantified in jointly re-fitted models (`analysis_mediation.py`, `analysis_walkspeed.py`, `analysis_decomposition.py`). The offset versions in `analysis_offset_attenuation.py` overstate attenuation for covariates collinear with the FI (notably walking speed) and were not used for the reported values.
- **KLoSA** is excluded from the mortality analysis (too few deaths within the 4-year window, concentrated in 2020–2022) but retained in the grip-strength anchor analysis.
- **NHANES** is used only as an external, hypothesis-generating check of whether downstream biomarkers account for the residual offset; it does not condition on the FI and does not estimate attenuation of θ.

## Citation

If you use this code, please cite the accompanying manuscript (journal reference to be added upon acceptance).
