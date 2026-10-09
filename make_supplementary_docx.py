# -*- coding: utf-8 -*-
"""supplementary.md → supplementary.docx（三线表 + 段落）。"""
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\supplementary.docx"


def set_font(run, size=11, bold=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold


def add_table(doc, header, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]; cell.text = ""
        set_font(cell.paragraphs[0].add_run(h), 10, bold=True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.rows[i + 1].cells[j]; cell.text = ""
            set_font(cell.paragraphs[0].add_run(v), 10)
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for tag, sz in [("top", 12), ("bottom", 12), ("insideH", 6)]:
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "000000")
        borders.append(el)
    for tag in ["left", "right", "insideV"]:
        el = OxmlElement(f"w:{tag}"); el.set(qn("w:val"), "none"); borders.append(el)
    tblPr.append(borders)
    doc.add_paragraph()


def caption(doc, text):
    p = doc.add_paragraph(); set_font(p.add_run(text), 11, bold=True)


def para(doc, text):
    p = doc.add_paragraph(); set_font(p.add_run(text), 11)
    p.paragraph_format.line_spacing = 1.5


doc = Document()
doc.styles["Normal"].font.name = "Times New Roman"
doc.styles["Normal"].font.size = Pt(11)

h = doc.add_heading("Supplementary Material", level=1); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
para(doc, "Supplementary Material for: The frailty index under-predicts mortality in socioeconomically disadvantaged older adults: cross-national calibration evidence from four cohorts")

caption(doc, "Table S1. Disability outcome: disadvantage gradient θ (new ADL disability, ~4-year follow-up)")
add_table(doc, ["Cohort", "N", "Events", "θ", "SE", "p"],
          [["CHARLS", "3,531", "487", "0.130", "0.036", "<0.001"],
           ["ELSA", "3,474", "502", "0.115", "0.037", "0.002"],
           ["SHARE", "6,365", "840", "0.097", "0.028", "<0.001"]])
para(doc, "θ estimated from logit(disability) = FI + age + sex + θ·disadvantage, among those free of ADL disability at baseline. Death before follow-up was treated as censored.")

caption(doc, "Table S2. Full frailty index: migration of θ and discrimination")
add_table(doc, ["Cohort", "12-item θ", "Full-FI items", "Full-FI θ", "Δθ", "12-item AUC"],
          [["CHARLS", "0.090", "28", "0.084", "−0.006", "0.751"],
           ["ELSA", "0.135", "25", "0.110", "−0.025", "0.782"],
           ["SHARE", "0.121", "26", "0.102", "−0.019", "0.777"],
           ["HRS", "0.139", "18", "0.115", "−0.023", "0.816"]])
para(doc, "Full FI included additional comparable items (mobility, cognition, depression, grip strength where available). AUC is the discrimination of the FI + age + sex mortality model.")

caption(doc, "Table S3. Clustered standard errors (household-level clustering, GEE)")
add_table(doc, ["Cohort", "SE (unclustered)", "SE (clustered)", "p (clustered)"],
          [["CHARLS", "0.034", "0.033", "0.007"],
           ["ELSA", "0.029", "0.029", "<0.001"],
           ["SHARE", "0.020", "0.022", "<0.001"],
           ["HRS", "0.016", "0.016", "<0.001"]])
para(doc, "Household identifiers were used for clustering; within-household correlation was weak, and clustering did not materially change the standard errors.")

caption(doc, "Table S4. Differential item functioning (logistic DIF, poorest vs wealthiest quintile)")
add_table(doc, ["Cohort", "Items with DIF"],
          [["CHARLS", "diabetes, cancer, self-rated health"],
           ["ELSA", "lung disease, IADL, self-rated health"],
           ["SHARE", "IADL"],
           ["HRS", "ADL, IADL, low BMI, self-rated health"]])
para(doc, "DIF was assessed by logistic regression of each item on the purified total score (remaining items) plus a poor/rich indicator; an item was flagged when p < 0.05 and ΔMcFadden R² ≥ 0.02. Self-rated health—the only subjective item—showed DIF in three cohorts. After removing items with DIF, θ did not attenuate but strengthened (e.g., HRS θ 0.139 → 0.215).")

caption(doc, "Table S5. Functional calibration: calibration intercept across wealth deciles")
add_table(doc, ["Cohort", "Bottom deciles", "Middle deciles", "Top deciles", "Bottom−Top", "Linear slope"],
          [["CHARLS", "+0.132", "+0.164", "−0.646", "0.779", "−0.028"],
           ["ELSA", "+0.310", "−0.136", "−0.281", "0.591", "−0.034"],
           ["SHARE", "+0.260", "−0.215", "−0.206", "0.466", "−0.028"],
           ["HRS", "+0.296", "+0.049", "−0.530", "0.826", "−0.038"]])
para(doc, "Calibration intercept estimated within each wealth decile using an offset model. The intercept varied smoothly and monotonically with continuous wealth, with no evidence of a threshold cliff at the extreme of poverty.")

caption(doc, "Table S6. Race/ethnicity in HRS: characteristics and within-group disadvantage gradient")
add_table(doc, ["Group", "N", "4-y mortality (%)", "Median FI", "Median wealth (US$)", "θ (SE)", "p"],
          [["Non-Hispanic White", "22,982", "11.5", "0.167", "168,000", "0.14 (0.018)", "<0.001"],
           ["Non-Hispanic Black", "5,749", "12.6", "0.250", "32,365", "0.21 (0.044)", "<0.001"],
           ["Hispanic", "3,548", "7.1", "0.167", "37,013", "0.19 (0.069)", "0.007"],
           ["Non-Hispanic Other", "882", "9.8", "0.167", "97,000", "0.14 (0.091)", "0.14"]])
para(doc, "Race/ethnicity was defined in HRS only (RAND RARACEM and RAHISPAN). Within-group θ was estimated from logit(mortality) = FI + age + sex + θ·disadvantage within each group.")

caption(doc, "Table S7. Race/ethnicity adjustment of the HRS disadvantage gradient θ")
add_table(doc, ["Model", "θ or OR (95% CI)", "p"],
          [["θ, unadjusted (FI + age + sex + disadvantage)", "0.139 (0.015)", "<0.001"],
           ["θ, race-adjusted", "0.158 (0.017)", "<0.001"],
           ["Non-Hispanic Black vs White (net of wealth)", "OR 0.98 (0.88–1.09)", "0.67"],
           ["Hispanic vs White (net of wealth)", "OR 0.58 (0.50–0.68)", "<0.001"],
           ["Non-Hispanic Other vs White (net of wealth)", "OR 0.92 (0.70–1.20)", "0.53"]])
para(doc, "Race-adjusted θ and race odds ratios were estimated from logit(mortality) = FI + age + sex + θ·disadvantage + race/ethnicity (reference: non-Hispanic White).")

caption(doc, "Table S8. Ten-item common frailty index: robustness of θ")
add_table(doc, ["Cohort", "12-item θ", "10-item θ", "Δθ"],
          [["CHARLS", "0.090", "0.090", "−0.001"],
           ["ELSA", "0.135", "0.135", "0.000"],
           ["SHARE", "0.121", "0.125", "+0.004"],
           ["HRS", "0.139", "0.134", "−0.004"]])
para(doc, "The 10-item FI omitted low and high body mass index (both missing in ELSA), leaving the seven chronic conditions plus ADL, IADL, and self-rated health common to all four cohorts. θ was essentially unchanged.")

caption(doc, "Table S9. Wealth-quintile indicators instead of a linear disadvantage score")
add_table(doc, ["Cohort", "Disadv 1", "Disadv 2", "Disadv 3", "Disadv 4 (poorest)", "LR p (linearity)"],
          [["CHARLS", "0.513", "0.619", "0.446", "0.555", "0.017"],
           ["ELSA", "0.135", "0.177", "0.490", "0.501", "0.511"],
           ["SHARE", "0.057", "−0.020", "0.322", "0.428", "0.020"],
           ["HRS", "0.212", "0.459", "0.442", "0.622", "0.033"]])
para(doc, "Each coefficient is the log-odds of mortality for that disadvantage level relative to the wealthiest quintile, from logit(mortality) = FI + age + sex + disadvantage indicators. The gradient was monotonic and concentrated at the poorest end; the likelihood-ratio test of the linear (1 df) against the indicator (4 df) specification is reported, and the substantive conclusion was unchanged.")

caption(doc, "Supplementary Note")
para(doc, "Meta-analysis: pooled θ (random effects) = 0.137; Hartung–Knapp corrected 95% CI [0.101, 0.174] (df = 3); I² = 8%, τ² = 0.0002.")
para(doc, "Missing data: CHARLS self-rated health missing 28% and body mass index 21%; ELSA lacks body mass index; SHARE and HRS missing <3%. The frailty index required ≥80% of items non-missing.")
para(doc, "Cohort exclusion: KLoSA excluded from mortality analysis (14 deaths within 5 years); retained in grip-strength anchor analysis.")
para(doc, "Software: Python 3.12, statsmodels.")

# Supplementary Figure S1：SHARE 12 国 θ 森林图
import glob
m = glob.glob(r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\figures\FigS1_country_forest.png")
if m:
    caption(doc, "Figure S1. Disadvantage gradient θ within SHARE's 12 constituent countries (within-country wealth quintiles).")
    doc.add_picture(m[0], width=Inches(5.5))
    para(doc, "The gradient was consistent in direction across countries (I²=18%; heterogeneity Q test p=0.26), though only 3 of 12 countries were individually significant, reflecting limited death-event counts per country.")

doc.save(OUT)
print("已生成", OUT)
