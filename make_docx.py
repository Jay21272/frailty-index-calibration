# -*- coding: utf-8 -*-
"""manuscript.md + 3三线表 + 4图 → .docx（J Gerontol A 格式：双倍行距、Times New Roman 12pt）。"""
import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MD = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\manuscript.md"
OUT = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\manuscript.docx"
FIG = r"C:\Users\jayzh\Desktop\业务论文\社会健康预测\figures"


def set_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_three_line_table(doc, header, rows):
    """三线表：顶线、栏目线、底线。"""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(h); set_font(r, 10, bold=True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(v); set_font(r, 10)
    # 边框：顶线+底线粗，栏目线细
    tbl = t._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for tag, sz in [("top", 12), ("bottom", 12), ("insideH", 6)]:
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "000000")
        borders.append(el)
    for tag in ["left", "right", "insideV"]:
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tblPr.append(borders)
    doc.add_paragraph()


def parse_md(doc):
    with open(MD, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        if line.startswith("#### "):
            p = doc.add_paragraph(); set_font(p.add_run(line[5:]), 12, bold=True)
            p.paragraph_format.line_spacing = 2.0
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            h = doc.add_heading(line[2:], level=1)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            # 处理 **粗体** 和 *斜体*
            p = doc.add_paragraph()
            parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", line)
            for part in parts:
                if part.startswith("**") and part.endswith("**") and len(part) >= 4:
                    set_font(p.add_run(part[2:-2]), 12, bold=True)
                elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
                    set_font(p.add_run(part[1:-1]), 12, italic=True)
                else:
                    set_font(p.add_run(part), 12)
            p.paragraph_format.line_spacing = 2.0
    return doc


doc = Document()
# 默认字体
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

parse_md(doc)

# 表（三线表）
doc.add_heading("Tables", level=2)
def table_caption(doc, text):
    p = doc.add_paragraph(); set_font(p.add_run(text), 10, bold=True)

table_caption(doc, "Table 1. Cohort characteristics (analysis sample, age ≥60).")
add_three_line_table(doc,
    ["Cohort", "N", "Age (IQR)", "Female", "Median FI", "4-y mortality"],
    [["CHARLS", "4,982", "68 (63–73)", "49%", "0.182", "10.6%"],
     ["ELSA", "6,969", "71 (65–77)", "55%", "0.200", "12.7%"],
     ["SHARE", "17,277", "70 (64–76)", "54%", "0.167", "9.8%"],
     ["HRS", "31,702", "66 (60–71)", "55%", "0.167", "11.1%"]])
doc.add_paragraph()
table_caption(doc, "Table 2. Socioeconomic calibration offset θ in mortality and disability.")
add_three_line_table(doc,
    ["Cohort", "Mortality θ (SE)", "Disability θ (SE)"],
    [["CHARLS", "0.090 (0.034)", "0.131"],
     ["ELSA", "0.135 (0.028)", "0.120"],
     ["SHARE", "0.121 (0.020)", "0.100"],
     ["HRS", "0.139 (0.015)", "—"],
     ["Pooled (RE)", "0.129 (0.102–0.155)", "—"]])
doc.add_paragraph()
table_caption(doc, "Table 3. Calibration components and correction.")
add_three_line_table(doc,
    ["Cohort", "Poor intercept", "Rich intercept", "Poor slope", "Rich slope", "Gap closure"],
    [["CHARLS", "+0.109", "−0.446", "1.164", "0.941", "0.166"],
     ["ELSA", "+0.196", "−0.306", "0.911", "1.146", "0.222"],
     ["SHARE", "+0.237", "−0.192", "0.952", "0.970", "0.207"],
     ["HRS", "+0.218", "−0.405", "0.822", "1.143", "0.223"]])
doc.add_paragraph()
table_caption(doc, "Table 4. Objective physical function and the disadvantage gradient θ.")
add_three_line_table(doc,
    ["Cohort", "Grip Δ (kg)", "Grip θ attenuation", "Walk-speed θ attenuation", "FI→grip β"],
    [["CHARLS", "3.0–5.4", "48%", "−3%", "−7.3"],
     ["ELSA", "—†", "21%†", "13%", "−9.7"],
     ["SHARE", "3.1–5.3", "7%", "—", "−13.4"],
     ["HRS", "—", "—", "—", "—"]])
p = doc.add_paragraph(); set_font(p.add_run("† ELSA grip strength measured at wave 2 (wave 1 did not assess it); walking speed unavailable in SHARE and HRS."), 10)

# 图（插入到文末，按编号）
doc.add_heading("Figures", level=2)
for i, cap in [(1, "Disadvantage gradient θ in mortality across four cohorts. Forest plot with pooled random-effects estimate (θ=0.137; 95% CI 0.101–0.174; I²=8%)."),
               (2, "Calibration curves (observed vs predicted mortality) for the poorest (dashed) and wealthiest (solid) quintiles in HRS and ELSA. Deviation above the reference line in the poorest quintile indicates under-prediction."),
               (3, "Grip strength by frailty index tertile for the poorest (dashed) and wealthiest (solid) quintiles in three cohorts. At a given FI, disadvantaged older adults have lower grip strength."),
               (4, "Recalibration and reclassification. (A) The poor-quintile calibration intercept before and after a socioeconomic adjustment. (B) Reclassification at a fixed risk threshold: disadvantaged older adults upgraded more than advantaged older adults downgraded."),
               (5, "Calibration curve for the poorest quintile before (dashed) and after (solid) a socioeconomic adjustment, in HRS and SHARE. The adjustment flattens the curve onto the reference line."),
               (6, "Attenuation of θ after adding grip strength versus the FI–grip strength association across cohorts. Cohorts where the FI weakly captured grip strength (CHARLS) showed the largest θ attenuation, whereas SHARE, where the FI captured grip most strongly, showed the smallest. The dashed line is a descriptive reference.")]:
    # 实际文件名
    import glob, os
    m = glob.glob(os.path.join(FIG, f"Fig{i}_*.png"))
    if m:
        doc.add_picture(m[0], width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); set_font(p.add_run(f"Figure {i}. {cap}"), 10)

doc.save(OUT)
print("已生成", OUT)
