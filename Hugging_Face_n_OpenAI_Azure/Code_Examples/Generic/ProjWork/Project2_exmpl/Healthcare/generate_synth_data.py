# Run this script once to generate the PDF
#pip install reportlab

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate(
    "sarah_johnson_labs.pdf",
    pagesize=letter,
    rightMargin=50, leftMargin=50,
    topMargin=50, bottomMargin=50
)

styles = getSampleStyleSheet()
elements = []

# ── Header ────────────────────────────────────────────────────────
elements.append(Paragraph("CITY MEDICAL CENTER", styles["Title"]))
elements.append(Paragraph("Laboratory Investigation Report", styles["Heading2"]))
elements.append(Spacer(1, 12))

# ── Patient Info ──────────────────────────────────────────────────
patient_info = [
    ["Patient Name:", "Sarah Johnson",     "Patient ID:", "P002"],
    ["Date of Birth:", "March 12, 1979",   "Gender:", "Female"],
    ["Ordering Doctor:", "Dr. Sharma",     "Specialty:", "Pulmonologist"],
    ["Sample Collected:", "January 6, 2025", "Report Date:", "January 7, 2025"],
    ["Clinical Notes:", "Asthma follow-up. Rule out infection.", "", ""],
]

t = Table(patient_info, colWidths=[120, 150, 100, 150])
t.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
    ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
    ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
    ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("PADDING", (0,0), (-1,-1), 6),
]))
elements.append(t)
elements.append(Spacer(1, 16))

# ── CBC ───────────────────────────────────────────────────────────
elements.append(Paragraph("1. Complete Blood Count (CBC)", styles["Heading3"]))
cbc_data = [
    ["Test", "Result", "Reference Range", "Unit", "Status"],
    ["Hemoglobin", "13.8", "12.0 - 16.0", "g/dL", "Normal"],
    ["WBC Count", "9.2", "4.0 - 11.0", "x10³/μL", "Normal"],
    ["RBC Count", "4.5", "3.8 - 5.2", "x10⁶/μL", "Normal"],
    ["Platelets", "220", "150 - 400", "x10³/μL", "Normal"],
    ["Neutrophils", "68", "40 - 75", "%", "Normal"],
    ["Lymphocytes", "24", "20 - 45", "%", "Normal"],
    ["Eosinophils", "6.2", "1 - 4", "%", "HIGH ⚠"],
    ["Hematocrit", "41.2", "36 - 46", "%", "Normal"],
    ["MCV", "88.4", "80 - 100", "fL", "Normal"],
]

t2 = Table(cbc_data, colWidths=[140, 70, 130, 70, 80])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E86AB")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
    ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("TEXTCOLOR", (4,7), (4,7), colors.red),
    ("FONTNAME", (4,7), (4,7), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6),
]))
elements.append(t2)
elements.append(Spacer(1, 16))

# ── Thyroid Function ──────────────────────────────────────────────
elements.append(Paragraph("2. Thyroid Function Tests", styles["Heading3"]))
thyroid_data = [
    ["Test", "Result", "Reference Range", "Unit", "Status"],
    ["TSH", "2.1", "0.4 - 4.0", "mIU/L", "Normal"],
    ["Free T4", "1.2", "0.8 - 1.8", "ng/dL", "Normal"],
    ["Free T3", "3.4", "2.3 - 4.2", "pg/mL", "Normal"],
]

t3 = Table(thyroid_data, colWidths=[140, 70, 130, 70, 80])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E86AB")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
    ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("PADDING", (0,0), (-1,-1), 6),
]))
elements.append(t3)
elements.append(Spacer(1, 16))

# ── Pulmonary / Allergy Markers ───────────────────────────────────
elements.append(Paragraph("3. Allergy & Pulmonary Markers", styles["Heading3"]))
allergy_data = [
    ["Test", "Result", "Reference Range", "Unit", "Status"],
    ["Total IgE", "420", "< 100", "IU/mL", "HIGH ⚠"],
    ["Specific IgE (Dust mites)", "3.8", "< 0.35", "kUA/L", "HIGH ⚠"],
    ["Specific IgE (Pollen)", "2.1", "< 0.35", "kUA/L", "HIGH ⚠"],
    ["CRP (C-Reactive Protein)", "8.4", "< 5.0", "mg/L", "HIGH ⚠"],
    ["Peak Expiratory Flow", "310", "380 - 500", "L/min", "LOW ⚠"],
]

t4 = Table(allergy_data, colWidths=[160, 70, 120, 70, 80])
t4.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E86AB")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
    ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("TEXTCOLOR", (4,1), (4,5), colors.red),
    ("FONTNAME", (4,1), (4,5), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6),
]))
elements.append(t4)
elements.append(Spacer(1, 16))

# ── Metabolic Panel ───────────────────────────────────────────────
elements.append(Paragraph("4. Basic Metabolic Panel", styles["Heading3"]))
metabolic_data = [
    ["Test", "Result", "Reference Range", "Unit", "Status"],
    ["Glucose (Fasting)", "94", "70 - 100", "mg/dL", "Normal"],
    ["Sodium", "139", "136 - 145", "mmol/L", "Normal"],
    ["Potassium", "3.9", "3.5 - 5.1", "mmol/L", "Normal"],
    ["Creatinine", "0.78", "0.5 - 1.1", "mg/dL", "Normal"],
    ["eGFR", "88", "> 60", "mL/min/1.73m²", "Normal"],
    ["ALT", "22", "7 - 40", "U/L", "Normal"],
    ["AST", "19", "10 - 40", "U/L", "Normal"],
]

t5 = Table(metabolic_data, colWidths=[150, 70, 120, 90, 80])
t5.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E86AB")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4F8")]),
    ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
    ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("PADDING", (0,0), (-1,-1), 6),
]))
elements.append(t5)
elements.append(Spacer(1, 16))

# ── Doctor Notes ──────────────────────────────────────────────────
elements.append(Paragraph("5. Interpreting Physician Notes", styles["Heading3"]))
notes = """
Dr. Sharma (Pulmonologist) — January 7, 2025

Key Findings:
- Elevated eosinophils (6.2%) suggest allergic inflammation, consistent with asthma.
- Markedly elevated Total IgE (420 IU/mL) and positive specific IgE to dust mites 
  and pollen strongly indicate allergic sensitization as a trigger for asthma symptoms.
- CRP mildly elevated at 8.4 mg/L — suggests low-grade airway inflammation.
- Peak expiratory flow reduced at 310 L/min (below expected 380-500 L/min range) 
  — indicates moderate airflow limitation at time of testing.
- Thyroid function normal — Levothyroxine dose appears adequate.
- Kidney and liver function within normal limits.

Recommendations:
1. Consider allergen immunotherapy referral given high IgE and specific sensitization.
2. Review inhaler technique and step up to combination ICS/LABA if not already done.
3. Avoid known allergens (dust mites, pollen) — environmental control measures advised.
4. Repeat PEF measurement after bronchodilator to assess reversibility.
5. Follow up in 4 weeks to reassess respiratory status.

Signed: Dr. Priya Sharma, MD | Pulmonology | City Medical Center
"""
elements.append(Paragraph(notes.replace("\n", "<br/>"), styles["Normal"]))

doc.build(elements)
print("PDF generated: sarah_johnson_labs.pdf")
