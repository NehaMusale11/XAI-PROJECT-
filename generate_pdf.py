"""
Script to generate a professional PDF project report for Faculty Presentation.
Includes Project Overview, Dataset details, Trained Model Metrics,
Post-hoc XAI Techniques (SHAP, LIME, PDP, Counterfactuals), and Hinglish explanation guide.
"""

import sys
import os

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
except ImportError:
    print("[Error] reportlab is still installing. Please wait or run pip install reportlab.")
    sys.exit(1)


def build_pdf(filename="XAI_Project_Explanation_Report.pdf"):
    pdf_path = os.path.abspath(filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e1b4b'),
        alignment=1, # Center
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4338ca'),
        alignment=1,
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # Title Banner
    story.append(Paragraph("POST-HOC EXPLANATION OF BLACK-BOX MODELS", title_style))
    story.append(Paragraph("Explainable AI (XAI) Project Report & Faculty Presentation Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366f1'), spaceAfter=12))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Objective", h2_style))
    story.append(Paragraph(
        "Modern Machine Learning models (like XGBoost and Deep Neural Networks) yield exceptional accuracy but function as <b>Black-Box Models</b> because their inner mathematical logic is opaque to human interpretation. This project implements a comprehensive <b>Post-hoc Explainable AI (XAI) Suite</b> and <b>Interactive Web Dashboard</b> to interpret black-box predictions without altering underlying model structures.",
        body_style
    ))

    # Section 2: Model Performance Table
    story.append(Paragraph("2. Trained Classification Models Performance", h2_style))
    
    table_data = [
        [Paragraph("<b>Model Architecture</b>", body_style), Paragraph("<b>Model Type</b>", body_style), Paragraph("<b>Accuracy</b>", body_style), Paragraph("<b>F1-Score</b>", body_style), Paragraph("<b>ROC-AUC</b>", body_style)],
        [Paragraph("<b>XGBoost Classifier</b>", body_style), Paragraph("Black-Box (Ensemble)", body_style), Paragraph("88.33%", body_style), Paragraph("0.9067", body_style), Paragraph("0.9674", body_style)],
        [Paragraph("<b>Multi-Layer Perceptron</b>", body_style), Paragraph("Black-Box (Neural Net)", body_style), Paragraph("90.00%", body_style), Paragraph("0.9221", body_style), Paragraph("0.9641", body_style)],
        [Paragraph("<b>Random Forest</b>", body_style), Paragraph("Black-Box (Ensemble)", body_style), Paragraph("84.17%", body_style), Paragraph("0.8758", body_style), Paragraph("0.9384", body_style)],
        [Paragraph("<b>Logistic Regression</b>", body_style), Paragraph("White-Box (Baseline)", body_style), Paragraph("87.50%", body_style), Paragraph("0.9032", body_style), Paragraph("0.9419", body_style)],
    ]

    t = Table(table_data, colWidths=[130, 130, 80, 80, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e1b4b')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Section 3: Four Post-hoc XAI Techniques
    story.append(Paragraph("3. Implemented Post-hoc Interpretability Techniques", h2_style))
    story.append(Paragraph("<b>A. SHAP (SHapley Additive exPlanations):</b> Game-theoretic coalitional attribution calculating exact Shapley values. Satisfies Additivity Theorem: Base Value + sum(SHAP) = Prediction.", bullet_style))
    story.append(Paragraph("<b>B. LIME (Local Interpretable Model-agnostic Explanations):</b> Generates local sparse linear surrogate models around perturbed instance neighborhoods. Evaluated using local R^2 fidelity scores (R^2 = 0.60 - 0.91).", bullet_style))
    story.append(Paragraph("<b>C. Partial Dependence Plots (PDP) & ICE:</b> Visualizes marginal response curves illustrating feature interaction trends on predicted disease probabilities.", bullet_style))
    story.append(Paragraph("<b>D. Counterfactual Explanations:</b> Optimization-based minimal actionable feature modifications required to flip high-risk predictions to low-risk.", bullet_style))

    story.append(Spacer(1, 10))

    # Section 4: Faculty Explanation Guide in Hinglish
    story.append(Paragraph("4. Simple Hinglish Explanation Guide for Faculty Presentation", h2_style))

    hinglish_text = [
        "<b>Q1. Project ka main purpose kya hai?</b><br/>"
        "<i>Answer:</i> Sir/Ma'am, jab hum complex ML models (jaise XGBoost ya Neural Networks) use karte hain, toh wo 'Black-Box' hote hain — yaani wo high accuracy toh dete hain par bata nahi sakte ki decision KYUN liya. Our project applies <b>Post-hoc XAI techniques</b> to explain black-box decisions after prediction without modifying the trained model.",

        "<b>Q2. Konsi 4 main post-hoc techniques implement ki hain?</b><br/>"
        "<i>Answer:</i><br/>"
        "1. <b>SHAP:</b> Game theory based attributions jo exact feature impact point dikhati hai.<br/>"
        "2. <b>LIME:</b> Prediction ke aaspas local linear model fit karke local feature weights + R^2 fidelity score batata hai.<br/>"
        "3. <b>PDP / ICE:</b> Single feature change hone par overall risk curve dikhaata hai.<br/>"
        "4. <b>Counterfactuals:</b> Actionable recommendations deta hai (e.g., 'If ST depression drops from 2.8 to 0.8mm, disease risk drops to 22%').",

        "<b>Q3. Implementation & GitHub details?</b><br/>"
        "<i>Answer:</i> Humne Flask REST API + HTML5/CSS3 glassmorphism UI + Plotly.js se live dashboard banaya hai. Entire project is pushed to GitHub: <b>https://github.com/NehaMusale11/XAI-PROJECT-</b>"
    ]

    for item in hinglish_text:
        story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"[PDF Generator] Successfully generated PDF at: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    build_pdf()
