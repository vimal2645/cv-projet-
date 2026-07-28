"""
Cricket Pose Analyzer AI - Report Generator Module
===================================================
Generates publication-quality PDF technical reports using ReportLab.
Includes executive scorecards, key metrics tables, embedded graph charts,
spatial motion heatmaps, and rule-based coaching plans.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cricket_pose_ai.coaching_engine import CoachingReport
from cricket_pose_ai.config import OUTPUT_DIR
from cricket_pose_ai.motion_analyzer import MotionSummaryMetrics
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.ReportGenerator")


class PDFReportGenerator:
    """Generates professional PDF coaching and technical reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_single_report(
        self,
        summary: MotionSummaryMetrics,
        coaching: CoachingReport,
        graph_paths: Dict[str, Path],
        heatmap_paths: Dict[str, Path],
        pdf_filename: str = "analysis_report.pdf"
    ) -> Path:
        """Renders comprehensive PDF report for a single cricket batting video analysis."""
        filepath = self.output_dir / pdf_filename
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom Styling Palette
        c_primary = colors.HexColor("#0F172A")
        c_accent = colors.HexColor("#0284C7")
        c_text = colors.HexColor("#334155")
        c_card = colors.HexColor("#F8FAFC")

        style_title = ParagraphStyle('DocTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.white, alignment=1)
        style_sub = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor("#E2E8F0"), alignment=1)
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=14, textColor=c_primary, spaceBefore=12, spaceAfter=6)
        style_body = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, textColor=c_text, leading=13)
        style_bold = ParagraphStyle('BoldBody', parent=style_body, fontName='Helvetica-Bold')

        story = []

        # 1. Header Banner
        header_table = Table(
            [[Paragraph("CRICKET POSE ANALYZER AI", style_title)],
             [Paragraph("Comprehensive Biomechanical & Technique Performance Report", style_sub)]],
            colWidths=[540]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_primary),
            ('TOPPADDING', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 14))

        # 2. Executive Scorecard Box
        score_data = [
            [
                Paragraph(f"<b>Overall Score:</b> {summary.overall_technique_score}/100", style_body),
                Paragraph(f"<b>Technique Grade:</b> {coaching.technique_grade}", style_body),
                Paragraph(f"<b>Duration:</b> {summary.duration_sec}s ({summary.total_frames} frames)", style_body)
            ]
        ]
        score_table = Table(score_data, colWidths=[180, 180, 180])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#E0F2FE")),
            ('BOX', (0, 0), (-1, -1), 1, c_accent),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BAE6FD")),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 14))

        # 3. Biomechanics Metrics Table
        story.append(Paragraph("Biomechanical Ratings & Key Performance Indicators", style_h1))
        metrics_data = [
            ["Metric Parameter", "Observed Value", "Status Rating"],
            ["Balance Index", f"{summary.balance_index} / 100", "Excellent" if summary.balance_index >= 85 else "Needs Work"],
            ["Head Stability Index", f"{summary.head_stability_index} / 100", "Optimal" if summary.head_stability_index >= 80 else "Excess Drift"],
            ["Weight Transfer Index", f"{summary.weight_transfer_index} / 100", "Good Shift" if summary.weight_transfer_index >= 80 else "Passive Shift"],
            ["Posture Quality", f"{summary.posture_quality_score} / 100", "Optimal Flex" if summary.posture_quality_score >= 80 else "Rigid"],
            ["Max Bat Speed", f"{summary.max_bat_speed_px} px/s", "Peak Velocity"],
            ["Max Spine Lean", f"{summary.max_spine_lean_deg}°", "Controlled Lean"]
        ]
        m_table = Table(metrics_data, colWidths=[200, 170, 170])
        m_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_accent),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), c_card),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(m_table)
        story.append(Spacer(1, 14))

        # 4. Embedded Visual Charts
        story.append(Paragraph("Kinematic Analysis & Joint Angle Profiles", style_h1))
        if "joint_angles" in graph_paths and graph_paths["joint_angles"].exists():
            img_angles = Image(str(graph_paths["joint_angles"]), width=520, height=220)
            story.append(img_angles)
            story.append(Spacer(1, 10))

        if "movement_speed" in graph_paths and graph_paths["movement_speed"].exists():
            img_speed = Image(str(graph_paths["movement_speed"]), width=520, height=220)
            story.append(img_speed)
            story.append(Spacer(1, 14))

        story.append(PageBreak())

        # Page 2: Heatmaps & AI Coaching Engine
        story.append(Paragraph("Spatial Motion Heatmaps & Trajectory Analysis", style_h1))
        if "wrist_heatmap" in heatmap_paths and heatmap_paths["wrist_heatmap"].exists():
            h_img = Image(str(heatmap_paths["wrist_heatmap"]), width=520, height=260)
            story.append(h_img)
            story.append(Spacer(1, 14))

        story.append(Paragraph("AI Coaching Insights & Prescription Plan", style_h1))
        story.append(Paragraph(f"<b>Verdict:</b> {coaching.summary_verdict}", style_body))
        story.append(Spacer(1, 8))

        # Strengths & Weaknesses
        story.append(Paragraph("Technique Strengths:", style_bold))
        for st in coaching.strengths:
            story.append(Paragraph(f"• {st}", style_body))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Areas for Improvement:", style_bold))
        for wk in coaching.weaknesses:
            story.append(Paragraph(f"• {wk}", style_body))
        story.append(Spacer(1, 10))

        # Recommended Drills Table
        story.append(Paragraph("Recommended Training Drills:", style_bold))
        drill_rows = [["Drill Name", "Instructions & Protocol"]]
        for d in coaching.recommended_drills:
            drill_rows.append([d["title"], d["description"]])

        d_table = Table(drill_rows, colWidths=[180, 360])
        d_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_primary),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), c_card),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(d_table)

        doc.build(story)
        logger.info(f"Generated PDF report at {filepath}")
        return filepath

    def generate_comparison_report(
        self,
        summary_a: MotionSummaryMetrics,
        summary_b: MotionSummaryMetrics,
        label_a: str = "Session A",
        label_b: str = "Session B",
        pdf_filename: str = "comparison_report.pdf"
    ) -> Path:
        """Generates side-by-side comparative PDF report for 2 videos."""
        filepath = self.output_dir / pdf_filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        c_primary = colors.HexColor("#0F172A")
        style_title = ParagraphStyle('TitleComp', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=20, textColor=colors.white, alignment=1)
        style_h1 = ParagraphStyle('H1Comp', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=14, textColor=c_primary, spaceBefore=12, spaceAfter=6)
        style_body = ParagraphStyle('BodyComp', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, textColor=colors.HexColor("#334155"))

        story = []

        # Banner
        banner = Table([[Paragraph("CRICKET POSE ANALYZER AI - DUAL COMPARISON", style_title)]], colWidths=[540])
        banner.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), c_primary), ('PADDING', (0, 0), (-1, -1), 12)]))
        story.append(banner)
        story.append(Spacer(1, 14))

        # Comparison Table
        story.append(Paragraph("Side-by-Side Biomechanical Metrics Comparison", style_h1))
        comp_data = [
            ["Metric", label_a, label_b, "Difference (Delta)"],
            ["Overall Score", f"{summary_a.overall_technique_score}", f"{summary_b.overall_technique_score}", f"{summary_b.overall_technique_score - summary_a.overall_technique_score:+.1f}"],
            ["Balance Index", f"{summary_a.balance_index}", f"{summary_b.balance_index}", f"{summary_b.balance_index - summary_a.balance_index:+.1f}"],
            ["Head Stability", f"{summary_a.head_stability_index}", f"{summary_b.head_stability_index}", f"{summary_b.head_stability_index - summary_a.head_stability_index:+.1f}"],
            ["Weight Transfer", f"{summary_a.weight_transfer_index}", f"{summary_b.weight_transfer_index}", f"{summary_b.weight_transfer_index - summary_a.weight_transfer_index:+.1f}"],
            ["Max Bat Speed", f"{summary_a.max_bat_speed_px} px/s", f"{summary_b.max_bat_speed_px} px/s", f"{summary_b.max_bat_speed_px - summary_a.max_bat_speed_px:+.1f}"],
            ["Head Drift", f"{summary_a.head_drift_px} px", f"{summary_b.head_drift_px} px", f"{summary_b.head_drift_px - summary_a.head_drift_px:+.1f}"]
        ]

        ctable = Table(comp_data, colWidths=[150, 130, 130, 130])
        ctable.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(ctable)

        doc.build(story)
        logger.info(f"Generated comparison PDF report at {filepath}")
        return filepath
