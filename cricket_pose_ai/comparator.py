"""
Cricket Pose Analyzer AI - Comparator Module
=============================================
Comparative analysis engine for comparing two cricket batting videos or past
sessions side-by-side, evaluating technique progression and delta metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cricket_pose_ai.motion_analyzer import MotionSummaryMetrics
from cricket_pose_ai.report_generator import PDFReportGenerator
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.Comparator")


@dataclass
class ComparisonDelta:
    """Metric differences between Video A and Video B."""
    score_diff: float
    balance_diff: float
    head_stability_diff: float
    weight_transfer_diff: float
    bat_speed_diff: float
    head_drift_diff: float
    better_video_label: str


class VideoComparator:
    """Comparative biomechanics evaluation engine."""

    def __init__(self):
        self.report_generator = PDFReportGenerator()

    def compare_summaries(
        self,
        summary_a: MotionSummaryMetrics,
        summary_b: MotionSummaryMetrics,
        label_a: str = "Video A",
        label_b: str = "Video B"
    ) -> Tuple[ComparisonDelta, Path]:
        """
        Calculates metric deltas between Video A and Video B and generates comparison PDF report.
        """
        score_diff = summary_b.overall_technique_score - summary_a.overall_technique_score
        bal_diff = summary_b.balance_index - summary_a.balance_index
        head_diff = summary_b.head_stability_index - summary_a.head_stability_index
        wt_diff = summary_b.weight_transfer_index - summary_a.weight_transfer_index
        speed_diff = summary_b.max_bat_speed_px - summary_a.max_bat_speed_px
        drift_diff = summary_b.head_drift_px - summary_a.head_drift_px

        if score_diff > 2.0:
            better = label_b
        elif score_diff < -2.0:
            better = label_a
        else:
            better = "Equal Technical Performance"

        delta = ComparisonDelta(
            score_diff=round(score_diff, 1),
            balance_diff=round(bal_diff, 1),
            head_stability_diff=round(head_diff, 1),
            weight_transfer_diff=round(wt_diff, 1),
            bat_speed_diff=round(speed_diff, 1),
            head_drift_diff=round(drift_diff, 1),
            better_video_label=better
        )

        pdf_path = self.report_generator.generate_comparison_report(
            summary_a=summary_a,
            summary_b=summary_b,
            label_a=label_a,
            label_b=label_b,
            pdf_filename="comparison_report.pdf"
        )

        logger.info(f"Compared {label_a} vs {label_b}. Better: {better}")
        return delta, pdf_path
