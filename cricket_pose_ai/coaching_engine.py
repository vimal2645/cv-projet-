"""
Cricket Pose Analyzer AI - Rule-Based AI Coaching Engine Module
================================================================
Evaluates biomechanical parameters against professional cricket technique
benchmarks to generate deterministic coaching insights, strengths, weaknesses,
recommended drills, and technique grades.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from cricket_pose_ai.config import CricketBenchmarks
from cricket_pose_ai.motion_analyzer import MotionSummaryMetrics
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.CoachingEngine")


@dataclass
class CoachingReport:
    """Complete coaching evaluation output."""
    technique_grade: str
    overall_score: float
    summary_verdict: str
    strengths: List[str]
    weaknesses: List[str]
    actionable_improvements: List[str]
    recommended_drills: List[Dict[str, str]]
    focus_areas: List[str]


class RuleBasedCoachingEngine:
    """Deterministic cricket biomechanics coaching expert."""

    def __init__(self, benchmarks: CricketBenchmarks = CricketBenchmarks()):
        self.benchmarks = benchmarks

    def evaluate_performance(self, summary: MotionSummaryMetrics) -> CoachingReport:
        """Evaluates motion metrics and produces structured coaching report."""
        score = summary.overall_technique_score
        grade, verdict = self._determine_grade(score)

        strengths = []
        weaknesses = []
        improvements = []
        drills = []
        focus_areas = []

        # 1. Head Stability Evaluation
        if summary.head_stability_index >= 85.0:
            strengths.append("Excellent Head Stability: Maintains still head position through downswing and impact.")
        elif summary.head_stability_index >= 70.0:
            strengths.append("Acceptable Head Control: Minimal head drift during initial stroke setup.")
            weaknesses.append("Minor Head Drift: Slight lateral head movement observed prior to impact.")
            improvements.append("Keep eyes level with the horizon and focus on keeping head still over ball contact line.")
            drills.append({
                "title": "Stationary Head Tee Drill",
                "description": "Place ball on tee; play 30 shadow drives while focusing exclusively on keeping chin still."
            })
        else:
            weaknesses.append("Significant Head Movement: Excessive head bobbing destabilizes gaze and alignment.")
            improvements.append("Avoid falling over or lunging early; let the ball come under your eyes before swinging.")
            drills.append({
                "title": "Coin-Head Balance Drill",
                "description": "Perform front foot strides with a flat marker on cap; retain balance without dropping it."
            })
            focus_areas.append("Head Position & Gaze Fixation")

        # 2. Balance & Stance Evaluation
        if summary.balance_index >= 88.0:
            strengths.append("Solid Base of Support: Center of mass stays nicely balanced between feet.")
        else:
            weaknesses.append("Off-Balance Stance: Center of mass drifts outside foot base during downswing.")
            improvements.append("Widen stance slightly and maintain flex in knees to lower center of gravity.")
            drills.append({
                "title": "Wide Stance Drop-Ball Drill",
                "description": "Practice front foot drives from a firm base with partner dropping ball from waist height."
            })
            focus_areas.append("Base Stability & Footwork Balance")

        # 3. Spine Posture & Body Lean
        if 10.0 <= summary.max_spine_lean_deg <= 28.0:
            strengths.append("Optimal Spine Angle: Controlled forward body lean over the front knee.")
        elif summary.max_spine_lean_deg > 28.0:
            weaknesses.append("Excessive Body Collapse: Leaning too far forward, breaking vertical alignment.")
            improvements.append("Maintain an upright spine torso during backlift and drive through hips.")
            focus_areas.append("Core Alignment & Spine Control")
        else:
            weaknesses.append("Upright Rigid Spine: Lack of flex and forward leaning into the shot line.")
            improvements.append("Bend front knee slightly more to allow natural spine inclination toward ball.")

        # 4. Weight Transfer
        if summary.weight_transfer_index >= 85.0:
            strengths.append("Effective Weight Transfer: Dynamic shift from back foot to front foot.")
        else:
            weaknesses.append("Passive Weight Shift: Weight remains stuck on back foot at point of impact.")
            improvements.append("Initiate stride with front hip and transfer body weight firmly into ball contact.")
            drills.append({
                "title": "Step-Through Drive Drill",
                "description": "Take an exaggerated step into the drive, walking through after impact."
            })
            focus_areas.append("Dynamic Weight Transfer")

        # 5. Follow-Through & Extension
        if summary.followthrough_quality_score >= 85.0:
            strengths.append("High Follow-Through: Complete extension of hands toward target zone.")
        else:
            weaknesses.append("Restricted Follow-Through: Deceleration or checked swing right after contact.")
            improvements.append("Allow elbows to extend fully and finish bat path high over lead shoulder.")
            drills.append({
                "title": "High-Elbow Extension Finish Drill",
                "description": "Complete full swing path holding high finish for 3 seconds post-contact."
            })
            focus_areas.append("Swing Path Extension & Finish")

        # Guarantee at least 2 strengths/weaknesses/drills
        if len(strengths) < 2:
            strengths.append("Good Overall Kinetic Coordination: Smooth sequence transition between backlift and downswing.")
        if len(weaknesses) < 2:
            weaknesses.append("Minor Rhythm Variance: Opportunity to accelerate downswing timing.")
        if len(drills) < 2:
            drills.append({
                "title": "Shadow Batting Rhythm Drill",
                "description": "Execute 20 repetitive full swings focusing on smooth rhythm and high elbow."
            })

        if not focus_areas:
            focus_areas = ["Consolidating Current Technique", "Bat Speed Acceleration"]

        logger.info(f"Generated coaching report. Overall Grade: {grade} ({score}/100)")
        return CoachingReport(
            technique_grade=grade,
            overall_score=score,
            summary_verdict=verdict,
            strengths=strengths,
            weaknesses=weaknesses,
            actionable_improvements=improvements,
            recommended_drills=drills,
            focus_areas=focus_areas
        )

    @staticmethod
    def _determine_grade(score: float) -> Tuple[str, str]:
        if score >= 90.0:
            return "A+ Elite", "Exceptional technique demonstrating high stability, clean swing path, and balance."
        elif score >= 82.0:
            return "A Professional", "Strong batting mechanics with high consistency and minor refinement areas."
        elif score >= 74.0:
            return "B+ Advanced", "Good foundation with solid kinetic chain; minor balance or head drift adjustments needed."
        elif score >= 65.0:
            return "B Technical Work Needed", "Moderate technique. Focus required on balance base, head position, and follow-through."
        else:
            return "C Fundamental Drills Required", "Significant technical flaws detected in balance, stance, or swing timing."
