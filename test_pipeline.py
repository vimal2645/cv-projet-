"""
Test script for verifying end-to-end processing pipeline of Cricket Pose Analyzer AI.
"""

import sys
from pathlib import Path

# Ensure package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cricket_pose_ai.analyzer import CricketPoseAnalyzer
from cricket_pose_ai.app import create_sample_cricket_video
from cricket_pose_ai.config import OUTPUT_DIR, TEMP_DIR
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.TestPipeline")


def run_pipeline_test():
    logger.info("--- Starting End-to-End Dry Run Test ---")

    # 1. Create synthetic test video
    test_video_path = TEMP_DIR / "synthetic_test_batting.mp4"
    logger.info(f"Generating synthetic test video at {test_video_path}...")
    create_sample_cricket_video(test_video_path)

    # 2. Instantiate master analyzer
    analyzer = CricketPoseAnalyzer()

    # 3. Progress callback logger
    def progress_handler(pct: float, msg: str):
        logger.info(f"[{pct*100:5.1f}%] {msg}")

    # 4. Run full pipeline
    results = analyzer.analyze_video(
        video_path=test_video_path,
        progress_callback=progress_handler
    )

    logger.info("--- Verification Results ---")
    logger.info(f"Overall Technique Score: {results.summary.overall_technique_score}")
    logger.info(f"Technique Grade: {results.coaching.technique_grade}")
    logger.info(f"Processed Video: {results.processed_video_path} (Exists: {results.processed_video_path.exists()})")
    logger.info(f"PDF Report: {results.pdf_report_path} (Exists: {results.pdf_report_path.exists()})")
    logger.info(f"Metrics CSV: {results.csv_path} (Exists: {results.csv_path.exists()})")
    logger.info(f"Metrics JSON: {results.json_path} (Exists: {results.json_path.exists()})")
    logger.info(f"Full Analysis ZIP: {results.zip_path} (Exists: {results.zip_path.exists()})")

    assert results.processed_video_path.exists(), "Processed video file missing!"
    assert results.pdf_report_path.exists(), "PDF report missing!"
    assert results.csv_path.exists(), "CSV metrics missing!"
    assert results.json_path.exists(), "JSON metrics missing!"
    assert results.zip_path.exists(), "ZIP archive missing!"

    logger.info("✅ ALL PIPELINE CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_pipeline_test()
