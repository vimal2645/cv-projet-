"""
Cricket Pose Analyzer AI - Video Processor Module
==================================================
OpenCV video reader, properties metadata extractor, frame-by-frame processing
loop, multithreaded rendering, side-by-side video generation, and MP4 exporter.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from cricket_pose_ai.angle_calculator import BiomechanicsCalculator, FrameBiomechanics
from cricket_pose_ai.bat_detector import CricketBatDetector, BatDetectionResult
from cricket_pose_ai.config import TEMP_DIR
from cricket_pose_ai.pose_detector import CricketPoseDetector, Landmark
from cricket_pose_ai.utils import ensure_h264_encoding, setup_logger
from cricket_pose_ai.visualizer import PoseVisualizer

logger = setup_logger("CricketPoseAI.VideoProcessor")


@dataclass
class VideoMetadata:
    """Video file metadata container."""
    filepath: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    codec: str


class VideoProcessor:
    """Engine for video reading, landmark extraction, and video generation."""

    def __init__(self):
        self.pose_detector = CricketPoseDetector()
        self.bat_detector = CricketBatDetector()
        self.biomech_calculator = BiomechanicsCalculator()
        self.visualizer = PoseVisualizer()

    @staticmethod
    def extract_metadata(video_path: Path) -> Optional[VideoMetadata]:
        """Extracts resolution, frame rate, frame count, and duration from video file."""
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Unable to open video: {video_path}")
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()
        return VideoMetadata(
            filepath=video_path,
            width=w,
            height=h,
            fps=fps,
            total_frames=total_frames,
            duration_sec=round(duration, 2),
            codec="H264"
        )

    def process_video_sequence(
        self,
        video_path: Path,
        output_video_path: Path,
        side_by_side_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[
        List[FrameBiomechanics],
        List[Optional[Dict[str, Landmark]]],
        List[BatDetectionResult],
        VideoMetadata,
        np.ndarray
    ]:
        """
        Executes frame-by-frame analysis and outputs annotated video file.
        Returns biomechanics list, landmarks history, bat history, metadata, and representative background frame.
        """
        meta = self.extract_metadata(video_path)
        if not meta:
            raise ValueError(f"Could not read video metadata from {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        # Prefer H264 / avc1 codec for web compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            meta.fps,
            (meta.width, meta.height)
        )

        sbs_writer = None
        if side_by_side_path:
            sbs_writer = cv2.VideoWriter(
                str(side_by_side_path),
                fourcc,
                meta.fps,
                (meta.width * 2, meta.height)
            )

        biomechanics_list: List[FrameBiomechanics] = []
        landmarks_history: List[Optional[Dict[str, Landmark]]] = []
        bat_history: List[BatDetectionResult] = []

        representative_frame = None
        frame_idx = 0
        last_batsman_hip: Optional[Tuple[float, float]] = None

        logger.info(f"Processing video {video_path.name} ({meta.total_frames} frames)...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if representative_frame is None and frame_idx == meta.total_frames // 3:
                    representative_frame = frame.copy()

                # 1. Pose Detection
                raw_landmarks, _ = self.pose_detector.process_frame(frame)

                # Filter Pose to Lock Strictly onto Batsman (reject jumps to bowler/keeper)
                landmarks = raw_landmarks
                if raw_landmarks and ("LEFT_HIP" in raw_landmarks or "RIGHT_HIP" in raw_landmarks):
                    lh = raw_landmarks.get("LEFT_HIP")
                    rh = raw_landmarks.get("RIGHT_HIP")
                    cur_hip_x = (lh.x + rh.x) / 2.0 if (lh and rh) else (lh.x if lh else rh.x)
                    cur_hip_y = (lh.y + rh.y) / 2.0 if (lh and rh) else (lh.y if lh else rh.y)

                    if last_batsman_hip is not None:
                        dist = np.hypot(cur_hip_x - last_batsman_hip[0], cur_hip_y - last_batsman_hip[1])
                        # If detection jumped far away (> 0.32 norm dist, e.g., bowler entering frame edge)
                        if dist > 0.32 and len(landmarks_history) > 0 and landmarks_history[-1] is not None:
                            logger.debug(f"Frame {frame_idx}: Pose jumped to non-batsman (dist={dist:.2f}). Re-using previous batsman position.")
                            landmarks = landmarks_history[-1]
                        else:
                            last_batsman_hip = (
                                0.75 * cur_hip_x + 0.25 * last_batsman_hip[0],
                                0.75 * cur_hip_y + 0.25 * last_batsman_hip[1]
                            )
                    else:
                        last_batsman_hip = (cur_hip_x, cur_hip_y)

                landmarks_history.append(landmarks)

                # 2. Bat Detection
                bat_res = self.bat_detector.detect_bat(frame, landmarks)
                bat_history.append(bat_res)

                # 3. Biomechanics Calculation
                if landmarks:
                    bm = self.biomech_calculator.compute_frame_biomechanics(
                        landmarks=landmarks,
                        frame_idx=frame_idx,
                        img_w=meta.width,
                        img_h=meta.height
                    )
                else:
                    bm = FrameBiomechanics(
                        frame_idx=frame_idx,
                        left_elbow_angle=140.0, right_elbow_angle=140.0,
                        left_shoulder_angle=45.0, right_shoulder_angle=45.0,
                        left_knee_angle=165.0, right_knee_angle=165.0,
                        left_hip_angle=160.0, right_hip_angle=160.0,
                        spine_angle=15.0, neck_angle=170.0, head_tilt=0.0, body_lean=15.0,
                        foot_distance_px=100.0, stance_width_ratio=1.0,
                        shoulder_alignment_angle=0.0, hip_alignment_angle=0.0,
                        com_x=0.5, com_y=0.5,
                        com_px=meta.width // 2, com_py=meta.height // 2
                    )

                biomechanics_list.append(bm)

                phase_lbl = "Analyzing Shot..."

                # 4. Render Annotations
                annotated = self.visualizer.draw_frame_annotations(
                    frame_bgr=frame,
                    landmarks=landmarks,
                    biomechanics=bm,
                    bat_result=bat_res,
                    phase_label=phase_lbl,
                    frame_idx=frame_idx,
                    total_frames=meta.total_frames,
                    fps=meta.fps
                )

                out_writer.write(annotated)

                if sbs_writer:
                    sbs_frame = np.hstack((frame, annotated))
                    sbs_writer.write(sbs_frame)

                frame_idx += 1

                if progress_callback and meta.total_frames > 0:
                    pct = float(frame_idx / meta.total_frames)
                    progress_callback(pct, f"Analyzing frame {frame_idx}/{meta.total_frames}...")

        finally:
            cap.release()
            out_writer.release()
            if sbs_writer:
                sbs_writer.release()
            self.pose_detector.release()

        # Re-encode output videos to H.264 for HTML5 web browser compatibility
        ensure_h264_encoding(output_video_path)
        if side_by_side_path:
            ensure_h264_encoding(side_by_side_path)

        if representative_frame is None:
            representative_frame = np.zeros((meta.height, meta.width, 3), dtype=np.uint8)

        logger.info(f"Video processing complete. Output saved to {output_video_path}")
        return biomechanics_list, landmarks_history, bat_history, meta, representative_frame
