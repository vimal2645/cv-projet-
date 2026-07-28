"""
Cricket Pose Analyzer AI - Session History Manager Module
=========================================================
Manages session history, persistence in JSON/SQLite format, retrieval of
past analysis records, and session progress tracking.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cricket_pose_ai.config import SESSIONS_FILE
from cricket_pose_ai.motion_analyzer import MotionSummaryMetrics
from cricket_pose_ai.utils import setup_logger

logger = setup_logger("CricketPoseAI.SessionManager")


@dataclass
class SessionRecord:
    """Represents a saved video analysis session."""
    session_id: str
    video_name: str
    timestamp: str
    overall_score: float
    technique_grade: str
    duration_sec: float
    total_frames: int
    head_stability: float
    balance_index: float
    weight_transfer: float
    max_bat_speed: float
    artifacts_dir: str


class SessionManager:
    """Manages history of video analysis sessions."""

    def __init__(self, db_file: Optional[Path] = None):
        self.db_file = db_file or SESSIONS_FILE
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_file.exists():
            self._save_data([])

    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads sessions JSON array."""
        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading session database: {e}")
            return []

    def _save_data(self, data: List[Dict[str, Any]]):
        """Saves sessions JSON array."""
        try:
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving session database: {e}")

    def save_session(
        self,
        video_name: str,
        summary: MotionSummaryMetrics,
        grade: str,
        artifacts_dir: str = ""
    ) -> SessionRecord:
        """Saves a new analysis session record."""
        session_id = f"SESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = SessionRecord(
            session_id=session_id,
            video_name=video_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_score=summary.overall_technique_score,
            technique_grade=grade,
            duration_sec=summary.duration_sec,
            total_frames=summary.total_frames,
            head_stability=summary.head_stability_index,
            balance_index=summary.balance_index,
            weight_transfer=summary.weight_transfer_index,
            max_bat_speed=summary.max_bat_speed_px,
            artifacts_dir=artifacts_dir
        )

        data = self._load_data()
        data.insert(0, asdict(record))
        self._save_data(data)
        logger.info(f"Saved session {session_id} for video '{video_name}'")
        return record

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Returns list of all stored session records."""
        return self._load_data()

    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Finds session by unique ID."""
        for s in self._load_data():
            if s.get("session_id") == session_id:
                return s
        return None

    def clear_history(self):
        """Clears all stored sessions."""
        self._save_data([])
        logger.info("Cleared all session history.")
