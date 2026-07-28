"""
Cricket Pose Analyzer AI - Streamlit Dashboard
==============================================
Modern sports analytics interface for cricket batting technique analysis,
interactive Plotly visualizations, motion heatmaps, shot phase timeline,
AI coaching scorecard, dual video comparison mode, and session history.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
package_dir = Path(__file__).resolve().parent
project_root = package_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

try:
    from cricket_pose_ai.analyzer import CricketPoseAnalyzer, FullAnalysisOutput
    from cricket_pose_ai.comparator import VideoComparator
    from cricket_pose_ai.config import OUTPUT_DIR, TEMP_DIR, THEME_COLORS
    from cricket_pose_ai.session_manager import SessionManager
    from cricket_pose_ai.utils import clean_temp_files, setup_logger
except ImportError:
    from analyzer import CricketPoseAnalyzer, FullAnalysisOutput
    from comparator import VideoComparator
    from config import OUTPUT_DIR, TEMP_DIR, THEME_COLORS
    from session_manager import SessionManager
    from utils import clean_temp_files, setup_logger

logger = setup_logger("CricketPoseAI.App")

# Page Configuration
st.set_page_config(
    page_title="Cricket Pose Analyzer AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Theme Sports Analytics Aesthetic)
st.markdown("""
<style>
    /* Dark Theme Setup */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-bottom: 2px solid #38BDF8;
        padding: 20px 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    
    .main-title {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .sub-title {
        color: #38BDF8;
        font-size: 1.05rem;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom Badge */
    .badge-grade {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes Streamlit session state parameters."""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = CricketPoseAnalyzer()
    if "comparator" not in st.session_state:
        st.session_state.comparator = VideoComparator()
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()


def render_header():
    """Renders main application title banner."""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🏏 Cricket Pose Analyzer AI</div>
        <div class="sub-title">Production Computer Vision & Biomechanical Shot Analysis System</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    init_session_state()
    render_header()

    # Sidebar Options
    with st.sidebar:
        st.header("⚙️ Control Panel")

        mode = st.radio("Select Analysis Mode", ["Single Video Analysis", "Dual Video Comparison", "Session History"])

        st.divider()

        st.subheader("Video Source")
        uploaded_file = st.file_uploader("Upload Cricket Batting Video", type=["mp4", "mov", "avi", "mkv"])

        use_sample = st.checkbox("Or use Synthetic Sample Video Mode")

        st.divider()

        st.subheader("Model Settings")
        conf_thresh = st.slider("Pose Detection Confidence", 0.3, 0.9, 0.5, 0.05)
        enable_bat_yolo = st.checkbox("Enable YOLO Bat Tracking", value=True)

        st.divider()
        st.info("Everything runs 100% offline using OpenCV and MediaPipe.")

    # Main Tabs / Views
    if mode == "Single Video Analysis":
        render_single_analysis_tab(uploaded_file, use_sample)
    elif mode == "Dual Video Comparison":
        render_comparison_tab()
    else:
        render_history_tab()


def render_single_analysis_tab(uploaded_file, use_sample: bool):
    """Single Video Processing and Analytics View."""

    if not uploaded_file and not use_sample:
        st.info("👈 Please upload a cricket batting video (.mp4, .mov, .avi, .mkv) in the sidebar to start analysis.")
        st.image("https://images.unsplash.com/photo-1531415074968-036ba1b575da?auto=format&fit=crop&w=1200&q=80", caption="Cricket Batting Biomechanics Analysis")
        return

    # Trigger Analysis Button
    if st.button("🚀 Analyze Video Action", type="primary", use_container_width=True):
        temp_input_path = TEMP_DIR / "uploaded_input.mp4"

        if uploaded_file:
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            # Generate synthetic demo video if sample selected
            temp_input_path = create_sample_cricket_video(temp_input_path)

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(f"⏳ {msg}")

        try:
            results: FullAnalysisOutput = st.session_state.analyzer.analyze_video(
                video_path=temp_input_path,
                progress_callback=update_progress
            )
            st.session_state.analysis_results = results
            status_text.success("✅ Analysis Complete!")
        except Exception as e:
            st.error(f"Error during video processing: {e}")
            logger.error(f"Video analysis error: {e}", exc_info=True)
            return

    res: Optional[FullAnalysisOutput] = st.session_state.analysis_results

    if res is None:
        return

    # Tabs for analytics
    t_video, t_metrics, t_heatmaps, t_timeline, t_coach, t_downloads = st.tabs([
        "🎥 Video & HUD", "📊 Biomechanics", "🔥 Motion Heatmaps", "⏱️ Shot Timeline", "🏏 AI Coach", "📥 Download Center"
    ])

    # 1. Video & HUD Player
    with t_video:
        st.subheader("Annotated Pose & Side-by-Side Playback")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("##### Processed HUD Video")
            st.video(str(res.processed_video_path))
        with col_v2:
            st.markdown("##### Side-by-Side Comparison")
            st.video(str(res.side_by_side_video_path))

    # 2. Biomechanics Dashboard
    with t_metrics:
        st.subheader("Biomechanical Metrics & Scores")

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.summary.overall_technique_score}</div><div class="metric-lbl">Overall Score</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.summary.balance_index}</div><div class="metric-lbl">Balance Index</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.summary.head_stability_index}</div><div class="metric-lbl">Head Stability</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{res.summary.weight_transfer_index}</div><div class="metric-lbl">Weight Transfer</div></div>', unsafe_allow_html=True)

        st.divider()

        # Plotly Interactive Graphs
        df_records = pd.DataFrame(res.framewise_records)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_angles = px.line(
                df_records, x="frame", y=["right_elbow_angle", "right_knee_angle", "spine_angle"],
                title="Joint Angles Across Stroke (Degrees)",
                template="plotly_dark",
                labels={"value": "Angle (°)", "frame": "Frame Index"}
            )
            st.plotly_chart(fig_angles, use_container_width=True)

        with col_g2:
            fig_speed = px.line(
                df_records, x="frame", y=["wrist_velocity_px_s", "bat_velocity_px_s"],
                title="Kinetic Velocity Profiles (px/sec)",
                template="plotly_dark",
                labels={"value": "Velocity (px/s)", "frame": "Frame Index"}
            )
            st.plotly_chart(fig_speed, use_container_width=True)

    # 3. Motion Heatmaps
    with t_heatmaps:
        st.subheader("Spatial Density Heatmaps")
        st.markdown("Visual spatial intensity distributions for key landmarks during the batting shot.")

        cols_hm = st.columns(3)
        hm_keys = list(res.heatmap_paths.keys())

        for idx, key in enumerate(hm_keys):
            path = res.heatmap_paths[key]
            with cols_hm[idx % 3]:
                st.image(str(path), use_container_width=True)

    # 4. Interactive Shot Timeline
    with t_timeline:
        st.subheader("Shot Phase Segmentation Timeline")
        tl = res.summary.timeline

        st.markdown(f"""
        - **Stance / Address**: Frame 0 ({0:.2f}s)
        - **Backlift Peak**: Frame {tl.backlift_peak_frame} ({tl.backlift_time_sec:.2f}s)
        - **Downswing**: Frame {tl.downswing_start_frame}
        - **Impact**: Frame {tl.impact_frame} ({tl.impact_time_sec:.2f}s)
        - **Follow-Through Peak**: Frame {tl.followthrough_peak_frame} ({tl.followthrough_time_sec:.2f}s)
        - **Finish**: Frame {tl.finish_frame}
        """)

        st.divider()
        frame_slider = st.slider("Inspect Frame Step", 0, res.summary.total_frames - 1, tl.impact_frame)
        if frame_slider < len(res.framewise_records):
            rec = res.framewise_records[frame_slider]
            st.json(rec)

    # 5. AI Coaching Scorecard
    with t_coach:
        st.subheader("AI Technique Evaluation & Action Plan")
        st.markdown(f'<div class="badge-grade">Grade: {res.coaching.technique_grade}</div>', unsafe_allow_html=True)
        st.markdown(f"**Verdict:** {res.coaching.summary_verdict}")

        st.divider()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 💪 Key Strengths")
            for st_item in res.coaching.strengths:
                st.success(st_item)

        with col_c2:
            st.markdown("#### ⚠️ Technical Weaknesses")
            for wk_item in res.coaching.weaknesses:
                st.warning(wk_item)

        st.markdown("#### 🏏 Recommended Coaching Drills")
        for drill in res.coaching.recommended_drills:
            st.info(f"**{drill['title']}**: {drill['description']}")

    # 6. Download Center
    with t_downloads:
        st.subheader("Export Center & Generated Artifacts")

        d1, d2, d3, d4 = st.columns(4)

        if res.pdf_report_path.exists():
            with open(res.pdf_report_path, "rb") as f:
                d1.download_button("📥 PDF Report", f.read(), "analysis_report.pdf", "application/pdf", use_container_width=True)

        if res.processed_video_path.exists():
            with open(res.processed_video_path, "rb") as f:
                d2.download_button("📥 Annotated MP4", f.read(), "processed_video.mp4", "video/mp4", use_container_width=True)

        if res.csv_path.exists():
            with open(res.csv_path, "rb") as f:
                d3.download_button("📥 Metrics CSV", f.read(), "joint_angles.csv", "text/csv", use_container_width=True)

        if res.zip_path.exists():
            with open(res.zip_path, "rb") as f:
                d4.download_button("📦 Full Analysis ZIP", f.read(), "analysis.zip", "application/zip", use_container_width=True)


def render_comparison_tab():
    """Dual Video Comparison Mode Interface."""
    st.subheader("⚔️ Dual Video Comparison Mode")
    st.markdown("Upload two videos to compare technique progression, balance, and bat speed.")

    col1, col2 = st.columns(2)
    with col1:
        v1 = st.file_uploader("Upload Video A (Baseline)", type=["mp4", "mov"], key="vid_a")
    with col2:
        v2 = st.file_uploader("Upload Video B (New Attempt)", type=["mp4", "mov"], key="vid_b")

    if st.button("⚔️ Run Dual Video Comparison", type="primary"):
        if not v1 or not v2:
            st.warning("Please upload both Video A and Video B.")
            return

        st.info("Running side-by-side comparative analysis pipeline...")
        # Execute analysis on both videos
        p1 = TEMP_DIR / "vid_a.mp4"
        p2 = TEMP_DIR / "vid_b.mp4"
        with open(p1, "wb") as f: f.write(v1.getbuffer())
        with open(p2, "wb") as f: f.write(v2.getbuffer())

        res1 = st.session_state.analyzer.analyze_video(p1)
        res2 = st.session_state.analyzer.analyze_video(p2)

        delta, pdf_comp = st.session_state.comparator.compare_summaries(
            summary_a=res1.summary,
            summary_b=res2.summary,
            label_a=v1.name,
            label_b=v2.name
        )

        st.success(f"Comparison Complete! Superior Performance: **{delta.better_video_label}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Score Delta", f"{res2.summary.overall_technique_score}", f"{delta.score_diff:+.1f}")
        m2.metric("Balance Delta", f"{res2.summary.balance_index}", f"{delta.balance_diff:+.1f}")
        m3.metric("Head Stability Delta", f"{res2.summary.head_stability_index}", f"{delta.head_stability_diff:+.1f}")
        m4.metric("Bat Speed Delta", f"{res2.summary.max_bat_speed_px} px/s", f"{delta.bat_speed_diff:+.1f} px/s")

        if pdf_comp.exists():
            with open(pdf_comp, "rb") as f:
                st.download_button("📥 Download Comparison PDF", f.read(), "comparison_report.pdf", "application/pdf")


def render_history_tab():
    """Session History View."""
    st.subheader("📜 Session Analysis History")
    sessions = st.session_state.session_manager.get_all_sessions()

    if not sessions:
        st.info("No past sessions found in database.")
        return

    df_sess = pd.DataFrame(sessions)
    st.dataframe(df_sess, use_container_width=True)

    if len(df_sess) > 1:
        fig_hist = px.line(
            df_sess, x="timestamp", y="overall_score",
            title="Overall Score Progression Over Time",
            markers=True, template="plotly_dark"
        )
        st.plotly_chart(fig_hist, use_container_width=True)


def create_sample_cricket_video(output_path: Path) -> Path:
    """Generates a synthetic cricket batting video for dry-run testing."""
    import cv2
    w, h, fps = 640, 480, 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    total_f = 60
    for i in range(total_f):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 24, 20)  # Dark slate background

        # Draw synthetic batsman silhouette
        t = i / total_f
        head_x = int(320 + np.sin(t * np.pi) * 15)
        head_y = 120
        cv2.circle(img, (head_x, head_y), 25, (200, 200, 200), -1)

        # Torso
        cv2.line(img, (head_x, head_y + 25), (head_x, head_y + 140), (180, 180, 180), 8)

        # Arms swinging bat
        hand_x = int(320 + np.cos(t * np.pi * 2) * 60)
        hand_y = int(220 + np.sin(t * np.pi * 2) * 40)
        cv2.line(img, (head_x, head_y + 40), (hand_x, hand_y), (180, 180, 180), 5)

        # Bat
        bat_tip_x = hand_x + 80
        bat_tip_y = hand_y - 40
        cv2.line(img, (hand_x, hand_y), (bat_tip_x, bat_tip_y), (0, 215, 255), 6)

        out.write(img)

    out.release()
    return output_path


if __name__ == "__main__":
    main()
