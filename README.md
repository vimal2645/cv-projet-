# 🏏 Cricket Pose Analyzer AI

> **A Production-Grade Computer Vision & Biomechanics System for Cricket Batting Technique Analysis**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange.svg)](https://google.github.io/mediapipe/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Executive Summary

**Cricket Pose Analyzer AI** is an advanced offline Computer Vision application built using Python, OpenCV, MediaPipe Pose, Ultralytics YOLO, and Streamlit. It performs frame-by-frame anatomical landmark detection on cricket batting videos, calculates kinematic & biomechanical posture metrics, tracks bat trajectory & swing speed, segments shot phases (*Stance, Backlift, Downswing, Impact, Follow-Through, Finish*), renders spatial motion heatmaps, and generates publication-grade PDF coaching reports alongside an interactive web dashboard.

---

## 🚀 Key Features

- **Full-Body Pose Estimation**: MediaPipe Pose detection tracking all 33 body keypoints with temporal smoothing and confidence masking.
- **Hybrid Bat & Swing Arc Detection**: YOLOv8-based cricket bat object detection with graceful fallbacks to HSV color segmentation and wrist direction vector extension.
- **Biomechanical Math Engine**: Real-time 2D joint angle calculation (elbow extension, knee flexion, spine inclination, neck tilt, head drift, foot stance ratio, and segmental Center of Mass approximation).
- **Automated Shot Phase Segmentation**: Signal processing algorithms identifying *Stance*, *Backlift Peak*, *Downswing*, *Impact*, *Follow-Through*, and *Finish*.
- **Spatial Density Heatmaps**: 2D Gaussian kernel density heatmaps overlaying trajectories of Wrists, Bat Tip, Head, Foot Placement, and Center of Mass.
- **Deterministic Rule-Based AI Coach**: Offline coaching engine providing technique scores, strengths, weaknesses, actionable suggestions, and drill recommendations.
- **Dual Video Comparison Mode**: Side-by-side comparison of 2 batting videos with metric delta calculations and comparative PDF export.
- **Session History & Analytics Database**: Local session database storing analysis runs with historical progression charts.
- **Complete Export Suite**: Exports `processed_video.mp4`, `side_by_side_video.mp4`, `analysis_report.pdf`, `joint_angles.csv`, `metrics.json`, high-res graphs PNGs, and a single-click `analysis.zip` archive.

---

## 🛠️ Tech Stack & Dependencies

- **Core Language**: Python 3.12
- **Computer Vision**: OpenCV (`opencv-python`), MediaPipe Pose (`mediapipe`), Ultralytics YOLO (`ultralytics`)
- **Scientific Computing & Signal Processing**: NumPy (`numpy`), SciPy (`scipy`), Pandas (`pandas`)
- **Data Visualization**: Matplotlib (`matplotlib`), Plotly (`plotly`)
- **PDF Report Generation**: ReportLab (`reportlab`)
- **Web UI & Dashboard**: Streamlit (`streamlit`)

---

## 🏗️ System Architecture

```
                                +---------------------------+
                                |    User Input Video       |
                                |  (MP4 / MOV / AVI / MKV)  |
                                +-------------+-------------+
                                              |
                                              v
                                +-------------+-------------+
                                |     Video Processor       |
                                |  (OpenCV Frame Loop)      |
                                +------+--------------+------+
                                       |              |
                       +---------------+              +---------------+
                       |                                              |
                       v                                              v
         +-------------+-------------+                  +-------------+-------------+
         |    MediaPipe Pose Tracker |                  |   Cricket Bat Detector     |
         |   (33 Body Landmarks)     |                  | (YOLO / HSV / Extension)  |
         +-------------+-------------+                  +-------------+-------------+
                       |                                              |
                       +---------------+--------------+---------------+
                                       |
                                       v
                        +--------------+--------------+
                        |  Biomechanics & Angle Math  |
                        | (Joints, Spine, Head, COM)  |
                        +--------------+--------------+
                                       |
                                       v
                        +--------------+--------------+
                        |   Motion & Kinematics Engine |
                        | (Velocity, Shot Phase Seg)  |
                        +--------------+--------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
          v                            v                            v
+---------+----------+       +---------+----------+       +---------+----------+
|  Spatial Heatmaps  |       | Rule-Based AI Coach|       | OpenCV Video HUD   |
| (Gaussian Density) |       |  (Coaching Engine) |       | & Plotly Visuals   |
+---------+----------+       +---------+----------+       +---------+----------+
          |                            |                            |
          +----------------------------+----------------------------+
                                       |
                                       v
                        +--------------+--------------+
                        | ReportLab PDF & Zip Exporter|
                        | (PDF, CSV, JSON, ZIP Output)|
                        +-----------------------------+
```

---

## 📐 Biomechanical Formulas & Mathematics

### 1. 2D Three-Point Joint Angle
For joint vertex $B$ connected to points $A$ and $C$:
$$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|}\right) \times \frac{180}{\pi}$$

### 2. Segmental Center of Mass (COM) Approximation
$$\text{COM}_{\text{xy}} = w_{\text{head}} P_{\text{head}} + w_{\text{torso}} P_{\text{torso}} + w_{\text{legs}} P_{\text{legs}} + w_{\text{arms}} P_{\text{arms}}$$
*Where $w_{\text{torso}} = 0.48, w_{\text{legs}} = 0.30, w_{\text{arms}} = 0.14, w_{\text{head}} = 0.08$.*

### 3. Spine Inclination Angle
Angle of segment between Mid-Hip and Mid-Shoulder relative to vertical axis:
$$\theta_{\text{spine}} = \left| \arctan2\left(\Delta x_{\text{spine}}, -\Delta y_{\text{spine}}\right) \right| \times \frac{180}{\pi}$$

### 4. Head Stability Index
Calculated from spatial coordinate variance during downswing and impact:
$$\text{Drift Ratio} = \frac{\sigma_x(\text{Head}) + \sigma_y(\text{Head})}{H_{\text{frame}}}$$
$$\text{Score}_{\text{head}} = \max\left(30.0, 100.0 - 1000.0 \times \text{Drift Ratio}\right)$$

---

## 📦 Project Directory Structure

```
pose-nalytics/
├── .gitignore             # Git ignore configuration
├── README.md              # Technical project documentation
├── requirements.txt       # Python package dependencies
├── test_pipeline.py       # End-to-end integration test runner
├── yolov8n.pt             # Pre-trained YOLO weights
└── cricket_pose_ai/       # Main package directory
    ├── __init__.py        # Package initialization
    ├── app.py             # Streamlit Analytics Dashboard UI
    ├── analyzer.py        # Master pipeline orchestrator
    ├── pose_detector.py   # MediaPipe Pose keypoint wrapper
    ├── bat_detector.py    # Bat detection (YOLO / HSV / Wrist extension)
    ├── angle_calculator.py# Vector geometry & biomechanics calculator
    ├── motion_analyzer.py # Kinematics, velocity, shot phase segmentation
    ├── heatmap_generator.py# Spatial motion density heatmaps
    ├── coaching_engine.py # Rule-based AI coaching system
    ├── visualizer.py      # OpenCV HUD overlays & Plotly chart generator
    ├── video_processor.py # Frame processing loop & video writer
    ├── report_generator.py# ReportLab PDF report builder
    ├── session_manager.py # Session history database manager
    ├── comparator.py      # Dual video comparison engine
    ├── utils.py           # File I/O, formatting, ZIP archive builder
    ├── config.py          # Thresholds & theme design tokens
    ├── assets/            # App media assets
    ├── temp/              # Temporary buffer directory
    └── outputs/           # PDF reports, annotated videos, CSV, JSON exports
```

---

## ⚡ Installation & Setup Guide

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/pose-nalytics.git
cd pose-nalytics

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Test (Optional Verification)
```bash
python test_pipeline.py
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run cricket_pose_ai/app.py
```

---

## 📸 Dashboard Preview & Visuals

*(Placeholder screenshots for portfolio demonstration)*

| Video HUD & Pose Skeleton | Spatial Density Heatmap |
| :---: | :---: |
| ![HUD Preview](https://via.placeholder.com/500x300.png?text=OpenCV+Pose+Skeleton+HUD) | ![Heatmap Preview](https://via.placeholder.com/500x300.png?text=Wrist+Spatial+Heatmap) |

| Shot Phase Timeline | AI Coaching Scorecard |
| :---: | :---: |
| ![Timeline Preview](https://via.placeholder.com/500x300.png?text=Interactive+Shot+Timeline) | ![Coaching Preview](https://via.placeholder.com/500x300.png?text=PDF+Coaching+Scorecard) |

---

## 💼 Resume Portfolio Project Description

> **Cricket Pose Analyzer AI**  
> *Engineered an offline Computer Vision & Biomechanics analysis platform in Python to analyze cricket batting techniques from video inputs.*
> - **Pose Tracking & Kinematics**: Utilized MediaPipe Pose to track 33 anatomical body keypoints, calculating frame-by-frame joint angles (elbows, knees, spine inclination, head drift, stance width ratio) and segmental Center of Mass (COM).
> - **Bat Tracking**: Developed a hybrid detection pipeline using YOLOv8 object detection with HSV color contour and wrist direction vector fallbacks to measure bat angle, swing speed, and arc.
> - **Shot Phase Segmentation**: Implemented signal smoothing (SciPy Savitzky-Golay) and velocity thresholding to segment shots into *Stance, Backlift, Downswing, Impact, and Follow-Through*.
> - **AI Coaching Engine**: Built a deterministic, rule-based coaching system evaluating balance, head stability, and weight transfer to generate technique grades, strengths, weaknesses, and drill prescriptions.
> - **Interactive Dashboard & PDF Reports**: Developed a dark-themed Streamlit application with Plotly visualizations, dual video comparison mode, and automated ReportLab PDF report generation.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
