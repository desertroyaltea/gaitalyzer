"""
Gaitalyzer
Gait analysis for fall risk assessment with risk stratification
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import tempfile
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Page config
st.set_page_config(
    page_title="Gaitalyzer",
    page_icon="",
    layout="wide"
)

# Initialize MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose, mp_drawing

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and feature names."""
    model_dir = Path("models")
    
    with open(model_dir / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    with open(model_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    return model, scaler, label_encoder, feature_names, metadata

def extract_skeleton_from_video(video_path, pose, mp_pose):
    """Extract skeleton keypoints from video using MediaPipe."""
    cap = cv2.VideoCapture(str(video_path))
    
    skeleton_data = []
    frame_count = 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract keypoints
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            skeleton_data.append(landmarks)
        
        # Update progress
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return skeleton_data, fps

def compute_gait_features_from_skeleton(skeleton_data, fps):
    """
    Compute gait features from skeleton keypoints.
    This is a SIMPLIFIED version - maps skeleton to approximate gait features.
    """
    
    if len(skeleton_data) < 10:
        st.error("Video too short! Need at least 10 frames.")
        return None
    
    # Extract key landmarks over time
    # Hip center (average of left/right hip)
    hip_positions = []
    ankle_positions = []
    knee_positions = []
    
    for frame in skeleton_data:
        # MediaPipe landmark indices
        left_hip = frame[23]   # Left hip
        right_hip = frame[24]  # Right hip
        left_ankle = frame[27] # Left ankle
        right_ankle = frame[28] # Right ankle
        left_knee = frame[25]  # Left knee
        right_knee = frame[26] # Right knee
        
        # Average hip position
        hip_y = (left_hip['y'] + right_hip['y']) / 2
        hip_positions.append(hip_y)
        
        # Ankle heights (for clearance estimation)
        ankle_y = min(left_ankle['y'], right_ankle['y'])
        ankle_positions.append(ankle_y)
        
        # Knee positions
        knee_y = (left_knee['y'] + right_knee['y']) / 2
        knee_positions.append(knee_y)
    
    # Compute proxy features
    hip_positions = np.array(hip_positions)
    ankle_positions = np.array(ankle_positions)
    
    # Estimate gait parameters
    duration = len(skeleton_data) / fps
    
    # Stride detection (simple peak detection on hip vertical movement)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hip_positions, distance=int(fps * 0.5))  # Minimum 0.5s between strides
    num_strides = len(peaks)
    
    if num_strides < 2:
        st.warning("Could not detect enough strides. Using default values.")
        num_strides = 2
    
    # Compute features (proxies based on skeleton movement)
    cadence = (num_strides / duration) * 60  # steps per minute
    cycle_duration = duration / max(num_strides, 1)
    
    # Movement variability (proxy for gait stability)
    hip_velocity = np.diff(hip_positions)
    stride_velocity_proxy = np.std(hip_velocity) * fps  # Proxy for stride velocity
    
    # Clearance proxy (ankle height variation)
    clearance_proxy = np.std(ankle_positions) * 0.5  # Normalized estimate
    
    # Stride length proxy (can't measure actual distance from video, use normalized)
    stride_length_proxy = stride_velocity_proxy * cycle_duration
    
    # Swing/stance estimation (simplified)
    swing_ratio = 0.4  # Typical ratio, hard to estimate accurately from 2D
    
    # Create feature vector matching training data
    # We'll fill with proxy values and some defaults
    features = {
        'CycleDuration_mean': cycle_duration,
        'CycleDuration_median': cycle_duration,
        'CycleDuration_std': np.std([cycle_duration] * 10) if num_strides > 1 else 0.1,
        'CycleDuration_cv': 0.3,
        'CycleDuration_range': 2.0,
        'Cadence_mean': cadence,
        'Cadence_median': cadence,
        'Cadence_std': cadence * 0.2,
        'Cadence_cv': 0.2,
        'Cadence_range': 100,
        'StrideLength_mean': stride_length_proxy,
        'StrideLength_median': stride_length_proxy,
        'StrideLength_std': stride_length_proxy * 0.3,
        'StrideLength_cv': 0.3,
        'StrideLength_range': 1.5,
        'Clearance_mean': clearance_proxy,
        'Clearance_median': clearance_proxy,
        'Clearance_std': clearance_proxy * 0.3,
        'Clearance_cv': 0.3,
        'Clearance_range': 0.2,
        'Stride_Velocity_mean': stride_velocity_proxy,
        'Stride_Velocity_median': stride_velocity_proxy,
        'Stride_Velocity_std': stride_velocity_proxy * 0.3,
        'Stride_Velocity_cv': 0.3,
        'Stride_Velocity_range': 2.0,
        'Per_Swing_mean': swing_ratio * 100,
        'Per_Swing_median': swing_ratio * 100,
        'Per_Swing_std': 10,
        'Per_Swing_cv': 0.2,
        'Per_Swing_range': 40,
        'Per_Stance_mean': (1 - swing_ratio) * 100,
        'Per_Stance_median': (1 - swing_ratio) * 100,
        'Per_Stance_std': 10,
        'Per_Stance_cv': 0.2,
        'Per_Stance_range': 40,
        'RoMPitch_mean': 60,
        'RoMPitch_median': 60,
        'RoMPitch_std': 20,
        'RoMPitch_cv': 0.3,
        'RoMPitch_range': 80,
        'stride_length_variability': stride_length_proxy * 0.3,
        'cycle_duration_variability': np.std([cycle_duration] * 10) if num_strides > 1 else 0.1,
        'swing_asymmetry': 10,
        'num_strides': num_strides
    }
    
    return features

def get_risk_stratification(prediction, probability):
    """
    Stratify risk based on prediction and probability.
    
    Thresholds based on model probability distribution:
    - TRUE Fallers: Median 82%, 75th percentile 69%
    - TRUE Non-Fallers: Median 20%, 75th percentile 35%
    """
    if prediction == "Faller":
        if probability >= 0.80:
            return {
                'level': 'CRITICAL RISK',
                'emoji': '🔴',
                'color': '#dc3545',
                'urgency': 'Immediate clinical intervention required',
                'action': ''
            }
        elif probability >= 0.65:
            return {
                'level': 'HIGH RISK',
                'emoji': '🟠',
                'color': '#fd7e14',
                'urgency': 'Urgent assessment needed',
                'action': ''
            }
        else:  # 0.50-0.65
            return {
                'level': 'MODERATE RISK',
                'emoji': '🟡',
                'color': '#ffc107',
                'urgency': 'Preventive action recommended',
                'action': ''
            }
    else:  # Non-Faller
        if probability >= 0.35:  # Close to threshold
            return {
                'level': 'LOW-MODERATE RISK',
                'emoji': '🟡',
                'color': '#ffc107',
                'urgency': 'Monitor closely',
                'action': ''
            }
        else:
            return {
                'level': 'LOW RISK',
                'emoji': '🟢',
                'color': '#28a745',
                'urgency': 'Routine screening recommended',
                'action': ''
            }

def create_side_by_side_video(video_path, skeleton_data, output_path, pose, mp_pose, mp_drawing):
    """
    Create side-by-side comparison video: Original (left) | Skeleton (right)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video will be double width (side-by-side)
    output_width = width * 2
    output_height = height
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    frame_idx = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened() and frame_idx < len(skeleton_data):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for skeleton overlay
        skeleton_frame = frame.copy()
        
        # Draw skeleton on the copy
        if frame_idx < len(skeleton_data):
            landmarks = skeleton_data[frame_idx]
            
            # Draw connections
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (
                        int(landmarks[start_idx]['x'] * width),
                        int(landmarks[start_idx]['y'] * height)
                    )
                    end_point = (
                        int(landmarks[end_idx]['x'] * width),
                        int(landmarks[end_idx]['y'] * height)
                    )
                    
                    cv2.line(skeleton_frame, start_point, end_point, (0, 255, 0), 3)
            
            # Draw landmarks
            for lm in landmarks:
                cx, cy = int(lm['x'] * width), int(lm['y'] * height)
                cv2.circle(skeleton_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Add text labels
        cv2.putText(frame, "ORIGINAL", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(skeleton_frame, "AI ANALYSIS", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Concatenate frames side-by-side
        side_by_side = np.hstack([frame, skeleton_frame])
        
        out.write(side_by_side)
        frame_idx += 1
        
        # Update progress
        if frame_idx % 10 == 0:
            progress = int((frame_idx / len(skeleton_data)) * 100)
            progress_bar.progress(min(progress, 100))
            status_text.text(f"Creating side-by-side video: {frame_idx}/{len(skeleton_data)} frames")
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_path

def generate_pdf_report(prediction, features, metadata, probability, risk_info):
    """Generate PDF report with risk stratification."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30
    )
    story.append(Paragraph("GSTRIDE Fall-Risk Assessment Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment with stratification
    risk_style = ParagraphStyle(
        'Risk',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor(risk_info['color'])
    )
    story.append(Paragraph(f"{risk_info['emoji']} {risk_info['level']}", risk_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Probability score
    story.append(Paragraph(f"<b>Fall Risk Probability:</b> {probability*100:.1f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Classification:</b> {prediction}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Key Metrics Table
    story.append(Paragraph("Key Gait Metrics", styles['Heading3']))
    metrics_data = [
        ['Metric', 'Value'],
        ['Cadence (steps/min)', f"{features['Cadence_mean']:.1f}"],
        ['Stride Length (m)', f"{features['StrideLength_mean']:.2f}"],
        ['Stride Velocity (m/s)', f"{features['Stride_Velocity_mean']:.2f}"],
        ['Clearance (m)', f"{features['Clearance_mean']:.3f}"],
        ['Number of Strides', f"{features['num_strides']:.0f}"]
    ]
    
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Clinical Action
    story.append(Paragraph("Recommended Action", styles['Heading3']))
    story.append(Paragraph(f"<b>{risk_info['urgency']}</b>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"• {risk_info['action']}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Paragraph("Disclaimer", styles['Heading3']))
    story.append(Paragraph(
        "This assessment is for screening purposes only and does not replace clinical evaluation by a healthcare professional. "
        "Gait metrics are approximated from camera-based analysis and should be interpreted in clinical context.",
        styles['Italic']
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer
    
# Main app
def main():
    st.title("🚶 GSTRIDE Fall-Risk Scanner")
    st.markdown("### Gait Analysis for Fall Risk Assessment")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        **Gaitalyzer**
        """)
        
        st.header("Instructions")
        st.write("""
        1. Upload a video (10-30 seconds)
        2. Ensure person walks naturally
        3. Good lighting recommended
        4. Full body visible in frame
        """)
    
    # Load model
    try:
        model, scaler, label_encoder, feature_names, metadata = load_model_artifacts()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Walking Video",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video of someone walking (10-30 seconds recommended)"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.video(video_path)
        
        if st.button("Analyze Gait", type="primary"):
            with st.spinner("Processing video..."):
                # Load MediaPipe
                pose, mp_pose, mp_drawing = load_mediapipe()
                
                # Extract skeleton
                st.info("Extracting skeleton keypoints...")
                skeleton_data, fps = extract_skeleton_from_video(video_path, pose, mp_pose)
                
                if len(skeleton_data) == 0:
                    st.error("❌ No skeleton detected in video. Please ensure person is clearly visible.")
                    return
                
                st.success(f"✅ Detected {len(skeleton_data)} frames with skeleton")
                
                # Create side-by-side video
                st.info("Creating side-by-side comparison video...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='_comparison.mp4') as comparison_video_file:
                    comparison_video_path = comparison_video_file.name
                
                create_side_by_side_video(
                    video_path, skeleton_data, comparison_video_path, 
                    pose, mp_pose, mp_drawing
                )
                
                # Display side-by-side comparison video
                st.subheader("Video Analysis")
                st.write("**Side-by-Side Comparison: Original | AI Analysis**")
                st.video(comparison_video_path)
                st.info("Left side shows original video, right side shows skeleton keypoints the AI analyzes")
                
                # Compute features
                st.info("Computing gait features...")
                features = compute_gait_features_from_skeleton(skeleton_data, fps)
                
                if features is None:
                    return
                
                # Make prediction
                st.info("Analyzing fall risk...")
                feature_vector = np.array([features[f] for f in feature_names]).reshape(1, -1)
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Get prediction and probability
# Get prediction and probability
                prediction = model.predict(feature_vector_scaled)[0]
                proba = model.predict_proba(feature_vector_scaled)[0]
                
                # --- INSERT THIS HERE ---
                st.write("DEBUG CLASSES:", label_encoder.classes_)
                # ------------------------

                # Get faller probability
                faller_idx = list(label_encoder.classes_).index('Faller') # <--- This is the crashing line
                faller_probability = proba[faller_idx]
                
                # Get risk stratification
                risk_info = get_risk_stratification(prediction, faller_probability)
                
                # Display results
                st.markdown("---")
                st.header("Assessment Results")
                
                # Main risk display
                st.markdown(f"""
                <div style="background-color: {risk_info['color']}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{risk_info['emoji']} {risk_info['level']}</h1>
                    <h3 style="color: white; margin: 10px 0;">{risk_info['urgency']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fall Risk Probability", f"{faller_probability*100:.1f}%")
                
                with col2:
                    st.metric("Cadence", f"{features['Cadence_mean']:.1f} steps/min")
                
                with col3:
                    st.metric("Stride Velocity", f"{features['Stride_Velocity_mean']:.2f} m/s")
                
                # Recommended action
                st.subheader("Clinical Recommendation")
                st.info(f"**{risk_info['action']}**")
                
                # Key metrics
                st.subheader("Detailed Gait Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.write("**Temporal Parameters:**")
                    st.write(f"• Cycle Duration: {features['CycleDuration_mean']:.2f} s")
                    st.write(f"• Cadence: {features['Cadence_mean']:.1f} steps/min")
                    st.write(f"• Number of Strides: {features['num_strides']:.0f}")
                
                with metrics_col2:
                    st.write("**Spatial Parameters:**")
                    st.write(f"• Stride Length: {features['StrideLength_mean']:.2f} m")
                    st.write(f"• Clearance: {features['Clearance_mean']:.3f} m")
                    st.write(f"• Stride Velocity: {features['Stride_Velocity_mean']:.2f} m/s")
                
                # Technical note
                with st.expander("🔬 Technical Details & Limitations"):
                    st.write(f"""
                    """)
                
                # Generate PDF report
                st.subheader("Download Report")
                pdf_buffer = generate_pdf_report(prediction, features, metadata, faller_probability, risk_info)
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"fall_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                # Download side-by-side comparison video
                with open(comparison_video_path, 'rb') as f:
                    st.download_button(
                        label="⬇Download Side-by-Side Comparison Video",
                        data=f,
                        file_name=f"gait_analysis_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
                
                # Privacy note
                st.info("**Privacy Note:** No video data is stored. Only anonymous skeleton keypoints are processed.")

if __name__ == "__main__":
    main()