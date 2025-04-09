import streamlit as st
import torch
import numpy as np
import os
import io
import tempfile
import requests
from streamlit_lottie import st_lottie
import cv2
from model import load_model
from utils import load_image, save_image_with_predictions
from tqdm import tqdm

# ------------------------ Constants ------------------------
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ------------------------ Streamlit App ------------------------
st.title("🦾 YOLO Object Detection App")
st.markdown("Upload images or videos and detect objects using a YOLO model.")

# Load YOLO model
model = load_model('models\\yolov3u.pt')  # Change path as needed
if model is None:
    st.error("❌ Failed to load model.")
    st.stop()

uploaded_files = st.file_uploader("📤 Upload image(s) or video(s)", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'], accept_multiple_files=True)

# Directory for temporary results
temp_dir = tempfile.mkdtemp()

if uploaded_files and model:
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        st.markdown(f"---\n### 📷 {uploaded_file.name}")

        if file_extension in ['jpg', 'jpeg', 'png']:
            # Image processing
            image = load_image(uploaded_file)
            st.image(image, caption="Original Image", width=400)

            if st.button(f"🚀 Run YOLO on {uploaded_file.name}"):
                with st.spinner("Detecting objects..."):
                    progress_bar = st.progress(0)  # Initialize progress bar
                    results = model(image)[0]
                    output_image, output_path = save_image_with_predictions(
                        image.copy(), results, COCO_CLASS_NAMES, output_dir=temp_dir
                    )

                    # Update progress bar to 100%
                    for i in tqdm(range(100), desc="Processing", position=0, leave=True):
                        progress_bar.progress(i + 1)

                    # Show image with predictions
                    st.image(output_image, caption="🔍 Prediction Result", width=600)

                    # Download button for image
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Result (Right-click to save to a specific directory)",
                            data=file,
                            file_name=f"predicted_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
        
        elif file_extension in ['mp4', 'avi']:
            # Video processing
            video_bytes = uploaded_file.read()  # Read video bytes directly
            st.video(video_bytes)  # Display video directly from the byte stream

            if st.button(f"🚀 Run YOLO on {uploaded_file.name}"):
                with st.spinner("Detecting objects in video..."):
                    # Temporary file path for saving the processed video
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(video_bytes)

                    # Open the video and process frame-by-frame
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("❌ Error: Could not open the video file.")
                        st.stop()

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
                    progress_bar = st.progress(0)  # Initialize progress bar

                    output_video_path = os.path.join(temp_dir, f"predicted_{uploaded_file.name}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
                    width, height = int(cap.get(3)), int(cap.get(4))  # Get the frame dimensions
                    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

                    if not out.isOpened():
                        st.error("❌ Error: Could not open the output video file.")
                        st.stop()

                    # Loop through video frames
                    for frame_idx in tqdm(range(total_frames), desc="Processing video", position=0, leave=True):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Perform object detection on the frame
                        results = model(frame)[0]
                        frame_with_predictions, _ = save_image_with_predictions(frame, results, COCO_CLASS_NAMES)

                        # Write the frame with predictions to the output video
                        out.write(frame_with_predictions)

                        # Update progress bar based on frames processed
                        progress_bar.progress(int((frame_idx / total_frames) * 100))

                    # Release resources
                    cap.release()
                    out.release()

                    # Provide download link for the processed video
                    st.video(output_video_path)  # Display the processed video
                    
                    # Download button for processed video
                    with open(output_video_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=file,
                            file_name=f"predicted_{uploaded_file.name}",
                            mime="video/mp4"
                        )

else:
    st.info("📌 Please upload at least one image or video to begin.")


# --------------------------
# 📌 Sidebar - History & Download
# --------------------------
with st.sidebar:
    st.image("assets\sprints_logo.png", width=150)
    st.markdown("## 🔗 Connect with Me")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/baselamrbarakat/)")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/Basel-Amr)")
    st.markdown("[![Email](https://img.shields.io/badge/Email-red?style=flat&logo=gmail)](mailto:baselamr52@gmail.com)")
st.markdown("---")
st.markdown("Developed by Basel Amr Barakat")
