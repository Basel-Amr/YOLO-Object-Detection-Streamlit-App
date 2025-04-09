# YOLO Object Detection with Streamlit

![Logo](assets/sprints_logo.png)

---

## 📌 Table of Contents
- [📜 About the Project](#about-the-project)
- [📂 Folder Structure](#folder-structure)
- [⚡ Features](#features)
- [💻 Technologies](#Technologies)
- [🚀 Running the Project](#running-the-project)
- [📸 Demo & Screenshots](#demo--screenshots)
- [📜 License](#license)

---

## 📜 About the Project

Welcome to the YOLO Object Detection with Streamlit project! This application harnesses the power of YOLO (You Only Look Once), a state-of-the-art object detection algorithm, to identify and classify objects within images and videos. Using Streamlit, a framework for creating interactive web applications, this project allows users to upload images or videos, which will then be processed in real-time. The model predicts the objects present, and the results are shown instantly, providing a seamless user experience.

This project also includes a notebook for running experiments on the COCO dataset to further understand object detection tasks.

---

## 📂 Folder Structure

```
.
├── app.py                    # Main Streamlit app for running object detection
├── model.py                  # Code for loading the YOLO model
├── utils.py                  # Helper functions for image/video handling
├── run.bat                   # Batch file to run the application on Windows
├── inputs/                   # Folder containing sample input images and videos
│   ├── input_video.mp4       # Example input video
│   └── LosAngeles2022-101.jpg # Example input image
├── outputs/                  # Folder where the processed outputs are stored
│   ├── predicted_image.jpg   # Processed image with object detections
│   └── processed_video.mp4   # Processed video with object detections
└── notebooks/                # Jupyter notebook for experimenting with the COCO dataset
    └── computer-vision-coco-dataset.ipynb
```

---

## ⚡ Features

✅ Real-time Object Detection: Upload an image or video, and the YOLO model will identify objects in the media in real-time.
✅ Interactive Streamlit Interface: A user-friendly interface where you can upload files and view results instantly.
✅ Image and Video Processing: Supports both image and video inputs for object detection, with predictions displayed directly in the app.
✅ Downloadable Results: After processing, the user can download the predicted images or videos with the detected objects.
✅ COCO Dataset Integration: The project uses the COCO dataset's class names to classify objects in the uploaded media
---

## 💻 Technologies
Python 3.x: The programming language used for developing the project.

YOLO (You Only Look Once): A popular object detection model.

Streamlit: A Python library to create interactive web applications.

OpenCV: For handling video processing and image manipulation.

PyTorch: The framework for running the YOLO model.

TQDM: A library to create progress bars for long-running processes.

FFMPEG: For video processing and converting video formats.
---

## 🚀 Running the Project

1️⃣ **Clone the repository:**

```bash
$ git clone https://github.com/Basel-Amr/Sprints-AI-and-ML-Bootcamp.git
$ cd 09_GenerativeAI/56_Introduction to Large Language Modelling
```

2️⃣ **Create a virtual environment (Optional but recommended):**

```bash
pip install -r requirements.txt
```

3️⃣ **Run the Application**

Option 1: Run via Command Line
```bash
$ pip install -r requirements.txt
```

Option 2: Run via run.bat (Windows users)
If you're on a Windows machine, you can run the application using the run.bat file. Just double-click the file to start the application.


---

## 🚀 Running the Project

To launch the Streamlit application, simply run:

```bash
$ streamlit run app.py
or run run.bat
```

This will start a local development server. Open the provided URL in your browser to interact with the **object detection**.

---

## 📸 Demo & Screenshots

🎥 **Demo Video:** 
[Video before preprocessing](inputs\input_video.mp4)
[Video After preprocessing](output\predicted_input_video.mp4)
[Demo Video](output\demo_video.mp4)

🖼️ **Images:**
- ![Input Image](inputs\LosAngeles2022-101.jpg)
- ![Output Image](output\predicted_image.jpg)

---
## 🎯 Developed By

**Basel Amr Barakat**  
📧 [baselamr52@gmail.com](mailto:baselamr52@gmail.com)  
💼 [LinkedIn](https://www.linkedin.com/in/baselamrbarakat/)
