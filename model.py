# model.py
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

def load_model(model_path='models/yolov3u.pt', device='cpu'):
    """
    Load a YOLOv5 or YOLOv8 model using Ultralytics package.

    Args:
    - model_path: Path to the YOLO model (.pt).
    - device: 'cpu' or 'cuda'

    Returns:
    - model: Loaded model instance
    """
    try:
        model = YOLO(model_path)  # Load model
        model.to(device)  # Move to desired device
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def predict_image(model, image_np):
    """
    Perform prediction using the YOLO model.

    Args:
    - model: Loaded YOLO model.
    - image_np: NumPy image (H, W, 3) in uint8 format.

    Returns:
    - results: YOLO prediction results.
    """
    try:
        # Ensure the image is in correct format
        if isinstance(image_np, np.ndarray):
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)
        else:
            raise ValueError("Input must be a NumPy array")

        results = model(image_np)  # Let Ultralytics handle preprocessing
        return results
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None


def visualize_yolo_prediction(model, image, show_rec=True):
    """
    Visualize YOLO predictions on an image with optional bounding box rectangles and class names.

    Args:
    - model: YOLO model.
    - image: Image to perform prediction on.
    - show_rec: Boolean flag to show bounding box rectangles and class names.
    """
    results = predict_image(model, image)
    result = results[0]  # Get the first result (assuming one image)

    # Create a figure for plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # Optionally draw bounding boxes (rectangles) with 3D-like colors and class names
    if show_rec:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            conf = conf.item()  # Confidence score
            if conf > 0.5:  # Only show boxes with confidence > 0.5
                x1, y1, x2, y2 = box.cpu().numpy()  # Get box coordinates
                class_idx = int(cls.item())  # Get class index
                class_name = result.names[class_idx]  # Get class name

                # Use a color map to create a gradient effect for the bounding box
                colormap = plt.cm.viridis  # You can experiment with different colormaps
                color = colormap(conf)  # Map confidence to a color
                rgba_color = (color[0], color[1], color[2], 0.5)  # Set the alpha for transparency

                # Add a bounding box with the color gradient
                ax.add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=rgba_color, linewidth=3)
                )

                # Add class name text on the bounding box
                ax.text(
                    x1, y1 - 10, f'{class_name}: {conf:.2f}', color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5')
                )

    # Display the image with annotations
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(f'YOLO Prediction', fontsize=16)
    plt.show()
