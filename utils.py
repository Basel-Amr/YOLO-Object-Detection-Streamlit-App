import numpy as np
import cv2
from PIL import Image
import os

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image)
    

def save_image_with_predictions(image, results, class_names, output_dir='downloads'):
    """
    Draws bounding boxes with labels and saves the image in the specified directory.

    Args:
        image (np.array): Original image (RGB).
        results: YOLO model result (ultralytics format).
        class_names (list): List of class names.
        output_dir (str): Directory to save the image.

    Returns:
        image_with_boxes (np.array), output_path (str)
    """
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_id = int(cls)
        label = f"{class_names[class_id]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(255, 0, 0), thickness=2)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, 'predicted_image.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return image, output_path

