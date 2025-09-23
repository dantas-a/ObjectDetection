import os
import cv2
from ultralytics import YOLO

# Paths
MODEL_PATH = 'saved_model/license_plate_model.pt'
TEST_IMAGES_DIR = '../software/OIDv4_ToolKit/OID/Dataset/test/Vehicle registration plate/images'
OUTPUT_DIR = 'test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Get test images
image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    results = model(img_path)
    result_img = results[0].plot()  # Draw boxes on image

    # Save result
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, result_img)

print(f"Results saved to {OUTPUT_DIR}")