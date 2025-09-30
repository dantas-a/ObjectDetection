import os
from ultralytics import YOLO

# Paths to datasets
dataset_path = '../software/OIDv4_ToolKit/OID/Dataset'
train_images = os.path.join(dataset_path, 'train/Vehicle registration plate/images')
train_labels = os.path.join(dataset_path, 'train/Vehicle registration plate/labels')
val_images = os.path.join(dataset_path, 'validation/Vehicle registration plate/images')
val_labels = os.path.join(dataset_path, 'validation/Vehicle registration plate/labels')

# Create a YAML config for YOLO dataset
data_yaml = 'license_plate_data.yaml'
with open(data_yaml, 'w') as f:
    f.write(f"""
train: {train_images}
val: {val_images}
nc: 1
names: ['vehicle_registration_plate']
""")

# Train YOLO model
model = YOLO('yolov8n.pt')  # Use a small pre-trained model for transfer learning
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4
)
model.save('./saved_model/license_plate_model.pt')