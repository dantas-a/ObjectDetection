import os
from PIL import Image

# Choose which dataset split you want to work with: 'train', 'validation', or 'test'
type = 'test'

# Define paths to the dataset folders based on the chosen split
# These paths point to a specific object class: "Vehicle registration plate"
labels_not_yolo_path = "../software/OIDv4_ToolKit/OID/Dataset/"+type+"/Vehicle registration plate/Labels_Not_Yolo"
# Path where YOLO-formatted labels will be (or already are) stored
labels_path = "../software/OIDv4_ToolKit/OID/Dataset/"+type+"/Vehicle registration plate/labels"
# Path where the actual image files are located
images_path = "../software/OIDv4_ToolKit/OID/Dataset/"+type+"/Vehicle registration plate/images"

# Mapping of object classes to numeric IDs (required by YOLO format)
# In YOLO, each class is represented by an integer starting from 0
class_map = {"Vehicle registration plate": 0}


def convert_to_yolo(img_width, img_height, x_min, y_min, x_max, y_max):
    """
    Convert bounding box coordinates from absolute pixel values (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)
    All outputs are normalized by image dimensions (range 0-1)
    
    Parameters:
        img_width (int): width of the image in pixels
        img_height (int): height of the image in pixels
        x_min, y_min (int/float): top-left coordinates of the bounding box
        x_max, y_max (int/float): bottom-right coordinates of the bounding box

    Returns:
        tuple: (x_center, y_center, width, height) in YOLO format
    """
    # Compute the center of the bounding box
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    
    # Compute width and height of the bounding box
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Return normalized YOLO coordinates
    return x_center, y_center, width, height

# Iterate over every label file in the folder containing non-YOLO format labels
for label_file in os.listdir(labels_not_yolo_path):
    if not label_file.endswith(".txt"):
        continue
    
    # path to the non-yolo format label file
    label_not_yolo_path = os.path.join(labels_not_yolo_path, label_file)
    # path to the yolo format label file
    label_path = os.path.join(labels_path, label_file)
    # corresponding picture to the label
    image_file = label_file.replace(".txt", ".jpg")  
    # path to the picture associated to the label file
    image_path = os.path.join(images_path, image_file) 
    
    if not os.path.exists(image_path):
        print(f"[WARN] No picture found for this label file : {label_file}")
        continue  

    # Open the image to get its width and height
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    
    yolo_lines = []
    # Read the non-YOLO label file
    with open(label_not_yolo_path, "r") as f:
        # Go through each line
        for line in f.readlines():
            parts = line.strip().split()
            # Last 4 elements are bounding box coordinates: xmin, ymin, xmax, ymax
            x_min, y_min, x_max, y_max = map(float, parts[-4:])
            # Everything before the last 4 elements is the class name
            class_name = " ".join(parts[:-4])

            # Map the class name to its YOLO class ID
            class_id = class_map[class_name]
            # Convert bounding box to YOLO format
            x_c, y_c, w, h = convert_to_yolo(img_w, img_h, x_min, y_min, x_max, y_max)
             # Add the YOLO-formatted line to the output list
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Write YOLO-formatted labels to the output file
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))