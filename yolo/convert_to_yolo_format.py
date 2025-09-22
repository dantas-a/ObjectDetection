import os
from PIL import Image

type = 'train'  # ou 'validation'

labels_not_yolo_path = "../software/OIDv4_ToolKit/OID/Dataset/"+type+"/Vehicle registration plate/Labels_Not_Yolo"
labels_path = "../software/OIDv4_ToolKit/OID/Dataset/"+type+"/Vehicle registration plate/Labels"
images_path = "../software/OIDv4_ToolKit/OID/Dataset/"+type+"/Vehicle registration plate/Images"

# mapping classe -> id (ici une seule classe)
class_map = {"Vehicle registration plate": 0}

def convert_to_yolo(img_width, img_height, x_min, y_min, x_max, y_max):
    # conversion en centre + taille
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

for label_file in os.listdir(labels_not_yolo_path):
    if not label_file.endswith(".txt"):
        continue
    
    label_not_yolo_path = os.path.join(labels_not_yolo_path, label_file)
    label_path = os.path.join(labels_path, label_file)
    image_file = label_file.replace(".txt", ".jpg")  # change extension si besoin (.png)
    image_path = os.path.join(images_path, image_file) 
    
    if not os.path.exists(image_path):
        print(f"[WARN] Pas d'image correspondante pour {label_file}")
        continue  

    # récup dimensions image
    with Image.open(image_path) as img:
        img_w, img_h = img.size
        
    yolo_lines = []
    with open(label_not_yolo_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            x_min, y_min, x_max, y_max = map(float, parts[-4:])
            class_name = " ".join(parts[:-4])

            class_id = class_map[class_name]
            x_c, y_c, w, h = convert_to_yolo(img_w, img_h, x_min, y_min, x_max, y_max)
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # réécriture du fichier label au format YOLO
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))