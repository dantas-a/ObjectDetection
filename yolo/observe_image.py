import os
from PIL import Image, ImageDraw

# chemins (à adapter)
image_path = "../software/OIDv4_ToolKit/OID/Dataset/train/Vehicle registration plate/Images/0a6ef05cd485f144.jpg"
label_path = "../software/OIDv4_ToolKit/OID/Dataset/train/Vehicle registration plate/Labels/0a6ef05cd485f144.txt"

# ouvrir image
img = Image.open(image_path)
img_w, img_h = img.size

# dessinateur
draw = ImageDraw.Draw(img)

# lire fichier label
with open(label_path, "r") as f:
    for line in f.readlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_c, y_c, w, h = map(float, parts)

        # dénormalisation
        x_c *= img_w
        y_c *= img_h
        w *= img_w
        h *= img_h

        # conversion centre -> coins
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        x_max = x_c + w / 2
        y_max = y_c + h / 2

        # dessiner rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), f"class {int(class_id)}", fill="red")

# afficher
img.show()