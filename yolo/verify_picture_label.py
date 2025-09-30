import os
from PIL import Image, ImageDraw

# Path towards a picture and the corresponding label
image_path = "../software/OIDv4_ToolKit/OID/Dataset/train/Vehicle registration plate/images/0a6ef05cd485f144.jpg"
label_path = "../software/OIDv4_ToolKit/OID/Dataset/train/Vehicle registration plate/labels/0a6ef05cd485f144.txt"

# open the picture and get the width and the heigth
img = Image.open(image_path)
img_w, img_h = img.size

draw = ImageDraw.Draw(img)

# Read the label file
with open(label_path, "r") as f:
    for line in f.readlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        # get the 5 elements in the yolo format file
        class_id, x_c, y_c, w, h = map(float, parts)

        # un-normalization
        x_c *= img_w
        y_c *= img_h
        w *= img_w
        h *= img_h

        # center to corners conversion
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        x_max = x_c + w / 2
        y_max = y_c + h / 2

        # draw the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), f"class {int(class_id)}", fill="red")

# print
img.show()