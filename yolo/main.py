from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car

# Load models
vehicle_model = YOLO('yolov8n.pt')  
license_plate_model = YOLO('saved_model/license_plate_model.pt') 

mot_tracker = Sort()

# Load a video
cap = cv2.VideoCapture('./videos/car.mp4')

vehicles = [2,3,5,7]  # vehicle class IDs in COCO dataset

# read frames
frame_nmr = 0
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr < 10:
        # detect cars
        detections = vehicle_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2, score])
                
        # track vehicles
        track_ids = mot_tracker.update(np.array(detections_))
        
        # detect license plates
        license_plates = license_plate_model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # assign license plates to vehicles
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate,track_ids)
            
            # get license plate image
            license_plate_image = frame[int(y1):int(y2), int(x1):int(x2),:]
            
            # process license plate image
            # convert image to grey scale
            license_plate_image_grey = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
            # Everything under 120 -> 0, everything over 120 -> 255
            _, license_plate_image_thresh = cv2.threshold(license_plate_image_grey,120,255,cv2.THRESH_BINARY_INV)
            
            cv2.imshow("license_plate_image", license_plate_image)
            cv2.imshow("license_plate_image_grey", license_plate_image_grey)
            cv2.imshow("license_plate_image_thresh", license_plate_image_thresh)
            cv2.waitKey(0)