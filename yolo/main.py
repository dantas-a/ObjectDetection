from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv

# Load models
vehicle_model = YOLO('yolov8n.pt')  
license_plate_model = YOLO('saved_model/license_plate_model.pt') 

results = {}
mot_tracker = Sort()

# Load a video
cap = cv2.VideoCapture('./videos/car.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ou "XVID"
out = cv2.VideoWriter("result.mp4", fourcc, fps, (width, height))

# vehicle class IDs 
vehicles = [2,3,5,7] 

frame_nmr = 0
ret = True
# go through each frame of the video
while ret:
    frame_nmr += 1
    # read a frame
    ret, frame = cap.read()
    # if the frame was properly read (not end of video)
    if ret : 
        # initialize results for the current frame
        results[frame_nmr] = {}
        
        # we make a copy of the original frame for the video output    
        frame_res = frame.copy()
        
        # detect elements in the frame (ex: vehicles, persons, etc.)
        detections = vehicle_model(frame)[0]
        # we initialize an empty list to store only vehicle detections
        detections_ = []
        # we go through each detection
        for detection in detections.boxes.data.tolist():
            # box coordinates, confidence score, class id
            x1, y1, x2, y2, score, class_id = detection
            # we keep only vehicles
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2, score])
                # we draw a rectangle around the vehicle on the output frame
                cv2.rectangle(frame_res, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
           
        # if we detected vehicles     
        if detections_ != []:
            # track vehicles on the frame
            track_ids = mot_tracker.update(np.array(detections_))
        
            # detect license plates in the frame
            license_plates = license_plate_model(frame)[0]
            # we go through each detected license plate
            for license_plate in license_plates.boxes.data.tolist():
                # box coordinates, confidence score, class id of the license plate
                x1, y1, x2, y2, score, class_id = license_plate
                
                # we draw a rectangle around the license plate on the output frame
                cv2.rectangle(frame_res, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # find the car associated with the license plate in the frame
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate,track_ids)
                
                # if we found the car
                if car_id != -1:
                    # get license plate image
                    license_plate_image = frame[int(y1):int(y2), int(x1):int(x2),:]
                    
                    # process license plate image for the OCR
                    # convert image to grey scale
                    license_plate_image_grey = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
                    
                    # blur the grayscale license plate to reduce noise
                    license_plate_image_grey_blur = cv2.GaussianBlur(license_plate_image_grey,(3,3),0)
                    # binarize using Otsu's method
                    _, license_plate_image_thresh = cv2.threshold(license_plate_image_grey_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
                    
                    # we display the processed license plate image in the video output
                    # Resize the processed license plate image for display
                    display_height = int(y2 - y1)
                    display_width = int((x2 - x1) * 2)  # Make it wider for better visibility
                    license_plate_display = cv2.resize(license_plate_image_thresh, (display_width, display_height))

                    # Determine position to overlay (right of the license plate box, but within frame)
                    overlay_x = min(int(x2) + 10, frame_res.shape[1] - display_width)
                    overlay_y = max(int(y1), 0)
                    # Ensure overlay does not go out of frame vertically
                    if overlay_y + display_height > frame_res.shape[0]:
                        overlay_y = frame_res.shape[0] - display_height

                    # Overlay the processed image onto the frame
                    license_plate_display = cv2.cvtColor(license_plate_display, cv2.COLOR_GRAY2BGR)
                    frame_res[overlay_y:overlay_y+display_height, overlay_x:overlay_x+display_width] = license_plate_display
                    
                    # recover the text on the license plate with an OCR
                    license_plate_text, confidence_text_extraction = read_license_plate(license_plate_image_thresh)
                    
                    # we write the license plate text on the output video frame
                    cv2.putText(frame_res, license_plate_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                    # we save the results for the current frame and the current car
                    results[frame_nmr][car_id] = {
                        'car': {'bbox' : [xcar1, ycar1, xcar2, ycar2]},
                        
                        'license_plate' : {'bbox': [x1, y1, x2, y2],
                                        'bbox_score': score,
                                        'text': license_plate_text,
                                        'text_score': confidence_text_extraction}
                    }
        # save the frame in the video        
        out.write(frame_res)                    

# write results
write_csv(results, './results.csv')     