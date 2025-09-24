import string
import easyocr

# Initialize the EasyOCR reader once to avoid re-initialization overhead
reader = easyocr.Reader(['en'], gpu=True)

# Save the results in a CSV file
def write_csv(results, output_path):
    
    with open(output_path, mode='w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr','car_id','car_bbox','license_plate_bbox','license_plate_bbox_score','license_plate_text','license_plate_text_score'))
        
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr, car_id,
                                                            '[{} {} {} {}]'.format(results[frame_nmr][car_id]['car']['bbox'][0],results[frame_nmr][car_id]['car']['bbox'][1],results[frame_nmr][car_id]['car']['bbox'][2],results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(results[frame_nmr][car_id]['license_plate']['bbox'][0],results[frame_nmr][car_id]['license_plate']['bbox'][1],results[frame_nmr][car_id]['license_plate']['bbox'][2],results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score']
                                                            ))
    f.close()

# Get a license plate, then goes through the tracked cars to find the car associated with the license plate
def get_car(license_plate, track_ids):
    
    xlp1, ylp1, xlp2, ylp2, score, class_id = license_plate
    found_car = False
    
    # we go through each tracked car
    for j in track_ids :
        # box coordinates and track id of the car
        xcar1, ycar1, xcar2, ycar2, car_id = j
        
        # if the license plate is inside the car box, we found the car
        if xcar1 < xlp1 and xlp2 < xcar2 and ycar1 < ylp1 and ylp2 < ycar2:
            car = j
            found_car = True
            break
    
    if found_car:
        return car
    
    return -1,-1,-1,-1,-1

# Read the license plate with an OCR and return the text and the confidence of the extraction
def read_license_plate(license_plate_image):
    
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"
    
    detection = reader.readtext(license_plate_image, paragraph=False, allowlist=allowlist)
    
    if detection != []:
        bbox, text, confidence = detection[0]
    else :
        text = ''
        confidence = 0.0
    
    return text, confidence