from pathlib import Path
import os
import numpy as np
import cv2

p = Path(__file__).resolve().parent

def detectFaces(results_folder_path, upload_folder_path, img_to_analyse_name):

    image = cv2.imread(os.path.join(upload_folder_path, img_to_analyse_name))

    # load the pre-trained model - Haar cascade
    classifier = cv2.CascadeClassifier(os.path.join(p, 'haarcascade_frontalface_default.xml'))

    # Process the image to perform face detection
    bboxes = classifier.detectMultiScale(image)

    # Display bounding box for each detected face
    for box in bboxes:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # Draw a rectangle over the image to display the face area
        cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

    resultPath = os.path.join(results_folder_path, f"result-{img_to_analyse_name}")
    cv2.imwrite(resultPath, image)

    return resultPath

    
