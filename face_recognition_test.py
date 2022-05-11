'''Face recognition Projet : '''

# Import Library 
import face_recognition
import numpy as np
import os
import glob
from function_face_recognition import *
import cv2



font = cv2.FONT_HERSHEY_DUPLEX
color=(255, 255, 255)

# Load database image from load_database.py
know_faces = np.load('/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_encodings.npy')
know_face_names = np.load('/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_names.npy')

# Read file to compare 
load_img = face_recognition.load_image_file("/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_detection/dataset/clients/phillipe_poutou.jpg")
face_location = face_recognition.face_locations(load_img)
img_encoding = face_recognition.face_encodings(load_img,face_location)
image = cv2.cvtColor(load_img,cv2.COLOR_RGB2BGR)

face_name = []
face_distances = []
for face_encoding in know_faces:
    matches_faces = face_recognition.compare_faces(know_faces,img_encoding,tolerance=0.55)
    face_distances = face_recognition.face_distance(know_faces, img_encoding)
    best_match_index = np.argmin(face_distances)
    if matches_faces[best_match_index]:
        name = know_face_names[best_match_index]
    else:
        name = 'Nouveau_client'
    face_name.append(name)

for (top, right, bottom, left), name in zip(face_location, face_name):
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.putText(image, name, (left, bottom + 40), font, 1.0, (255, 255, 255), 2)
        # Display the resulting image
        cv2.imshow('face detection', image)

        # Until we press a key keep the window opened
        cv2.waitKey(0)

        # Close the window and return to terminal
        
cv2.destroyAllWindows()


    

