import face_recognition
import numpy as np
import os
import glob
import cv2


# def load_database(dir_clients):
#     face_encodings = []
#     face_names = []
#     id = 1
#     for dir_identite in os.listdir(dir_clients):
#             fichiers = []
#             for ext in ["*.jpg", "*.jpeg", "*.png"]:
#                 fichiers.extend(glob.glob(dir_clients+"/"+dir_identite+"/"+ext))
#             if len(fichiers) ==0:
#                 print("Repertoire vide", dir_identite)
#                 continue
#             for fichier in fichiers:
#                 print("ID", id)
#                 id += 1
#                 image = face_recognition.load_image_file(fichier)
#                 embedding = face_recognition.face_encodings(image)[0]
#                 face_encodings.append(embedding)
#                 face_names.append(dir_identite)


#     np.save("face_encodings", np.array(face_encodings))
#     np.save("face_names", np.array(face_names))

def load_numpy_files(path_face_encoding,path_face_name):
    # Load database image from load_database.py
    know_faces = np.load(path_face_encoding)
    know_face_names = np.load(path_face_name)
    return know_faces

def load_image_compare(path_image):
    load_img = face_recognition.load_image_file(path_image)
    face_location = face_recognition.face_locations(load_img)
    img_encoding = face_recognition.face_encodings(load_img,face_location)
    image = cv2.cvtColor(load_img,cv2.COLOR_RGB2BGR)  
    face_name = []
    face_distances = []  
    return image

def compare_image_database():
    for face_encoding in know_faces:
        matches_faces = face_recognition.compare_faces(know_faces,img_encoding,tolerance=0.55)
        face_distances = face_recognition.face_distance(know_faces, img_encoding)
        best_match_index = np.argmin(face_distances)
        if matches_faces[best_match_index]:
            name = know_face_names[best_match_index]
        else:
            name = 'Nouveau_client'
        face_name.append(name)

def render():
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

