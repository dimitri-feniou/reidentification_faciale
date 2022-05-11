import face_recognition
import numpy as np
import os
import glob
import cv2


class Database:
     def __init__(self, dir_clients):
        self.dir_clients = dir_clients

     def load_database(self):
            face_encodings = []
            face_names = []
            id = 1
            for dir_client in os.listdir(self.dir_clients):
                fichiers = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    fichiers.extend(glob.glob(self.dir_clients+"/"+dir_client+"/"+ext))
                if len(fichiers) == 0:
                    print("Repertoire vide", dir_client)
                    continue
                for fichier in fichiers:
                    id += 1
                    image = face_recognition.load_image_file(fichier)
                    embedding = face_recognition.face_encodings(image)[0]
                    face_encodings.append(embedding)
                    face_names.append(dir_client)
                
                print(f"La base de données client contient: {len(fichiers)} photos")
                    

            np.save("face_encodings", np.array(face_encodings))
            np.save("face_names", np.array(face_names))