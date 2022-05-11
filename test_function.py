import face_recognition
import numpy as np
import os
import glob
import cv2
# from model.class_face_recognition import Identification
from model.load_database import Database
from model.class_face_recognition import Identification

database = Database("/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset")

database.load_database()

identification = Identification('/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset/client/Guerre-en-Ukraine-Jean-Luc-Melenchon-juge-Poutine-coupable.jpg','/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_encodings.npy','/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_names.npy')

identification.compare_image_database()
identification.render()


