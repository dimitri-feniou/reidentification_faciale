import face_recognition
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw,ImageFilter
import pandas as pd
from datetime import datetime
import uuid


class Identification:
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (255, 255, 255)

    def __init__(self, path_image, path_face_encoding, path_face_name, tolerance=0.55):
        self.tolerance = tolerance
        self.path_image = path_image

        if not os.path.exists(path_face_encoding):
            print("Fichier", path_face_encoding, "non trouvé")
            quit()
        if not os.path.exists(path_face_name):
            print("Fichier", path_face_name, "non trouvé")
            quit()
        self.known_face_encodings = np.load(path_face_encoding)
        self.known_face_names = np.load(path_face_name)
        print(self.known_face_names)

    def compare_image_database(self):
        df = pd.read_csv(
            "/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset.csv")
        load_img = Image.open(self.path_image).convert("RGB")
        self.face_location = face_recognition.face_locations(
            np.asarray(load_img), model='hog')
        img_encodings = face_recognition.face_encodings(
            np.asarray(load_img), self.face_location, model='hog')
        self.face_name = []
        self.face_distances = []

        for img_encoding in img_encodings:
            matches_faces = face_recognition.compare_faces(
                self.known_face_encodings, img_encoding, self.tolerance)
            self.face_distances = face_recognition.face_distance(
                self.known_face_encodings, img_encoding)
            best_match_index = np.argmin(self.face_distances)
            # print(matches_faces)
            # print(face_distances)
            print(best_match_index)
            if matches_faces[best_match_index]:
                self.name = self.known_face_names[best_match_index]
                self.face_name.append(self.name)
                date = datetime.now()
                self.seen = str(matches_faces.count(True) + 1)
            # Append row to dataframe with name seen datetime today
                append_row = [date, self.seen, self.name]
                df = df.append(
                    pd.Series(append_row, index=df.columns[:len(append_row)]), ignore_index=True)
            else:
                self.name = 'nouveau_client'
                seen = 1
                self.face_name.append(self.name)
                self.seen = str(seen)
                now = datetime.now()
                date = now.strftime("%d/%m/%Y %H:%M:%S")
                append_row = [date, self.seen, self.name]
                # df2=({'date': pd.Timestamp.today,'seen':seen,'type_client':name})
                df = df.append(
                    pd.Series(append_row, index=df.columns[:len(append_row)]), ignore_index=True)

        df.to_csv(
            "/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset.csv", index=False)

    def render(self):
        pil_image = Image.open(self.path_image).convert("RGB")
        print(self.seen)
        print(self.name)
        draw = ImageDraw.Draw(pil_image)
        mask = Image.new('L',pil_image.size)
        draw_mask = ImageDraw.Draw(mask)
        for (top, right, bottom, left), self.name, self.seen in zip(self.face_location, self.face_name, self.seen):
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            

        # Draw a label with a name below the face
            text_width, text_height = draw.textsize(self.name)
            draw.rectangle(((left, bottom - text_height - 10),
                            (right, bottom)), outline='blue', width=0)
            draw_mask.rectangle(((left, top), (right, bottom)),fill=255)
            draw.text((left + 2, bottom - text_height + 10),
                      f"{self.name}\n seen :{self.seen}", fill='white', align='center')
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=20))
        res = Image.composite(blurred, pil_image, mask)
        filename = uuid.uuid4().hex
        save_image = res.save(
            "/home/dimitri/Documents/code/python/projet_E1_face_recognition/static/temp.jpg")
        save_image_database = res.save(
            f"/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset/client/{filename}.jpg")

        return save_image, save_image_database

# toto = Identification('/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_detection/dataset/client', '/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_detection/dataset/client/phillipe_poutou.jpg',
#                       '/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_encodings.npy', '/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_names.npy')


# toto.compare_image_database()
# print("#############")
# for name in zip(toto.face_name):
#     print("   ", name)
# toto.render()
# cv2.imshow("Frame render", toto.image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
