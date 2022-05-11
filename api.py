import face_recognition
from flask import Flask, flash, request, redirect, url_for, render_template , Response
from function_face_recognition import *
from model.class_face_recognition import Identification
from model.load_database import Database
from werkzeug.utils import secure_filename
import cv2

# import camera
app = Flask(__name__)
path_folder = '/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_detection/dataset/client/'

camera = cv2.VideoCapture(0)
face_encodings = []
face_names = []
id = 1
# for dir_client in os.listdir('/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset'):
#     fichiers = []
#     for ext in ["*.jpg", "*.jpeg", "*.png"]:
#         fichiers.extend(glob.glob('/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset'+"/"+dir_client+"/"+ext))
#     if len(fichiers) == 0:
#         print("Repertoire vide", dir_client)
#     for fichier in fichiers:
#                     id += 1
#                     image = face_recognition.load_image_file(fichier)
#                     embedding = face_recognition.face_encodings(image)[0]
#                     face_encodings.append(embedding)
#                     face_names.append(dir_client)
                
#     print(f"La base de données client contient: {len(fichiers)} photos")
                    

#     np.save("face_encodings", np.array(face_encodings))
#     np.save("face_names", np.array(face_names))


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
known_face_encodings = np.load("/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_encodings.npy")
known_face_names = np.load("/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_names.npy")

def gen_frames():
    global face_locations , face_names ,camera,seen
    camera = cv2.VideoCapture(0)
    while True:
        
        success, frame = camera.read()  # read the camera frame
        
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        seen = []
        face_names = []
        for face_encoding in face_encodings:
            matches_faces = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(matches_faces)
            print(face_distances)
            seen = str(matches_faces.count(True))
            print(type(seen))
            print(seen)
            if matches_faces[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = 'nouveau_client'
                seen = 1
            face_names.append(name)
            seen = str(seen)
        
    

        # Display the results
        for (top, right, bottom, left), name, seen in zip(face_locations, face_names,seen):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
        # Extract the region of the image that contains the face
            face_image = frame[top:bottom, left:right]
         # Blur the face image
            face_image = cv2.GaussianBlur(face_image, (99, 99), 40)
        # Draw a box around the face
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                        (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name,'seen:',seen}", (left + 6, bottom + 20),
                        font, 0.5, (255, 255, 255), 1)
            # Put the blurred face region back into the frame image
            frame[top:bottom, left:right] = face_image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



 
UPLOAD_FOLDER = 'static/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
         return redirect(url_for('upload_image'),code=307)
    else :
        return render_template('index.html')

@app.route('/database')

def load_database_render():
    if request.method == 'GET':
        database = Database("/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset")
        database.load_database()
        return render_template('load_database.html')

@app.route('/upload_image', methods=['POST','GET'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            
            identification = Identification(file,'/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_encodings.npy','/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_names.npy')
            identification.compare_image_database()
            identification.render()
            filename = secure_filename('temp.jpg')
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload_image filename: ' + filename)
            flash("Résultat de l'analyse")
            return render_template('upload_image.html',filename=filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    elif request.method == 'GET':
        return render_template("upload_image.html")
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='upload/' + filename), code=301)

@app.route('/identification', methods=['GET', 'POST'])
def identification():
    if request.method == 'GET':
        identification = Identification('/home/dimitri/Documents/code/python/projet_E1_face_recognition/dataset/client/phillipe_poutou.jpg','/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_encodings.npy','/home/dimitri/Documents/code/python/projet_E1_face_recognition/face_names.npy')
        identification.compare_image_database()
        identification.render()
        cv2.imshow("Frame render", identification.image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return render_template('identification.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    return render_template('camera.html')
if __name__ == '__main__':
    app.run(debug=True)
