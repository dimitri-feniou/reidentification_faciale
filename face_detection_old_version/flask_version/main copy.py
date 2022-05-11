import os
from flask import Flask, flash, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import time
from algorithm.analysis import detectFaces

UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = os.path.join('static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Démarrage du serveur flask avec des chemins "UPLOAD" et "RESULTS" dans la configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Chemin d'accueil
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # On vérifie si il y a bien un fichier dans la requête
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # Si le fichier est un fichier "vide" donc il n'a pas de nom de fichier, on redirige l'utilisateur
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # On vérifie si le fichier est un fichier autorisé
        if file and allowed_file(file.filename):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{timestr}-{secure_filename(file.filename)}"

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('analysis', name=filename))

    return render_template('home.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    result_path = ""
    if request.method == 'GET':
        result_path = detectFaces(app.config["RESULTS_FOLDER"], app.config["UPLOAD_FOLDER"], request.args.get("name"))
    return render_template('analysis.html', result_path=result_path)

app.run(debug=True)