import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import hashlib
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

# ==========================================================
# Configuration de l'application
# ==========================================================

# Types de fichiers autorisés
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Dossiers de stockage
upload_folder = 'static/uploads'           # Dossier pour les images téléchargées
image_db_folder = 'static/image.orig'      # Dossier contenant les images de la base
features_folder = 'static/features'        # Dossier pour les fichiers de descripteurs
rp_save_dir = 'static/rp_files'            # Dossier pour enregistrer les fichiers et courbes RP

# ==========================================================
# Fonctions utilitaires pour l'upload d'images
# ==========================================================

# Vérifie si le fichier possède une extension autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Génère un nouveau nom pour l’image reçue côté serveur afin d’éviter les doublons
def new_image_name(extension='jpg'):
    now = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"img_req_{now}.{extension}"

# Calcule le hash de l’image pour vérifier son unicité dans le dossier d’upload
def hash_file(file_stream):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file_stream.read(4096), b""):
        hasher.update(chunk)
    file_stream.seek(0)
    return hasher.hexdigest()

# ==========================================================
# Routes Flask et lancement de l'application
# ==========================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file and allowed_file(file.filename):
        file_hash = hash_file(file)
        for existing_file in os.listdir(upload_folder):
            existing_path = os.path.join(upload_folder, existing_file)
            with open(existing_path, 'rb') as f:
                if hashlib.sha256(f.read()).hexdigest() == file_hash:
                    print(f"Duplicate image found: {existing_file}")
                    return jsonify({'filename': existing_file, 'file_path': existing_path})

        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = secure_filename(new_image_name(ext))
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        print(f"Uploaded file saved to {file_path}")
        return jsonify({'filename': filename, 'file_path': file_path})

    return jsonify({'error': 'Invalid file format'})

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    file_path = os.path.join(upload_folder, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'deleted': True})
    return jsonify({'deleted': False})

@app.route('/search', methods=['POST'])
def search():
    filename = request.form.get('filename')
    model_names = sorted(request.form.getlist('descriptor[]'))
    dist_metric = request.form.get('similarity')
    specified_class = None if request.form.get('image_class') == "" else int(request.form.get('image_class'))
    topn = int(request.form.get('topn'))
    file_path = os.path.join(upload_folder, filename)

    # Vérifie si l’image existe dans le dossier 'uploads'
    if not os.path.exists(file_path):
        print("[MYAPP] >> File not found error")
        return jsonify({'error': 'File not found'})

    # ------------------------ Affichage pour vérification des données reçues du formulaire ------------------------
    print(f"[MYAPP] >> File Name: {filename}")
    print(f"[MYAPP] >> Models: {model_names}")
    print(f"[MYAPP] >> Distance Metric: {dist_metric}")
    print(f"[MYAPP] >> Class specified by the user: {specified_class}")
    print(f"[MYAPP] >> Requested Top-N: {topn}")
    # --------------------------------------------------------------------------------------------------------------

    # A COMPLETER :

        # Charger les features des images de la base pour les descripteurs sélectionnés
        # Extraire les features de l’image requête à partir des descripteurs sélectionnés
        # Rechercher les images similaires avec la métrique choisie
        # Pour le calcul rappel/précision : utiliser la classe spécifiée par l’utilisateur, sinon, si l’utilisateur n’a rien spécifié, utiliser la classe prédite
        # Générer l’image de la courbe de Rappel/Précision (RP)

    # ------------------------ A REMPLACER PAR LES RESULTATS DE LA RECHERCHE ------------------------
    images_proches = ["to_be_completed.jpg"] * topn
    rp_img_path = os.path.join("static", "rp_files", "rp_img_files", "test.png")
    predicted_class = None
    # -----------------------------------------------------------------------------------------------
    # Envoi des résultats au frontend
    return jsonify({
        'filename': os.path.basename(file_path),
        'topn_similar_images': images_proches,
        'rp_curve': rp_img_path,
        'predicted_class': predicted_class,
        'specified_class': specified_class
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
