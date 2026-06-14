import warnings
warnings.filterwarnings("ignore")
import torch
torch.backends.mkldnn.enabled = False

import os
os.environ["NNPACK_SUPPRESS_WARNINGS"] = "1"
os.environ["PYTORCH_NO_NNPACK"] = "1"
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import hashlib
import matplotlib
matplotlib.use('Agg')
import numpy as np #ajoute de sanae

import mysql.connector
import time
# apport de librairie pour la sécurité 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
# Importation des fonctions utilitaires depuis app_utils.py
from app_utils import *


app = Flask(__name__)

# 1. Utilisation d'une variable d'environnement pour la cle secrete
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# 2. Utilisation d'empreintes (Hash) pour les mots de passe
USERS = {
    'cloud08': 'scrypt:32768:8:1$JoHsvejAAJ3Qayis$091c1d5deb88131139f7dc88eab710d7335667de0bd8488a668a36e3ea9df20ea44c1b7a6fd7f2f8f6f096691d58aa8a78e052296053f55b50bc0a5e30dc14ed'
}

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
query_cache_folder = 'static/query_cache'  # <--- AJOUT : Dossier pour le cache des requêtes
# ==========================================================
# Fonctions utilitaires pour l'upload d'images
# ==========================================================

def get_db_connection():
    max_retries = 5
    for i in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host=os.environ.get('MYSQL_HOST', 'localhost'),
                user=os.environ.get('MYSQL_USER', 'cloud08'),
                password=os.environ.get('MYSQL_PASSWORD', 'd1Jx5'),
                database=os.environ.get('MYSQL_DATABASE', 'image_search_engine')
            )
            return conn
        except mysql.connector.Error:
            print(f"[MYAPP] >> DB connection attempt {i+1}/{max_retries}...")
            time.sleep(5)
    return None

def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query_image_path VARCHAR(500),
                nb_images_returned INT,
                models_used VARCHAR(500),
                similarity_measure VARCHAR(100),
                precision_score FLOAT,
                recall_score FLOAT,
                rp_curve_path VARCHAR(500),
                specified_class INT NULL,
                predicted_class INT,
                search_datetime DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("[MYAPP] >> Database initialized successfully")

# Vérifie si le fichier possède une extension autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Génère un nouveau nom pour l'image reçue côté serveur afin d'éviter les doublons
def new_image_name(extension='jpg'):
    now = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"img_req_{now}.{extension}"

# Calcule le hash de l'image pour vérifier son unicité dans le dossier d'upload
def hash_file(file_stream):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file_stream.read(4096), b""):
        hasher.update(chunk)
    file_stream.seek(0)
    return hasher.hexdigest()

# ==========================================================
# Routes Flask et lancement de l'application
# ==========================================================

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

#@app.route('/login', methods=['GET', 'POST'])
#def login():
#    if request.method == 'POST':
#        username = request.form.get('username')
#        password = request.form.get('password')
#        if username in USERS and USERS[username] == password:
#            session['username'] = username
#            return redirect(url_for('index'))
#        return render_template('login.html', error='Identifiant ou mot de passe incorrect')
#    return render_template('login.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Verification securisee
        if username in USERS and check_password_hash(USERS[username], password):
            session['username'] = username
            return redirect(url_for('index'))
            
        return render_template('login.html', error='Identifiant ou mot de passe incorrect')
        
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

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
    # Securise le nom du fichier pour bloquer les "../"
    safe_filename = secure_filename(filename)
    
    # Verifie qu il n est pas vide apres nettoyage
    if not safe_filename:
        return jsonify({'deleted': False, 'error': 'Nom de fichier invalide'})
        
    file_path = os.path.join(upload_folder, safe_filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'deleted': True})
        
    return jsonify({'deleted': False})

@app.route('/search', methods=['POST'])
def search():
    try:
        filename = request.form.get('filename')
        model_names = sorted(request.form.getlist('descriptor[]'))
        dist_metric = request.form.get('similarity')
        specified_class = None if request.form.get('image_class') == "" else int(request.form.get('image_class'))
        topn = int(request.form.get('topn'))
        file_path = os.path.join(upload_folder, filename)

        # Vérifie si l'image existe dans le dossier 'uploads'
        if not os.path.exists(file_path):
            print("[MYAPP] >> File not found error")
            return jsonify({'error': 'File not found'})

        # Affichage pour vérification des données reçues du formulaire
        print(f"[MYAPP] >> File Name: {filename}")
        print(f"[MYAPP] >> Models: {model_names}")
        print(f"[MYAPP] >> Distance Metric: {dist_metric}")
        print(f"[MYAPP] >> Class specified by the user: {specified_class}")
        print(f"[MYAPP] >> Requested Top-N: {topn}")

        # ========== PROCESSUS DE RECHERCHE D'IMAGES SIMILAIRES ==========
        
        # 1. Charger les features des images de la base pour les descripteurs sélectionnés
        print("[MYAPP] >> Loading features from database...")
        features_dict, descriptor_label = load_features_dict(model_names, IMAGE_DB_FOLDER)
        print(f"[MYAPP] >> Loaded features for {len(features_dict)} images")

        # 2. Extraire les features de l'image requête (avec système de CACHE PERSISTANT)
        print("[MYAPP] >> Extracting features from query image...")
        
        # Création d'un identifiant unique basé sur le nom de l'image et les modèles utilisés
        models_str = "_".join(model_names)
        cache_key = hashlib.md5(f"{filename}_{models_str}".encode()).hexdigest()
        cache_file_path = os.path.join(query_cache_folder, f"{cache_key}.npy")

        # Vérification de la présence dans le cache
        if os.path.exists(cache_file_path):
            print(f"[CACHE] Hit ! Descripteur trouvé pour {filename}. Réutilisation.")
            query_feature = np.load(cache_file_path)
            # Décommente la ligne suivante si 'search_similar_images' exige une liste et non un array numpy :
            # query_feature = query_feature.tolist() 
        else:
            print("[CACHE] Miss ! Calcul du descripteur en cours...")
            query_feature = extract_combined_model_features(file_path, model_names)
            # Sauvegarde dans le volume de cache
            np.save(cache_file_path, query_feature)
            print("[CACHE] Descripteur sauvegardé dans le volume.")

        print(f"[MYAPP] >> Query feature vector size: {len(query_feature)}")
        
        
        # 3. Charger le dictionnaire des images de la base
        image_dict = get_image_dict(IMAGE_DB_FOLDER)
        print(f"[MYAPP] >> Found {len(image_dict)} images in database")

        # 4. Rechercher les images similaires avec la métrique choisie
        print(f"[MYAPP] >> Searching for {topn} similar images with metric '{dist_metric}'...")
        images_proches, predicted_class = search_similar_images(
            query_feature, 
            features_dict, 
            image_dict, 
            topn=topn, 
            dist_metric=dist_metric
        )
        print(f"[MYAPP] >> Found {len(images_proches)} similar images")
        print(f"[MYAPP] >> Predicted class: {predicted_class}")

        # 5. Déterminer la classe à utiliser pour le calcul rappel/précision
        # Utiliser la classe spécifiée par l'utilisateur si fournie, sinon utiliser la classe prédite
        class_for_rp = specified_class if specified_class is not None else predicted_class
        print(f"[MYAPP] >> Class used for RP calculation: {class_for_rp}")

        # 6. Générer l'image de la courbe de Rappel/Précision (RP)
        print("[MYAPP] >> Computing Recall/Precision curve...")
        rp_txt_file, final_precision, final_recall = Compute_RP(
            topn, 
            filename, 
            class_for_rp, 
            images_proches
        )
        print(f"[MYAPP] >> RP computed - Precision: {final_precision:.2f}%, Recall: {final_recall:.2f}%")

        # 7. Afficher et générer l'image PNG de la courbe RP
        print("[MYAPP] >> Generating RP curve image...")
        rp_img_path = Display_RP(rp_txt_file, descriptor_label)
        print(f"[MYAPP] >> RP curve generated: {rp_img_path}")

        # ================================================================
        
        # Enregistrement dans la base de données
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO search_logs 
                    (query_image_path, nb_images_returned, models_used, similarity_measure,
                     precision_score, recall_score, rp_curve_path, specified_class, predicted_class)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    file_path,
                    topn,
                    ','.join(model_names),
                    dist_metric,
                    final_precision,
                    final_recall,
                    rp_img_path,
                    specified_class,
                    predicted_class
                ))
                conn.commit()
                cursor.close()
                conn.close()
                print("[MYAPP] >> Search logged to database")
        except Exception as db_err:
            print(f"[MYAPP] >> DB logging error: {db_err}")

        # Envoi des résultats au frontend
        return jsonify({
            'filename': os.path.basename(file_path),
            'topn_similar_images': images_proches,
            'rp_curve': rp_img_path,
            'predicted_class': predicted_class,
            'specified_class': specified_class,
            'final_precision': final_precision,
            'final_recall': final_recall
        })

    except Exception as e:
        print(f"[MYAPP] >> Error during search: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Search error: {str(e)}'})


if __name__ == '__main__':
    # Créer les dossiers s'ils n'existent pas
    for folder in [upload_folder, image_db_folder, features_folder, rp_save_dir,query_cache_folder]:
        os.makedirs(folder, exist_ok=True)
    
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
