"""
app_utils.py
Module utilitaire pour l'extraction de features et la recherche d'images similaires.
Contient les fonctions nécessaires du notebook Multimedia_Research_Engine_DeepLearning.ipynb
"""

import os
import pickle
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import correlation
from torchvision.models import (
    VGG16_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ConvNeXt_Base_Weights
)

# Configuration des répertoires
STATIC_DIR = 'static'
IMAGE_DB_FOLDER = os.path.join(STATIC_DIR, 'image.orig')
FEATURES_FOLDER = os.path.join(STATIC_DIR, 'features')
RP_SAVE_DIR = os.path.join(STATIC_DIR, 'rp_files')

# Créer les dossiers s'ils n'existent pas
for _d in (STATIC_DIR, IMAGE_DB_FOLDER, FEATURES_FOLDER, RP_SAVE_DIR):
    os.makedirs(_d, exist_ok=True)

# Transformation des images pour les modèles
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================================
# EXTRACTION DE FEATURES
# ==========================================================

def vit16_features(img_path):
    """
    Extrait les caractéristiques d'une image en utilisant le modèle Vision Transformer (VIT).
    
    Args:
        img_path (str): Chemin de l'image à traiter.
    
    Returns:
        np.ndarray: Vecteur de caractéristiques extraits.
    """
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    img = Image.open(img_path).convert("RGB")
    vit = vit.to(device)

    # Utilisation de la transformation par défaut pour le modèle VIT
    preprocessing = models.ViT_B_16_Weights.DEFAULT.transforms()

    img = preprocessing(img).to(device)

    # Ajout d'une dimension batch
    img = img.unsqueeze(0)
    vit.eval()
    with torch.no_grad():
        features = vit._process_input(img)

        # Extension du token CLS à la dimension batch
        batch_class_token = vit.class_token.expand(img.shape[0], -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)

        features = vit.encoder(features)

        # Nous sommes intéressés par la représentation du token CLS à la position 0
        features = features[:, 0]
        return features.cpu().detach().numpy().squeeze().flatten()


def extract_single_model_features(img_path, model_name):
    """
    Extrait les caractéristiques d'une image en utilisant un modèle spécifique.

    Args:
        img_path (str): Chemin vers l'image.
        model_name (str): Nom du modèle ('vgg16', 'resnet50', 'efficientnet', 'mobilenet', 'convnext', 'vit_b_16').

    Returns:
        np.ndarray: Vecteur de caractéristiques extrait.
    """
    img = Image.open(img_path).convert("RGB")
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    image = transform(img).unsqueeze(0).to(device)

    if model_name == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        feature_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten(start_dim=1), model.classifier)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        feature_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten(start_dim=1), model.classifier)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        feature_extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
    elif model_name == "convnext":
        model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        feature_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten(start_dim=1))
    elif model_name == "vit_b_16":
        return vit16_features(img_path)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.cpu().numpy().squeeze().flatten()


def extract_combined_model_features(img_path, model_names):
    """
    Extrait et concatène les caractéristiques d'une image à partir d'une liste de modèles.

    Args:
        img_path (str): Chemin vers l'image.
        model_names (list of str): Liste des noms des modèles utilisés.

    Returns:
        list: Vecteur de caractéristiques concaténé issu de plusieurs modèles.
    """
    model_names_sorted = sorted(model_names)
    features = []
    for model_name in model_names_sorted:
        features.extend(extract_single_model_features(img_path, model_name))
    return features


def extract_and_save_features_db_to_pkl(input_folder, model_names):
    """
    Extrait les vecteurs de caractéristiques pour chaque image d'un dossier
    en utilisant une liste de modèles, puis enregistre les résultats dans un fichier .pkl.

    Args:
        input_folder (str): Dossier contenant les images à indexer.
        model_names (list of str): Liste des modèles à utiliser pour l'extraction.
    """
    model_names_sorted = sorted(model_names)
    output_folder = FEATURES_FOLDER

    features_dict = {}
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    descriptor_label = "_".join(model_names_sorted)
    output_pickle = os.path.join(output_folder, f"{descriptor_label}.pkl")

    if os.path.exists(output_pickle):
        print(f"[MYAPP] >> {output_pickle} already exists. Skipping feature extraction.")
    else:
        for filename in tqdm(image_files, desc=f"Extracting DB image features using models: {', '.join(model_names_sorted)}"):
            image_path = os.path.join(input_folder, filename)
            try:
                features = extract_combined_model_features(image_path, model_names_sorted)
                features_dict[os.path.splitext(filename)[0]] = features
            except Exception as e:
                print(f"[MYAPP] >> Error processing {filename}: {e}")
                continue

        with open(output_pickle, 'wb') as f:
            pickle.dump(features_dict, f)

        print(f"[MYAPP] >> Feature extraction completed and saved to {output_pickle}")


def load_features_dict(model_names, image_db_folder):
    """
    Charge les vecteurs de caractéristiques depuis un fichier .pkl correspondant aux modèles spécifiés.
    Si le fichier n'existe pas, il est automatiquement généré.

    Args:
        model_names (list of str): Liste des modèles utilisés pour l'extraction.
        image_db_folder (str): Dossier contenant les images de la base.

    Returns:
        tuple:
            - dict: Dictionnaire contenant les caractéristiques extraites pour chaque image.
            - str: Nom combiné des modèles utilisé pour nommer le fichier.
    """
    model_names_sorted = sorted(model_names)
    descriptor_label = "_".join(model_names_sorted)
    output_folder = FEATURES_FOLDER
    pkl_path = os.path.join(output_folder, f"{descriptor_label}.pkl")

    if not os.path.exists(pkl_path):
        print(f"[MYAPP] >> Feature file {pkl_path} not found. Generating...")
        extract_and_save_features_db_to_pkl(image_db_folder, model_names_sorted)

    with open(pkl_path, 'rb') as f:
        features_dict = pickle.load(f)

    return features_dict, descriptor_label


# ==========================================================
# CALCUL DES DISTANCES / METRIQUES DE SIMILARITE
# ==========================================================

def euclidianDistance(vec1, vec2):
    """Calcule la distance euclidienne entre deux vecteurs."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def chiSquareDistance(vec1, vec2):
    """Calcule la distance chi-carré entre deux vecteurs."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    n = min(len(vec1), len(vec2))
    return np.sum((vec1[:n] - vec2[:n])**2 / (vec2[:n] + 1e-10))


def cosineDistance(vec1, vec2):
    """Calcule la distance cosinus entre deux vecteurs."""
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_distances(vec1, vec2).item()


def bhattacharyyaDistance(vec1, vec2):
    """Calcule la distance Bhattacharyya entre deux vecteurs."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    n = min(len(vec1), len(vec2))
    N_1, N_2 = np.sum(vec1[:n])/n, np.sum(vec2[:n])/n
    score = np.sum(np.sqrt(vec1[:n] * vec2[:n]))
    den = math.sqrt(N_1 * N_2 * n * n)
    return math.sqrt(1 - score / den) if den != 0 else 1.0


def flannMatching(vec1, vec2):
    """Calcule la distance FLANN matching entre deux vecteurs."""
    vec1 = np.float32(vec1)
    vec2 = np.float32(vec2)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = list(map(lambda x: x.distance, matcher.match(vec1, vec2)))
    return np.mean(matches) if matches else 0.0


def bruteForceMatching(vec1, vec2):
    """Calcule la distance brute force matching entre deux vecteurs."""
    vec1 = np.array(vec1).astype('uint8')
    vec2 = np.array(vec2).astype('uint8')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(vec1, vec2)))
    return np.mean(matches) if matches else 0.0


# ==========================================================
# RECHERCHE DE K PLUS PROCHES VOISINS
# ==========================================================

def getkVoisins(features_dict, query_feature, k=20, dist_metric="euclidean"):
    """
    Trouve les k plus proches voisins du vecteur requête dans le dictionnaire de features.

    Args:
        features_dict (dict): Dictionnaire {nom_image: vecteur_feature}.
        query_feature (np.ndarray or list): Vecteur de features de l'image requête.
        k (int): Nombre de voisins à retourner.
        dist_metric (str): Métrique de distance à utiliser.

    Returns:
        list: Liste de tuples (nom_image, distance) triée par distance croissante.
    """
    distances = []
    for name, vector in features_dict.items():
        if dist_metric == "euclidean":
            dist = euclidianDistance(query_feature, vector)
        elif dist_metric == "correlation":
            dist = correlation(query_feature, vector)
        elif dist_metric == "chi-square":
            dist = chiSquareDistance(query_feature, vector)
        elif dist_metric == "cosine_distance":
            dist = cosineDistance(query_feature, vector)
        elif dist_metric == "bhattacharyya":
            dist = bhattacharyyaDistance(query_feature, vector)
        elif dist_metric == "brute-force-matcher":
            dist = bruteForceMatching(query_feature, vector)
        elif dist_metric == "flann":
            dist = flannMatching(query_feature, vector)
        else:
            raise ValueError(f"Métrique inconnue : {dist_metric}")

        distances.append((name, dist))
    distances.sort(key=lambda x: x[1])
    return distances[:k]


# ==========================================================
# RECHERCHE D'IMAGES SIMILAIRES
# ==========================================================

def get_image_dict(image_db_folder):
    """
    Crée un dictionnaire mapping nom_image -> chemin_image pour la base de données.

    Args:
        image_db_folder (str): Dossier contenant les images de la base.

    Returns:
        dict: Dictionnaire {nom_sans_ext: chemin_complet}.
    """
    image_dict = {}
    if not os.path.exists(image_db_folder):
        print(f"[MYAPP] >> Image database folder not found: {image_db_folder}")
        return image_dict
    
    for filename in os.listdir(image_db_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name_without_ext = os.path.splitext(filename)[0]
            full_path = os.path.join(image_db_folder, filename)
            image_dict[name_without_ext] = full_path
    return image_dict


def search_similar_images(query_feature, features_dict, image_dict, topn=20, dist_metric="euclidean"):
    """
    Recherche les Top-N images les plus similaires dans la base de données.

    Args:
        query_feature (np.ndarray or list): Vecteur de features de l'image requête.
        features_dict (dict): Dictionnaire des features de la base.
        image_dict (dict): Dictionnaire mapping nom -> chemin des images.
        topn (int): Nombre d'images similaires à retourner.
        dist_metric (str): Métrique de distance à utiliser.

    Returns:
        tuple:
            - list: Liste des noms de fichiers des images proches.
            - int: Classe prédite (calculée par majorité des voisins).
    """
    # Recherche des k plus proches voisins
    voisins = getkVoisins(features_dict, query_feature, topn, dist_metric)

    # Recherche d'une image identique dans la BD (distance 0.0)
    image_identique = next((image_dict[name] for name, dist in voisins if dist == 0.0 and name in image_dict), None)

    # Extraction des noms de fichiers (avec extension) des images voisines
    images_proches = [os.path.basename(image_dict[v[0]]) for v in voisins if v[0] in image_dict]

    if image_identique is not None:
        # Si une image identique existe, on extrait sa classe directement
        img_name = os.path.basename(image_identique)
        predicted_image_classe = int(os.path.splitext(img_name)[0]) // 100
    else:
        # Si aucune image identique, on déduit la classe par majorité
        classes = []
        for nom_image in images_proches:
            try:
                classe = int(os.path.splitext(nom_image)[0]) // 100
                classes.append(classe)
            except:
                continue

        # Classe la plus fréquente parmi les plus proches voisins
        predicted_image_classe = Counter(classes).most_common(1)[0][0] if classes else None

    return images_proches, predicted_image_classe


# ==========================================================
# CALCUL ET AFFICHAGE DE LA COURBE RAPPEL/PRECISION
# ==========================================================

def Compute_RP(top, query_image_name, image_class, similar_images):
    """
    Calcule et enregistre la courbe Rappel-Précision (RP) pour une image requête.

    Args:
        top (int): Nombre d'images les plus similaires analysées.
        query_image_name (str): Nom de l'image requête (ex: "107.jpg").
        image_class (int): Classe de l'image requête.
        similar_images (list of str): Liste des noms des images les plus proches.

    Returns:
        tuple:
            - str: Nom du fichier .txt contenant les valeurs RP.
            - float: Valeur finale de précision.
            - float: Valeur finale de rappel.
    """
    relevance_flags = []
    rp = []

    for j in range(min(top, len(similar_images))):
        retrieved_class = int(os.path.splitext(os.path.basename(similar_images[j]))[0]) // 100
        relevance_flags.append("pertinent" if image_class == retrieved_class else "non pertinent")

    val = 0
    for i in range(min(top, len(relevance_flags))):
        if relevance_flags[i] == "pertinent":
            val += 1
        precision = (val / (i + 1)) * 100 if (i + 1) > 0 else 0
        recall = (val / top) * 100 if top > 0 else 0
        rp.append(f"{precision} {recall}")

    prefix = os.path.splitext(os.path.basename(query_image_name))[0]

    rp_txt_dir = os.path.join(RP_SAVE_DIR, "rp_txt_files")
    os.makedirs(rp_txt_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(rp_txt_dir) if f.startswith(prefix + "_RP")] if os.path.exists(rp_txt_dir) else []
    rp_txt_file = f"{prefix}_RP-{len(existing_files) + 1}.txt"
    rp_file_path = os.path.join(rp_txt_dir, rp_txt_file)

    with open(rp_file_path, 'w') as s:
        for line in rp:
            s.write(line + '\n')

    print(f"[MYAPP] >> RP enregistré dans {rp_file_path}")

    if rp:
        final_precision, final_recall = map(float, rp[-1].split())
    else:
        final_precision, final_recall = 0.0, 0.0
    
    return rp_txt_file, final_precision, final_recall


def Display_RP(rp_file_name, descriptor_label):
    """
    Affiche la courbe Rappel-Précision (RP) à partir d'un fichier texte et génère une image PNG.

    Args:
        rp_file_name (str): Nom du fichier .txt contenant les valeurs RP.
        descriptor_label (str): Étiquette décrivant le(s) modèle(s) utilisé(s).

    Returns:
        str: Chemin vers l'image PNG générée de la courbe RP.
    """
    x, y = [], []

    rp_file_path = os.path.join(RP_SAVE_DIR, 'rp_txt_files', rp_file_name)
    with open(rp_file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 2:
                x.append(float(values[0]))
                y.append(float(values[1]))

    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    rp_img_dir = os.path.join(RP_SAVE_DIR, 'rp_img_files')
    os.makedirs(rp_img_dir, exist_ok=True)

    rp_file_path_png = os.path.join(
        rp_img_dir, os.path.splitext(os.path.basename(rp_file_path))[0]
    ) + '.png'

    plt.figure(figsize=(8, 6))
    plt.plot(y_tensor, x_tensor, 'C1', label=descriptor_label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Recall/Precision Curve (RP)")
    plt.legend()
    plt.grid(True)
    plt.savefig(rp_file_path_png)
    plt.close()
    
    print(f"[MYAPP] >> RP curve saved to {rp_file_path_png}")
    return rp_file_path_png
