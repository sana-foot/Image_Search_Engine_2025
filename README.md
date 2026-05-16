# Image Search Engine Based on Descriptors Extracted Using AI Models

Cette application Flask permet de rechercher des images similaires dans une base de données en utilisant des descripteurs d’images extraits à l’aide de modèles de vision par ordinateur pré-entraînés.

## Fonctionnalités

- **Extraction de caractéristiques (features)** :
  - À partir de modèles pré-entraînés tels que VGG16, ResNet50, MobileNet, EfficientNet, ConvNeXt et ViT.
  - Les descripteurs des images dans la base de données sont extraits et sauvegardés dans des fichiers .`.pkl` situés dans `static/features/`.
  - Descripteurs extraits en utilisant un ou plusieurs modèles.
  - Pour les combinaisons de modèles, les descripteurs sont concaténés.

- **Recherche d’image** :
  - Calcule le descripteur de l’image requête.
  - Compare les descripteurs avec ceux enregistrés dans les fichiers `.pkl`.
  - Utilise une métrique choisie (euclidean, cosine, etc.) pour calculer la distance.

- **Prédiction de la classe de l’image** :
  - Si une image identique (distance = 0) est trouvée, sa classe est directement attribuée.
  - Sinon, la classe la plus fréquente parmi les images similaires est choisie.

- **Sauvegarde dans base de données** :
  - Les résultats de recherche sont enregistrés dans une base MySQL.
  - PHPMyAdmin permet de visualiser les données facilement.

## Services Docker

- **Web (Flask)** : application principale accessible sur http://localhost:8080
- **MySQL** : base de données utilisée pour stocker les résultats.
- **phpMyAdmin** : interface web pour consulter les données dans MySQL.
  - URL : http://localhost:8081
  - Utilisateur : `user`
  - Mot de passe : `I-ILIA-208`

## Déploiement de l’application

### Prérequis
- Docker et Docker Compose installés.

### Étapes
1. **Cloner le projet :**
    
    ```bash
    git clone https://github.com/TValisoa/Image_Search_Engine_2025.git
    cd Image_Search_Engine_2025
    ```
2. **Construire et lancer les conteneurs :**

    ```bash
    docker-compose up --build
    ```

3. **Accéder à l'application :**

    - Application Flask (recherche d'images) : [http://localhost:8080](http://localhost:8080)
    - phpMyAdmin (interface de gestion MySQL) : [http://localhost:8081](http://localhost:8081)

        - **Utilisateur** : `root`  
        - **Mot de passe** : `I-ILIA-208`

4. **Commandes Docker utiles :**

    ```bash
    docker-compose down -v    # Supprime les conteneurs et les volumes
    docker-compose down       # Supprime uniquement les conteneurs
    docker-compose up --build # Relancer les services après un arrêt
    docker exec -it image_retrieval_web bash # Accéder au terminal du conteneur de l'application
    docker stop image_retrieval_web mysql_image_retrieval_db image_retrieval_phpmyadmin # Stopper les conteneurs
    ```  
---
© 2025 DeepILIA.
