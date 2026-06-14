from werkzeug.security import generate_password_hash, check_password_hash

# 1. Générer le hash
mot_de_passe_secret = "d1Jx5"
hash_genere = generate_password_hash(mot_de_passe_secret)

print(hash_genere) 
# Exemple de résultat : pbkdf2:sha256:600000$abc123$d41d8cd98f00b204e9800998ecf8427e

# 2. Vérifier le mot de passe (lors d'une connexion)
est_valide = check_password_hash(hash_genere, "d1Jx5") 
# Retourne True
