
Rapport d'Examen MLOps / DVC - Prédiction de Silice

Auteur

    Nom : SECHAO

    Prénom : JACQUES

    E-mail : jacques.sechao@icloud.com

    Lien Repo DagsHub : https://dagshub.com/JAKE-MLE/examen-dvc


Description Technique

Ce projet met en place une architecture MLOps robuste utilisant DVC pour le versionnement des données et DagsHub pour la collaboration.

Choix de Modélisation

- Prétraitement : Suppression de la colonne date (non prédictive) et normalisation via StandardScaler.

- Modèle : RandomForestRegressor optimisé par GridSearchCV.

- Résultat : Score $R^2$ de 0.1915. Bien que le score soit modeste, l'accent a été mis sur la reproductibilité totale de la pipeline.

Pipeline DVC

La pipeline est entièrement automatisée via le fichier dvc.yaml, permettant de reconstruire le modèle de bout en bout avec une seule commande dvc repro.