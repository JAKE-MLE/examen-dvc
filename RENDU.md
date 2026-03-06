SECHAO JACQUES

jacques.sechao@icloud.com

https://dagshub.com/JAKE-MLE/examen-dvc








Examen MLOps - Prédiction de la Silice

Ce projet implémente une pipeline MLOps complète pour prédire la concentration de silice dans un processus de flottation minérale.

Pipeline DVC

La pipeline est divisée en 5 étapes reproductibles :

Split : Nettoyage (suppression de la date) et division du dataset.

Normalize : Mise à l'échelle des features avec StandardScaler.

GridSearch : Optimisation des hyperparamètres d'un RandomForestRegressor.

Train : Entraînement du modèle final.

Evaluate : Calcul du score R2 et génération des prédictions.

Résultats

Les métriques sont disponibles dans le dossier metrics/ et suivies via l'interface DagsHub.