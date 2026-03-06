import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import json
import os

def find_best_params(data_dir, model_dir):
    """
    Je choisis d'utiliser une Random Forest Regressor pour sa robustesse.
    Ce script explore une grille de paramètres pour optimiser le modèle.
    """
    # Chargement des données scalées
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    
    # Définition de la grille (simplifiée pour l'examen)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Recherche par grille
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Extraction des meilleurs paramètres
    best_params = grid_search.best_params_
    
    # Sauvegarde en format JSON dans le dossier models
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f)
        
    print(f"Meilleurs paramètres trouvés : {best_params}")

if __name__ == "__main__":
    find_best_params("data/processed", "models")