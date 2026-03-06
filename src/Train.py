import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os

def train_final_model(data_dir, model_dir):
    """
    Une fois les paramètres optimisés trouvés, on entraîne le modèle final
    sur l'intégralité du set d'entraînement.
    """
    # Récupération des données et des paramètres
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    
    with open(os.path.join(model_dir, "best_params.json"), "r") as f:
        params = json.load(f)
    
    # Entraînement avec les bons paramètres
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Sauvegarde du modèle sérialisé
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    
    print("Entraînement du modèle terminé et sauvegardé.")

if __name__ == "__main__":
    train_final_model("data/processed", "models")