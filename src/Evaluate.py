import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(data_dir, model_dir, metrics_dir):
    """
    Dernière étape du workflow : on mesure l'erreur sur le set de test.
    On génère aussi le fichier de prédictions demandé.
    """
    # Chargement
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    # Prédictions
    predictions = model.predict(X_test)
    
    # Métriques
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Sauvegarde des scores
    os.makedirs(metrics_dir, exist_ok=True)
    scores = {
        "mse": float(mse),
        "r2": float(r2)
    }
    
    with open(os.path.join(metrics_dir, "scores.json"), "w") as f:
        json.dump(scores, f)

    # Sauvegarde des prédictions demandées
    pd.DataFrame(predictions, columns=['predictions']).to_csv(
        os.path.join(data_dir, "predictions.csv"), index=False
    )
    
    print(f"Évaluation terminée. R2 Score final : {r2:.4f}")

if __name__ == "__main__":
    evaluate_model("data/processed", "models", "metrics")