import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def scale_features(input_dir):
    """
    Les données de flottation ont des échelles très différentes (débits vs pH).
    J'utilise StandardScaler pour centrer et réduire les données, ce qui est 
    indispensable pour la plupart des modèles de régression.
    """
    # Chargement des données d'entraînement et de test
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    
    # Initialisation du scaler
    scaler = StandardScaler()
    
    # On "fit" sur le train uniquement pour ne pas apprendre de l'info du test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion en DataFrame pour garder la structure
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Sauvegarde des versions scalées
    X_train_scaled_df.to_csv(os.path.join(input_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(input_dir, "X_test_scaled.csv"), index=False)
    
    # Optionnel mais recommandé : sauvegarder le scaler lui-même
    # joblib.dump(scaler, "models/scaler.joblib")
    
    print("Normalisation terminée.")

if __name__ == "__main__":
    scale_features("data/processed")