import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_raw_data(input_path, output_dir):
    """
    On prépare les données en supprimant la colonne 'date' qui n'est pas
    une variable prédictive numérique exploitable par le scaler.
    """
    print(f"Lecture des données depuis : {input_path}")
    df = pd.read_csv(input_path)
    
    # --- CORRECTION ICI ---
    # On supprime la colonne 'date' si elle existe
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    # ----------------------
    
    # Séparation des features (X) et de la target (y)
    # La cible est la dernière colonne (silica_concentrate)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print("Étape de split terminée (colonne date supprimée).")

if __name__ == "__main__":
    split_raw_data("data/raw/raw.csv", "data/processed")