# data_preprocessing.py
import pandas as pd
import numpy as np

def preprocess_data(file_path):
    """
    Prétraitement du dataset des logs d'activités
    """
    print("Prétraitement des données...")
    
    # Charger le dataset
    df = pd.read_csv(file_path)
    
    # Convertir le timestamp en format datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Ajout de caractéristiques temporelles
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_after_hours'] = df['hour_of_day'].apply(lambda x: 1 if x < 8 or x > 18 else 0)
    
    # Vérifier les valeurs manquantes
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Valeurs manquantes détectées: {null_counts}")
        # Stratégie de gestion des valeurs manquantes
        df = df.dropna(subset=['user', 'pc', 'activity', 'resource'])  # Supprimer les lignes avec des valeurs critiques manquantes
    
    print(f"Prétraitement terminé. {len(df)} entrées valides.")
    return df

if __name__ == "__main__":
    # Test du script de prétraitement
    file_path = "/Users/HouDa/Desktop/Projet_NOSQL/demo_multi_activity_logs_10000.csv"
    processed_data = preprocess_data(file_path)
    print(processed_data.head())
    print(f"Nombre de lignes après traitement : {len(processed_data)}")
    print(f"Colonnes disponibles: {processed_data.columns.tolist()}")