import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import xgboost as xgb
import math
from scipy.stats import zscore
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')


def create_autoencoder(input_dim, encoding_dim=3, dropout_rate=0.2):
    """Créer un autoencoder amélioré pour la détection d'anomalies
    
    Args:
        input_dim (int): Dimension des données d'entrée
        encoding_dim (int): Dimension de la couche latente
        dropout_rate (float): Taux de dropout pour la régularisation
        
    Returns:
    
        tuple: (autoencoder, encoder_model) - modèle complet et encodeur seul
    """
    # Définition de l'encodeur
    input_layer = Input(shape=(input_dim,))
    
    # Architecture de l'encodeur avec dropout et plus de couches
    encoder = Dense(int(input_dim * 0.8), activation='relu', kernel_initializer='he_normal')(input_layer)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(int(input_dim * 0.6), activation='relu', kernel_initializer='he_normal')(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(int(input_dim * 0.4), activation='relu', kernel_initializer='he_normal')(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(encoding_dim, activation='relu', kernel_initializer='he_normal')(encoder)
    
    # Couche de représentation latente
    bottleneck = encoder
    
    # Architecture du décodeur avec dropout et plus de couches
    decoder = Dense(int(input_dim * 0.4), activation='relu', kernel_initializer='he_normal')(bottleneck)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(int(input_dim * 0.6), activation='relu', kernel_initializer='he_normal')(decoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(int(input_dim * 0.8), activation='relu', kernel_initializer='he_normal')(decoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(input_dim, activation='sigmoid', kernel_initializer='glorot_normal')(decoder)
    
    # Création du modèle complet
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Compilation du modèle avec un taux d'apprentissage adaptatif
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Modèle de l'encodeur seul
    encoder_model = Model(inputs=input_layer, outputs=bottleneck)
    
    return autoencoder, encoder_model



# Fonction pour créer un jeu de données synthétique pour les tests
def create_synthetic_data(n_samples=200, n_features=8):
   
    np.random.seed(42)
    # Données normales
    X_normal = np.random.randn(n_samples - 20, n_features)
    
    # Anomalies
    X_anomaly = np.random.randn(20, n_features) * 2 + 3
    
    # Combinaison des données
    X = np.vstack([X_normal, X_anomaly])
    
    # Étiquettes (1 pour anomalies, 0 pour normaux)
    y = np.zeros(n_samples)
    y[n_samples - 20:] = 1
    
    # Création d'un DataFrame avec des noms de caractéristiques significatifs
    feature_names = ['total_activities', 'after_hours_activities', 'weekend_activities',
                    'unique_systems', 'unique_resources', 'activity_types',
                    'activity_entropy', 'temporal_entropy']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['user'] = [f'user_{i}' for i in range(n_samples)]
    df['is_anomaly'] = y
    
    # Ajout de bruit et de corrélations
    df['after_hours_ratio'] = df['after_hours_activities'] / (df['total_activities'].abs() + 1)
    df['weekend_ratio'] = df['weekend_activities'] / (df['total_activities'].abs() + 1)
    
    
    return df


# Analyse non supervisée
def unsupervised_analysis(df, contamination=0.1):
    print("\n=== ANALYSE NON SUPERVISÉE ===")
    # Définir le répertoire de sortie pour les visualisations
    output_dir = os.path.join(os.path.dirname(__file__), 'visualisations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration générale des visualisations
    plt.style.use('default')
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop(['is_anomaly'], errors='ignore')
    X = df[numeric_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # 1. Isolation Forest optimisé
    print("Exécution de l'Isolation Forest...")
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples='auto',
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    y_pred_iso = iso_forest.fit_predict(X_scaled)
    scores_iso = -iso_forest.decision_function(X_scaled)
    results['isolation_forest'] = {
        'predictions': np.where(y_pred_iso == -1, 1, 0),
        'scores': scores_iso,
        'model': iso_forest
    }
    
    # 2. One-Class SVM optimisé
    print("Exécution de One-Class SVM...")
    # Optimisation des hyperparamètres
    ocsvm = OneClassSVM(
        kernel="rbf",
        nu=contamination,
        gamma='scale',  # Meilleure adaptation à l'échelle des données
        cache_size=500,  # Augmentation de la mémoire cache
        max_iter=-1,  # Pas de limite d'itérations
        shrinking=True,  # Utilisation de l'heuristique de réduction
        tol=1e-4  # Tolérance plus stricte
    )
    y_pred_ocsvm = ocsvm.fit_predict(X_scaled)
    scores_ocsvm = -ocsvm.decision_function(X_scaled)
    results['ocsvm'] = {
        'predictions': np.where(y_pred_ocsvm == -1, 1, 0),
        'scores': scores_ocsvm,
        'model': ocsvm
    }
    
    # 3. K-Means Clustering optimisé
    print("Exécution de K-Means...")
    # Optimisation des hyperparamètres
    kmeans = KMeans(
        n_clusters=2,
        init='k-means++',  # Meilleure initialisation
        n_init=20,  # Plus d'initialisations pour trouver le meilleur modèle
        max_iter=500,  # Plus d'itérations maximum
        tol=1e-6,  # Tolérance plus stricte
        algorithm='elkan',  # Algorithme plus efficace
        random_state=42
    )
    y_pred_kmeans = kmeans.fit_predict(X_scaled)
    
    # Déterminer quel cluster représente les anomalies (le plus petit)
    counts = np.bincount(y_pred_kmeans)
    anomaly_cluster = np.argmin(counts)
    
    # Calculer les distances aux centroïdes avec une normalisation améliorée
    distances = np.min(kmeans.transform(X_scaled), axis=1)
    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    results['kmeans'] = {
        'predictions': np.where(y_pred_kmeans == anomaly_cluster, 1, 0),
        'scores': distances,
        'model': kmeans
    }
    
    #DBSCAN est un algorithme de clustering (regroupement) basé sur la densité
    #détecter des anomalies (ou points de bruit) qui ne font pas partie de ces clusters.
    # 4. DBSCAN optimisé
    print("Exécution de DBSCAN...")
    # Optimisation des hyperparamètres
    dbscan = DBSCAN(
        eps=0.5,  # Rayon du voisinage réduit pour une meilleure sensibilité
        min_samples=max(5, int(len(X_scaled) * 0.01)),  # Adaptation dynamique basée sur la taille des données
        metric='euclidean',
        algorithm='auto',
        leaf_size=30,
        n_jobs=-1  # Utilisation de tous les cœurs disponibles
    )
    y_pred_dbscan = dbscan.fit_predict(X_scaled)
    
    # Calcul des scores basé sur la distance aux points du cluster le plus proche
    core_samples_mask = np.zeros_like(y_pred_dbscan, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    # Calcul des scores de distance pour chaque point
    from sklearn.metrics.pairwise import euclidean_distances
    distances = np.zeros(len(X_scaled))
    for i in range(len(X_scaled)):
        if y_pred_dbscan[i] == -1:  # Pour les points de bruit
            min_dist = np.min(euclidean_distances([X_scaled[i]], X_scaled[core_samples_mask]))
            distances[i] = min_dist
    
    # Normalisation des scores
    if len(distances[distances > 0]) > 0:
        distances = (distances - np.min(distances[distances > 0])) / \
                   (np.max(distances[distances > 0]) - np.min(distances[distances > 0]))
    
    # Normalisation des scores DBSCAN
    distances = np.zeros(len(X_scaled))
    for i in range(len(X_scaled)):
        if y_pred_dbscan[i] == -1:  # Pour les points de bruit
            min_dist = np.min(euclidean_distances([X_scaled[i]], X_scaled[core_samples_mask]))
            distances[i] = min_dist
        else:
            # Pour les points dans les clusters, utiliser la distance au centre du cluster
            cluster_points = X_scaled[y_pred_dbscan == y_pred_dbscan[i]]
            if len(cluster_points) > 0:
                distances[i] = np.mean(euclidean_distances([X_scaled[i]], cluster_points))
    
    # Normalisation min-max des scores
    if len(distances) > 0:
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-10)
    
    results['dbscan'] = {
        'predictions': np.where(y_pred_dbscan == -1, 1, 0),
        'scores': distances,
        'model': dbscan
    }
    
    # 5. Autoencoder optimisé
    print("Exécution de l'Autoencoder...")
    input_dim = X_scaled.shape[1]
    autoencoder, encoder = create_autoencoder(input_dim)
    
    # Entraînement amélioré de l'autoencoder
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    
    # Utilisation de la validation croisée pour un meilleur entraînement
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Calcul amélioré des scores d'anomalie
    reconstructed = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, (1 - contamination) * 100)
    
    results['autoencoder'] = {
        'predictions': np.where(mse > threshold, 1, 0),
        'scores': mse,
        'model': autoencoder,
        'encoder': encoder,
        'threshold': threshold
    }
    
    # 6. Système de vote majoritaire
    print("Application du système de vote majoritaire...")
    all_predictions = np.column_stack([
        results['isolation_forest']['predictions'],
        results['ocsvm']['predictions'],
        results['kmeans']['predictions'],
        results['dbscan']['predictions'],
        results['autoencoder']['predictions']
    ])
    
    # Calcul du vote majoritaire
    ensemble_predictions = np.sum(all_predictions, axis=1)
    ensemble_predictions = np.where(ensemble_predictions >= 3, 1, 0)  # Seuil de majorité à 3 votes
    
    # Calcul du score d'ensemble normalisé
    ensemble_scores = np.zeros(len(X_scaled))
    for i in range(len(X_scaled)):
        # Moyenne pondérée des scores normalisés
        weights = [0.25, 0.2, 0.15, 0.2, 0.2]  # Poids ajustés selon la performance des modèles
        normalized_scores = [
            results['isolation_forest']['scores'][i],
            results['ocsvm']['scores'][i],
            results['kmeans']['scores'][i],
            results['dbscan']['scores'][i],
            results['autoencoder']['scores'][i]
        ]
        ensemble_scores[i] = np.average(normalized_scores, weights=weights)
    
    results['ensemble'] = {
        'predictions': ensemble_predictions,
        'scores': ensemble_scores
    }
    
    # Réduction de dimension avec PCA pour la visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualisations améliorées
    plt.style.use('default')
    
    # 1. Visualisation PCA avec style amélioré
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['is_anomaly'] if 'is_anomaly' in df.columns else 'blue',
                cmap='coolwarm', alpha=0.7, s=100)
    plt.title('Analyse en Composantes Principales (ACP) des Données', fontsize=14, pad=20)
    plt.xlabel('Première Composante Principale', fontsize=12)
    plt.ylabel('Deuxième Composante Principale', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Anomalie')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_non_supervise.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Courbes ROC pour chaque modèle
    if 'is_anomaly' in df.columns:
        plt.figure(figsize=(12, 8))
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        for i, (model_name, result) in enumerate(results.items()):
            if model_name != 'ensemble':
                # Vérification des données avant de tracer la courbe ROC
                if len(df['is_anomaly']) > 0 and len(result['scores']) > 0:
                    try:
                        fpr, tpr, _ = roc_curve(df['is_anomaly'], result['scores'])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.2f})',
                                color=colors[i % len(colors)], linewidth=2)
                    except Exception as e:
                        print(f"Erreur lors du calcul de la courbe ROC pour {model_name}: {str(e)}")
                        continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs', fontsize=12)
        plt.ylabel('Taux de Vrais Positifs', fontsize=12)
        plt.title('Courbes ROC des Modèles de Détection d\'Anomalies', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10, bbox_to_anchor=(1.15, 0))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'courbes_roc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Métriques de performance comparatives
        metrics_data = []
        for model_name, result in results.items():
            if model_name != 'ensemble':
                metrics_data.append({
                    'Modèle': model_name,
                    'Précision': precision_score(df['is_anomaly'], result['predictions']),
                    'Rappel': recall_score(df['is_anomaly'], result['predictions']),
                    'F1-Score': f1_score(df['is_anomaly'], result['predictions'])
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        plt.figure(figsize=(12, 6))
        metrics_df.set_index('Modèle').plot(kind='bar', width=0.8)
        plt.title('Comparaison des Métriques de Performance', fontsize=14)
        plt.xlabel('Modèles', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(title='Métriques', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metriques_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Matrice de corrélation entre les prédictions des modèles
        correlation_data = pd.DataFrame({
            model_name: result['predictions'] 
            for model_name, result in results.items() 
            if model_name != 'ensemble'
        })
        correlation_data['Réel'] = df['is_anomaly']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True, cbar_kws={'label': 'Corrélation'})
        plt.title('Corrélation entre les Prédictions des Modèles', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Création d'un dataframe pour les résultats
    results_df = pd.DataFrame({
        'user': df['user'],
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'IsoForest_anomaly': results['isolation_forest']['predictions'],
        'IsoForest_score': results['isolation_forest']['scores'],
        'OCSVM_anomaly': results['ocsvm']['predictions'],
        'OCSVM_score': results['ocsvm']['scores'],
        'KMeans_anomaly': results['kmeans']['predictions'],
        'KMeans_score': results['kmeans']['scores'],
        'DBSCAN_anomaly': results['dbscan']['predictions'],
        'DBSCAN_score': results['dbscan']['scores'],  # Ajout de la colonne manquante
        'Autoencoder_anomaly': results['autoencoder']['predictions'],
        'Autoencoder_score': results['autoencoder']['scores']
    })
    
    # Si des étiquettes réelles sont disponibles
    if 'is_anomaly' in df.columns:
        results_df['real_anomaly'] = df['is_anomaly'].values
    
    # Sauvegarder les autres visualisations
    # Distribution des scores avec style amélioré
    plt.figure(figsize=(15, 10))
    plt.style.use('default')
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    for i, (model, result) in enumerate(results.items()):
        if model != 'ensemble':
            plt.subplot(2, 3, i+1)
            plt.hist(result['scores'], bins=50, density=True, alpha=0.7, color=colors[i])
            plt.title(f'Distribution des Scores - {model.replace("_", " ").title()}', fontsize=12)
            plt.xlabel('Score d\'Anomalie', fontsize=10)
            plt.ylabel('Fréquence', fontsize=10)
            plt.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution des Scores par Méthode de Détection', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Matrice de confusion avec style amélioré
    if 'is_anomaly' in df.columns:
        plt.figure(figsize=(15, 10))
        plt.style.use('default')
        
        for i, (model, result) in enumerate(results.items()):
            if model != 'ensemble':
                plt.subplot(2, 3, i+1)
                cm = confusion_matrix(df['is_anomaly'], result['predictions'])
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.title(f'Matrice de Confusion - {model.replace("_", " ").title()}', fontsize=12)
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Normal', 'Anomalie'])
                plt.yticks(tick_marks, ['Normal', 'Anomalie'])
                
                # Ajout des valeurs dans les cellules
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
                
                plt.xlabel('Prédiction', fontsize=10)
                plt.ylabel('Réalité', fontsize=10)
        
        plt.suptitle('Matrices de Confusion par Méthode de Détection', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'matrices_confusion.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Visualisations individuelles pour chaque modèle
    for model_name, result in results.items():
        if model_name != 'ensemble':
            # 1. Distribution des scores
            plt.figure(figsize=(10, 6))
            plt.hist(result['scores'], bins=50, density=True, alpha=0.7, 
                     color=colors[list(results.keys()).index(model_name)])
            plt.title(f'Distribution des Scores - {model_name.replace("_", " ").title()}')
            plt.xlabel('Score d\'Anomalie')
            plt.ylabel('Densité')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{model_name}_distribution.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            if 'is_anomaly' in df.columns:
                # 2. Matrice de confusion
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(df['is_anomaly'], result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Matrice de Confusion - {model_name.replace("_", " ").title()}')
                plt.xlabel('Prédiction')
                plt.ylabel('Réalité')
                plt.savefig(os.path.join(output_dir, f'{model_name}_confusion.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Courbe ROC
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(df['is_anomaly'], result['scores'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}',
                         color=colors[list(results.keys()).index(model_name)])
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('Taux de Faux Positifs')
                plt.ylabel('Taux de Vrais Positifs')
                plt.title(f'Courbe ROC - {model_name.replace("_", " ").title()}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'{model_name}_roc.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()
    
    # Affichage des performances finales
    if 'is_anomaly' in df.columns:
        print("\nPerformances finales des modèles :")
        print("\nModèle               Précision  Rappel    F1-Score")
        print("-" * 55)
        
        for model_name, result in results.items():
            if model_name != 'ensemble':
                precision = precision_score(df['is_anomaly'], result['predictions'])
                recall = recall_score(df['is_anomaly'], result['predictions'])
                f1 = f1_score(df['is_anomaly'], result['predictions'])
                print(f"{model_name.replace('_', ' ').title():<20} {precision:.3f}    {recall:.3f}    {f1:.3f}")
        
        # Performances du modèle d'ensemble
        precision = precision_score(df['is_anomaly'], results['ensemble']['predictions'])
        recall = recall_score(df['is_anomaly'], results['ensemble']['predictions'])
        f1 = f1_score(df['is_anomaly'], results['ensemble']['predictions'])
        print("-" * 55)
        print(f"{'Ensemble':<20} {precision:.3f}    {recall:.3f}    {f1:.3f}")
    
    return results, results_df, X_pca


# Analyse supervisée
def supervised_analysis(df):
    print("\n=== ANALYSE SUPERVISÉE ===")
    
    try:
        if 'is_anomaly' not in df.columns:
            raise ValueError("Pas d'étiquettes disponibles pour l'apprentissage supervisé")
        
        # Préparation des données
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop(['is_anomaly'], errors='ignore')
        X = df[numeric_columns].values
        y = df['is_anomaly'].values
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Gestion du déséquilibre des classes avec SMOTE
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # Création d'une pipeline de rééchantillonnage optimisée
        sampling_pipeline = Pipeline([
            ('over', SMOTE(sampling_strategy=0.7, random_state=42)),
            ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=42))
        ])
        
        # Division train/test avec stratification
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # Application du rééchantillonnage
        X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)
        
        results = {}
        
        # 1. Random Forest optimisé
        print("Entraînement de Random Forest...")
        rf_params = {
            'n_estimators': [200],
            'max_depth': [10],
            'min_samples_split': [4],
            'min_samples_leaf': [2],
            'max_features': ['sqrt'],
            'class_weight': ['balanced']
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_cv = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
            rf_cv.fit(X_train_resampled, y_train_resampled)
        
        rf = rf_cv.best_estimator_
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'model': rf,
            'y_true': y_test,
            'y_pred': y_pred_rf,
            'y_prob': y_prob_rf,
            'cv_scores': cross_val_score(rf, X_scaled, y, cv=3, scoring='roc_auc')
        }
        
        # 2. SVM optimisé
        print("Entraînement de SVM...")
        svm_params = {
            'kernel': ['rbf'],
            'C': [10.0],
            'gamma': ['scale'],
            'class_weight': ['balanced'],
            'probability': [True]
        }
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            svm = SVC(random_state=42)
            svm_cv = GridSearchCV(svm, svm_params, cv=3, scoring='f1', n_jobs=-1)
            svm_cv.fit(X_train_resampled, y_train_resampled)
        
        svm = svm_cv.best_estimator_
        y_pred_svm = svm.predict(X_test)
        y_prob_svm = svm.predict_proba(X_test)[:, 1]
        
        results['svm'] = {
            'model': svm,
            'y_true': y_test,
            'y_pred': y_pred_svm,
            'y_prob': y_prob_svm,
            'cv_scores': cross_val_score(svm, X_scaled, y, cv=3, scoring='roc_auc')
        }
        
        # 3. XGBoost optimisé
        print("Entraînement de XGBoost...")
        xgb_params = {
            'learning_rate': [0.01],
            'n_estimators': [300],
            'max_depth': [6],
            'min_child_weight': [2],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=1,
            random_state=42,
            eval_metric='logloss'
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            xgb_cv = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='f1', n_jobs=-1)
            xgb_cv.fit(X_train_resampled, y_train_resampled)
        
        xgb_model = xgb_cv.best_estimator_
        eval_set = [(X_train_resampled, y_train_resampled), (X_test, y_test)]
        
        xgb_model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=eval_set,
            verbose=False
        )
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        results['xgboost'] = {
            'model': xgb_model,
            'y_true': y_test,
            'y_pred': y_pred_xgb,
            'y_prob': y_prob_xgb,
            'cv_scores': cross_val_score(xgb_model, X_scaled, y, cv=3, scoring='roc_auc')
        }
        
        # Préparation des scores pour comparaison
        eval_df = pd.DataFrame({
            'real_anomaly': y_test,
            'RF_pred': y_pred_rf,
            'RF_prob': y_prob_rf,
            'SVM_pred': y_pred_svm,
            'SVM_prob': y_prob_svm,
            'XGB_pred': y_pred_xgb,
            'XGB_prob': y_prob_xgb
        })
        
        return results, eval_df
        
    except Exception as e:
        print(f"Erreur lors de l'analyse supervisée: {str(e)}")
        return None, None


# Visualisations
def create_visualizations(unsupervised_results, supervised_results, X_pca, df, unsup_df, sup_df=None):
    print("\n=== CRÉATION DES VISUALISATIONS ===")
    
    # Configurations des plots
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    # Création des visualisations pour l'analyse non supervisée
    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2)
    
    # Assurer que le répertoire de sortie existe
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration globale de matplotlib
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # 1. Comparaison des performances des modèles non supervisés
    plt.subplot(gs[0, 0])
    unsupervised_metrics = []
    model_names = ['Isolation Forest', 'One-Class SVM', 'K-Means', 'DBSCAN', 'Autoencoder']
    
    if 'is_anomaly' in df.columns:
        for model in ['isolation_forest', 'ocsvm', 'kmeans', 'dbscan', 'autoencoder']:
            if model in unsupervised_results:
                metrics = {
                    'accuracy': accuracy_score(df['is_anomaly'], unsupervised_results[model]['predictions']),
                    'precision': precision_score(df['is_anomaly'], unsupervised_results[model]['predictions'], zero_division=0),
                    'recall': recall_score(df['is_anomaly'], unsupervised_results[model]['predictions']),
                    'f1': f1_score(df['is_anomaly'], unsupervised_results[model]['predictions'])
                }
                unsupervised_metrics.append(metrics)
    
        metrics_df = pd.DataFrame(unsupervised_metrics, index=model_names)
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title('Comparaison des Performances - Modèles Non Supervisés')
        plt.xlabel('Modèles')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
    
    # 2. Comparaison des performances des modèles supervisés
    if supervised_results is not None and sup_df is not None:
        plt.subplot(gs[0, 1])
        supervised_metrics = []
        supervised_names = ['Random Forest', 'SVM', 'XGBoost']
        
        for model in ['random_forest', 'svm', 'xgboost']:
            if model in supervised_results:
                metrics = {
                    'accuracy': accuracy_score(supervised_results[model]['y_true'], supervised_results[model]['y_pred']),
                    'precision': precision_score(supervised_results[model]['y_true'], supervised_results[model]['y_pred']),
                    'recall': recall_score(supervised_results[model]['y_true'], supervised_results[model]['y_pred']),
                    'f1': f1_score(supervised_results[model]['y_true'], supervised_results[model]['y_pred'])
                }
                supervised_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(supervised_metrics, index=supervised_names)
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title('Comparaison des Performances - Modèles Supervisés')
        plt.xlabel('Modèles')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
    
    # 3. Courbes ROC pour les modèles supervisés
    if supervised_results is not None and sup_df is not None:
        plt.subplot(gs[1, 0])
        for model, color, name in zip(['random_forest', 'svm', 'xgboost'], colors[:3], supervised_names):
            if model in supervised_results:
                fpr, tpr, _ = roc_curve(supervised_results[model]['y_true'], supervised_results[model]['y_prob'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbes ROC - Modèles Supervisés')
        plt.legend(loc='lower right')
    
    # 4. Distribution des scores d'anomalie
    plt.subplot(gs[1, 1])
    for model, color, name in zip(['isolation_forest', 'ocsvm', 'kmeans', 'dbscan', 'autoencoder'], 
                                colors, model_names):
        if model in unsupervised_results:
            scores = unsupervised_results[model]['scores']
            sns.kdeplot(scores, color=color, label=name)
    
    plt.title("Distribution des Scores d'Anomalie")
    plt.xlabel("Score d'Anomalie")
    plt.ylabel("Densité")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # 1. Distribution des scores d'anomalie par méthode non supervisée avec KDE amélioré
    plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3)
    
    # Isolation Forest avec distribution par classe
    plt.subplot(gs[0, 0])
    sns.kdeplot(data=unsup_df, x='IsoForest_score', hue='IsoForest_anomaly', 
                palette=[colors[0], colors[1]], fill=True, common_norm=False)
    plt.title('Distribution des scores - Isolation Forest', fontsize=12)
    plt.xlabel('Score d\'anomalie')
    
    # One-Class SVM avec distribution par classe
    plt.subplot(gs[0, 1])
    sns.kdeplot(data=unsup_df, x='OCSVM_score', hue='OCSVM_anomaly', 
                palette=[colors[0], colors[1]], fill=True, common_norm=False)
    plt.title('Distribution des scores - One-Class SVM', fontsize=12)
    plt.xlabel('Score d\'anomalie')
    
    # K-Means avec distribution par classe
    plt.subplot(gs[0, 2])
    sns.kdeplot(data=unsup_df, x='KMeans_score', hue='KMeans_anomaly', 
                palette=[colors[0], colors[1]], fill=True, common_norm=False)
    plt.title('Distribution des scores - K-Means', fontsize=12)
    plt.xlabel('Distance au centroïde')
    
    # DBSCAN avec distribution par classe
    plt.subplot(gs[1, 0])
    sns.kdeplot(data=unsup_df, x='DBSCAN_score', hue='DBSCAN_anomaly',
                palette=[colors[0], colors[1]], fill=True, common_norm=False)
    plt.title('Distribution des scores - DBSCAN', fontsize=12)
    plt.xlabel('Score d\'anomalie')
    
    # Autoencoder avec distribution par classe
    plt.subplot(gs[1, 1])
    sns.kdeplot(data=unsup_df, x='Autoencoder_score', hue='Autoencoder_anomaly',
                palette=[colors[0], colors[1]], fill=True, common_norm=False)
    plt.title('Distribution des scores - Autoencoder', fontsize=12)
    plt.xlabel('Erreur de reconstruction')
    
    # Visualisation de l'espace latent de l'autoencoder
    plt.subplot(gs[1, 2])
    if 'autoencoder' in unsupervised_results:
        encoder = unsupervised_results['autoencoder']['encoder']
        latent_space = encoder.predict(df[numeric_columns].values)
        if latent_space.shape[1] >= 2:
            plt.scatter(latent_space[:, 0], latent_space[:, 1], 
                       c=unsup_df['Autoencoder_anomaly'], cmap='coolwarm')
            plt.title('Espace latent - Autoencoder', fontsize=12)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
    
    # Comparaison des performances
    plt.subplot(gs[2, :])
    if 'real_anomaly' in unsup_df.columns:
        methods = ['IsoForest', 'OCSVM', 'KMeans', 'DBSCAN', 'Autoencoder']
        metrics_data = []
        
        for method in methods:
            if f'{method}_anomaly' in unsup_df.columns:
                precision = precision_score(unsup_df['real_anomaly'], unsup_df[f'{method}_anomaly'], zero_division=0)
                recall = recall_score(unsup_df['real_anomaly'], unsup_df[f'{method}_anomaly'])
                f1 = f1_score(unsup_df['real_anomaly'], unsup_df[f'{method}_anomaly'])
                metrics_data.append([method, precision, recall, f1])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Méthode', 'Précision', 'Rappel', 'F1-Score'])
        metrics_df = metrics_df.set_index('Méthode')
        
        sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.2f', 
                    cbar_kws={'label': 'Score'}, center=0.5)
        plt.title('Comparaison des performances des méthodes non supervisées', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('distribution_scores_non_supervise.png', dpi=300, bbox_inches='tight')
    
    # 2. Visualisation avancée avec t-SNE et PCA
    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3)
    
    # Calcul t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)
    
    # PCA avec scores d'anomalie
    plt.subplot(gs[0, 0])
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=unsup_df['IsoForest_score'], 
                         cmap='viridis', s=70, alpha=0.7)
    plt.colorbar(scatter, label='Score d\'anomalie')
    plt.title('Projection PCA - Scores Isolation Forest', fontsize=12)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # t-SNE avec scores d'anomalie
    plt.subplot(gs[0, 1])
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=unsup_df['IsoForest_score'], 
                         cmap='viridis', s=70, alpha=0.7)
    plt.colorbar(scatter, label='Score d\'anomalie')
    plt.title('Projection t-SNE - Scores Isolation Forest', fontsize=12)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Matrice de confusion combinée
    plt.subplot(gs[0, 2])
    if 'real_anomaly' in unsup_df.columns:
        methods = ['IsoForest', 'OCSVM', 'KMeans', 'DBSCAN', 'Autoencoder']
        confusion_matrices = {}
        for method in methods:
            if f'{method}_anomaly' in unsup_df.columns:
                cm = confusion_matrix(unsup_df['real_anomaly'], 
                                     unsup_df[f'{method}_anomaly'], 
                                     normalize='true')
                confusion_matrices[method] = cm
        
        avg_cm = np.mean([cm for cm in confusion_matrices.values()], axis=0)
        sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=['Normal', 'Anomalie'],
                    yticklabels=['Normal', 'Anomalie'])
        plt.title('Matrice de confusion moyenne normalisée', fontsize=12)
    
    # Courbes ROC améliorées
    plt.subplot(gs[1, :])
    if 'real_anomaly' in unsup_df.columns:
        for method, color in zip(methods, colors):
            if f'{method}_score' in unsup_df.columns:
                fpr, tpr, _ = roc_curve(unsup_df['real_anomaly'], 
                                       unsup_df[f'{method}_score'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, 
                         label=f'{method} (AUC = {roc_auc:.2f})', 
                         lw=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbes ROC des méthodes non supervisées', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualisations_avancees.png', dpi=300, bbox_inches='tight')
    
    # 3. Analyse approfondie des caractéristiques
    plt.figure(figsize=(20, 10))
    
    # Corrélation entre les caractéristiques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.subplot(121)
    mask = np.triu(np.ones_like(correlation_matrix))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                cmap='coolwarm', fmt='.2f', square=True, 
                linewidths=0.5)
    plt.title('Matrice de corrélation des caractéristiques', fontsize=12)
    
    # Importance des caractéristiques
    plt.subplot(122)
    if 'real_anomaly' in df.columns:
        feature_importance = pd.DataFrame()
        for col in numeric_cols:
            if col != 'is_anomaly':
                auc_score = roc_auc_score(df['is_anomaly'], df[col])
                feature_importance.loc[col, 'Importance'] = auc_score
        
        feature_importance = feature_importance.sort_values('Importance', 
                                                           ascending=True)
        sns.barplot(data=feature_importance, 
                    x='Importance', y=feature_importance.index, 
                    palette='viridis')
        plt.title('Importance des caractéristiques (AUC-ROC)', fontsize=12)
        plt.xlabel('Score AUC-ROC')
    
    plt.tight_layout()
    plt.savefig('analyse_caracteristiques.png', dpi=300, bbox_inches='tight')
    
    # 4. Comparaison supervisé vs non supervisé (si disponible)
    if supervised_results is not None and sup_df is not None:
        plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3)
        
        # Matrices de confusion pour chaque méthode supervisée
        methods_sup = ['Random Forest', 'SVM', 'XGBoost']
        for i, (method, col) in enumerate(zip(methods_sup, 
                                            ['RF_pred', 'SVM_pred', 'XGB_pred'])):
            plt.subplot(gs[0, i])
            cm = confusion_matrix(sup_df['real_anomaly'], sup_df[col], 
                                normalize='true')
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['Normal', 'Anomalie'],
                        yticklabels=['Normal', 'Anomalie'])
            plt.title(f'Matrice de confusion - {method}', fontsize=12)
        
        # Courbes ROC des méthodes supervisées
        plt.subplot(gs[1, :])
        for method, col, color in zip(methods_sup, 
                                     ['RF_prob', 'SVM_prob', 'XGB_prob'], 
                                     colors[:3]):
            fpr, tpr, _ = roc_curve(sup_df['real_anomaly'], sup_df[col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, 
                     label=f'{method} (AUC = {roc_auc:.2f})', 
                     lw=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbes ROC - Méthodes supervisées', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparaison_methodes_supervisees.png', 
                    dpi=300, bbox_inches='tight')
    
    print("Visualisations améliorées sauvegardées dans le répertoire courant.")



# Fonction principale
def main():
    print("=== COMPARAISON DES MODÈLES SUPERVISÉS ET NON SUPERVISÉS AVEC NEO4J ===\n")
    
    # Connexion à Neo4j
    try:
        from neo4j_setup import connect_to_neo4j
        
        # Connexion à Neo4j
        uri = "bolt://localhost:7689"
        username = "neo4j"
        password = "2004@2004"
        graph = connect_to_neo4j(uri, username, password)
        
        if graph is None:
            print("Impossible de se connecter à Neo4j. Exécution avec des données synthétiques à la place...")
            # Création des données synthétiques
            print("Création de données synthétiques pour la démonstration...")
            df = create_synthetic_data(n_samples=200, n_features=8)
        else:
            # Extraction des caractéristiques pour ML depuis Neo4j
            from neo4j_analysis import extract_features_for_ml
            df = extract_features_for_ml(graph)
            
            # Création d'un jeu de données étiquetées (si nécessaire)
            # Utiliser Isolation Forest pour étiqueter les données si pas d'étiquettes existantes
            if 'is_anomaly' not in df.columns:
                print("Pas d'étiquettes d'anomalies trouvées, création d'étiquettes synthétiques avec Isolation Forest...")
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                X = df[numeric_columns].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                y_pred = iso_forest.fit_predict(X_scaled)
                df['is_anomaly'] = np.where(y_pred == -1, 1, 0)
                
                print(f"Étiquettes générées: {df['is_anomaly'].sum()} anomalies détectées sur {len(df)} utilisateurs")
    
    except ImportError:
        print("Module Neo4j non trouvé. Exécution avec des données synthétiques à la place...")
        df = create_synthetic_data(n_samples=200, n_features=8)
    
    print(f"Données préparées: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
    
    # Analyse non supervisée
    unsupervised_results, unsup_df, X_pca = unsupervised_analysis(df)
    
    # Analyse supervisée
    supervised_results, sup_df = supervised_analysis(df)
    
    # Créer les visualisations
    create_visualizations(unsupervised_results, supervised_results, X_pca, df, unsup_df, sup_df)
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    print("1. Modèles non supervisés:")
    print(f"   - Isolation Forest: {unsup_df['IsoForest_anomaly'].sum()} anomalies détectées")
    print(f"   - One-Class SVM: {unsup_df['OCSVM_anomaly'].sum()} anomalies détectées")
    print(f"   - K-Means: {unsup_df['KMeans_anomaly'].sum()} anomalies détectées")
    print(f"   - DBSCAN: {unsup_df['DBSCAN_anomaly'].sum()} anomalies détectées")
    
    if supervised_results is not None:
        from sklearn.metrics import accuracy_score, f1_score
        
        print("\n2. Modèles supervisés:")
        rf_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        rf_f1 = f1_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        
        svm_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        svm_f1 = f1_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        
        xgb_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        xgb_f1 = f1_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        
        print(f"   - Random Forest: Précision = {rf_accuracy:.4f}, F1-Score = {rf_f1:.4f}")
        print(f"   - SVM: Précision = {svm_accuracy:.4f}, F1-Score = {svm_f1:.4f}")
        print(f"   - XGBoost: Précision = {xgb_accuracy:.4f}, F1-Score = {xgb_f1:.4f}")
    
    # Sauvegarder les visualisations principales
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Visualisation PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                c=df['is_anomaly'] if 'is_anomaly' in df.columns else 'blue',
                cmap='coolwarm', alpha=0.6)
    plt.title('Visualisation PCA des algorithmes non supervisés')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.savefig(os.path.join(output_dir, 'pca_non_supervise.png'))
    plt.close()
    
    # Performance des modèles supervisés
    if supervised_results is not None:
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        values = [
            [rf_accuracy, rf_f1, precision_score(sup_df['real_anomaly'], sup_df['RF_pred']), recall_score(sup_df['real_anomaly'], sup_df['RF_pred'])],
            [svm_accuracy, svm_f1, precision_score(sup_df['real_anomaly'], sup_df['SVM_pred']), recall_score(sup_df['real_anomaly'], sup_df['SVM_pred'])],
            [xgb_accuracy, xgb_f1, precision_score(sup_df['real_anomaly'], sup_df['XGB_pred']), recall_score(sup_df['real_anomaly'], sup_df['XGB_pred'])]
        ]
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.bar(x - width, values[0], width, label='Random Forest')
        plt.bar(x, values[1], width, label='SVM')
        plt.bar(x + width, values[2], width, label='XGBoost')
        
        plt.xlabel('Métriques')
        plt.ylabel('Score')
        plt.title('Performance des modèles supervisés')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_supervise.png'))
        plt.close()
    
    print("\nCes résultats ont été enregistrés sous forme de visualisations dans le répertoire courant.")
    
    # Afficher les visualisations si les fichiers existent
    for viz_file in ['pca_non_supervise.png', 'performance_supervise.png']:
        viz_path = os.path.join(output_dir, viz_file)
        if os.path.exists(viz_path):
            plt.figure(figsize=(10, 6))
            plt.imshow(plt.imread(viz_path))
            plt.axis('off')
            plt.title(viz_file.replace('.png', '').replace('_', ' ').title())
            plt.show()
        else:
            print(f"Attention: {viz_file} n'a pas été trouvé dans le répertoire de sortie.")

    
    # Sauvegarder les résultats dans des fichiers CSV
    unsup_df.to_csv('resultats_non_supervise.csv', index=False)
    if sup_df is not None:
        sup_df.to_csv('resultats_supervise.csv', index=False)
        
    # Créer un rapport des performances
    if supervised_results is not None:
        report = pd.DataFrame({
            'Modèle': ['Isolation Forest', 'One-Class SVM', 'K-Means', 'DBSCAN', 'Random Forest', 'SVM', 'XGBoost'],
            'Type': ['Non supervisé', 'Non supervisé', 'Non supervisé', 'Non supervisé', 'Supervisé', 'Supervisé', 'Supervisé'],
            'Anomalies détectées': [
                unsup_df['IsoForest_anomaly'].sum(),
                unsup_df['OCSVM_anomaly'].sum(),
                unsup_df['KMeans_anomaly'].sum(),
                unsup_df['DBSCAN_anomaly'].sum(),
                sup_df['RF_pred'].sum(),
                sup_df['SVM_pred'].sum(),
                sup_df['XGB_pred'].sum()
            ]
        })
        
        # Ajouter les métriques pour les modèles supervisés
        if 'real_anomaly' in sup_df.columns:
            report['Exactitude'] = [
                None, None, None, None,
                rf_accuracy,
                svm_accuracy,
                xgb_accuracy
            ]
            report['F1-Score'] = [
                None, None, None, None,
                rf_f1,
                svm_f1,
                xgb_f1
            ]
        
        report.to_csv('rapport_performances.csv', index=False)
        print("\nRapport des performances sauvegardé dans 'rapport_performances.csv'")


if __name__ == "__main__":
    print("=== COMPARAISON DES MODÈLES SUPERVISÉS ET NON SUPERVISÉS AVEC NEO4J ===\n")
    
    # Connexion à Neo4j
    try:
        from neo4j_setup import connect_to_neo4j
        
        # Connexion à Neo4j
        uri = "bolt://localhost:7689"
        username = "neo4j"
        password = "2004@2004"
        graph = connect_to_neo4j(uri, username, password)
        
        if graph is None:
            print("Impossible de se connecter à Neo4j. Exécution avec des données synthétiques à la place...")
            # Création des données synthétiques
            print("Création de données synthétiques pour la démonstration...")
            df = create_synthetic_data(n_samples=200, n_features=8)
        else:
            # Extraction des caractéristiques pour ML depuis Neo4j
            from neo4j_analysis import extract_features_for_ml
            df = extract_features_for_ml(graph)
            
            # Création d'un jeu de données étiquetées (si nécessaire)
            # Utiliser Isolation Forest pour étiqueter les données si pas d'étiquettes existantes
            if 'is_anomaly' not in df.columns:
                print("Pas d'étiquettes d'anomalies trouvées, création d'étiquettes synthétiques avec Isolation Forest...")
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                X = df[numeric_columns].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                y_pred = iso_forest.fit_predict(X_scaled)
                df['is_anomaly'] = np.where(y_pred == -1, 1, 0)
                
                print(f"Étiquettes générées: {df['is_anomaly'].sum()} anomalies détectées sur {len(df)} utilisateurs")
    
    except ImportError:
        print("Module Neo4j non trouvé. Exécution avec des données synthétiques à la place...")
        df = create_synthetic_data(n_samples=200, n_features=8)
    
    print(f"Données préparées: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
    
    # Analyse non supervisée
    unsupervised_results, unsup_df, X_pca = unsupervised_analysis(df)
    
    # Analyse supervisée
    supervised_results, sup_df = supervised_analysis(df)
    
    # Créer les visualisations
    create_visualizations(unsupervised_results, supervised_results, X_pca, df, unsup_df, sup_df)
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    print("1. Modèles non supervisés:")
    print(f"   - Isolation Forest: {unsup_df['IsoForest_anomaly'].sum()} anomalies détectées")
    print(f"   - One-Class SVM: {unsup_df['OCSVM_anomaly'].sum()} anomalies détectées")
    print(f"   - K-Means: {unsup_df['KMeans_anomaly'].sum()} anomalies détectées")
    print(f"   - DBSCAN: {unsup_df['DBSCAN_anomaly'].sum()} anomalies détectées")
    
    if supervised_results is not None:
        from sklearn.metrics import accuracy_score, f1_score
        
        print("\n2. Modèles supervisés:")
        rf_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        rf_f1 = f1_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        
        svm_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        svm_f1 = f1_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        
        xgb_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        xgb_f1 = f1_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        
        print(f"   - Random Forest: Précision = {rf_accuracy:.4f}, F1-Score = {rf_f1:.4f}")
        print(f"   - SVM: Précision = {svm_accuracy:.4f}, F1-Score = {svm_f1:.4f}")
        print(f"   - XGBoost: Précision = {xgb_accuracy:.4f}, F1-Score = {xgb_f1:.4f}")
    
    # Sauvegarder les visualisations principales
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Visualisation PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                c=df['is_anomaly'] if 'is_anomaly' in df.columns else 'blue',
                cmap='coolwarm', alpha=0.6)
    plt.title('Visualisation PCA des algorithmes non supervisés')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.savefig(os.path.join(output_dir, 'pca_non_supervise.png'))
    plt.close()
    
    # Performance des modèles supervisés
    if supervised_results is not None:
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        values = [
            [rf_accuracy, rf_f1, precision_score(sup_df['real_anomaly'], sup_df['RF_pred']), recall_score(sup_df['real_anomaly'], sup_df['RF_pred'])],
            [svm_accuracy, svm_f1, precision_score(sup_df['real_anomaly'], sup_df['SVM_pred']), recall_score(sup_df['real_anomaly'], sup_df['SVM_pred'])],
            [xgb_accuracy, xgb_f1, precision_score(sup_df['real_anomaly'], sup_df['XGB_pred']), recall_score(sup_df['real_anomaly'], sup_df['XGB_pred'])]
        ]
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.bar(x - width, values[0], width, label='Random Forest')
        plt.bar(x, values[1], width, label='SVM')
        plt.bar(x + width, values[2], width, label='XGBoost')
        
        plt.xlabel('Métriques')
        plt.ylabel('Score')
        plt.title('Performance des modèles supervisés')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_supervise.png'))
        plt.close()
    
    print("\nCes résultats ont été enregistrés sous forme de visualisations dans le répertoire courant.")
    
    # Afficher les visualisations si les fichiers existent
    for viz_file in ['pca_non_supervise.png', 'performance_supervise.png']:
        viz_path = os.path.join(output_dir, viz_file)
        if os.path.exists(viz_path):
            plt.figure(figsize=(10, 6))
            plt.imshow(plt.imread(viz_path))
            plt.axis('off')
            plt.title(viz_file.replace('.png', '').replace('_', ' ').title())
            plt.show()
        else:
            print(f"Attention: {viz_file} n'a pas été trouvé dans le répertoire de sortie.")

    
    # Sauvegarder les résultats dans des fichiers CSV
    unsup_df.to_csv('resultats_non_supervise.csv', index=False)
    if sup_df is not None:
        sup_df.to_csv('resultats_supervise.csv', index=False)
        
    # Créer un rapport des performances
    if supervised_results is not None:
        report = pd.DataFrame({
            'Modèle': ['Isolation Forest', 'One-Class SVM', 'K-Means', 'DBSCAN', 'Random Forest', 'SVM', 'XGBoost'],
            'Type': ['Non supervisé', 'Non supervisé', 'Non supervisé', 'Non supervisé', 'Supervisé', 'Supervisé', 'Supervisé'],
            'Anomalies détectées': [
                unsup_df['IsoForest_anomaly'].sum(),
                unsup_df['OCSVM_anomaly'].sum(),
                unsup_df['KMeans_anomaly'].sum(),
                unsup_df['DBSCAN_anomaly'].sum(),
                sup_df['RF_pred'].sum(),
                sup_df['SVM_pred'].sum(),
                sup_df['XGB_pred'].sum()
            ]
        })
        
        # Ajouter les métriques pour les modèles supervisés
        if 'real_anomaly' in sup_df.columns:
            report['Exactitude'] = [
                None, None, None, None,
                rf_accuracy,
                svm_accuracy,
                xgb_accuracy
            ]
            report['F1-Score'] = [
                None, None, None, None,
                rf_f1,
                svm_f1,
                xgb_f1
            ]
        
        report.to_csv('rapport_performances.csv', index=False)
        print("\nRapport des performances sauvegardé dans 'rapport_performances.csv'")


if __name__ == "__main__":
    print("=== COMPARAISON DES MODÈLES SUPERVISÉS ET NON SUPERVISÉS AVEC NEO4J ===\n")
    
    # Connexion à Neo4j
    try:
        from neo4j_setup import connect_to_neo4j
        
        # Connexion à Neo4j
        uri = "bolt://localhost:7689"
        username = "neo4j"
        password = "2004@2004"
        graph = connect_to_neo4j(uri, username, password)
        
        if graph is None:
            print("Impossible de se connecter à Neo4j. Exécution avec des données synthétiques à la place...")
            # Création des données synthétiques
            print("Création de données synthétiques pour la démonstration...")
            df = create_synthetic_data(n_samples=200, n_features=8)
        else:
            # Extraction des caractéristiques pour ML depuis Neo4j
            from neo4j_analysis import extract_features_for_ml
            df = extract_features_for_ml(graph)
            
            # Création d'un jeu de données étiquetées (si nécessaire)
            # Utiliser Isolation Forest pour étiqueter les données si pas d'étiquettes existantes
            if 'is_anomaly' not in df.columns:
                print("Pas d'étiquettes d'anomalies trouvées, création d'étiquettes synthétiques avec Isolation Forest...")
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                X = df[numeric_columns].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                y_pred = iso_forest.fit_predict(X_scaled)
                df['is_anomaly'] = np.where(y_pred == -1, 1, 0)
                
                print(f"Étiquettes générées: {df['is_anomaly'].sum()} anomalies détectées sur {len(df)} utilisateurs")
    
    except ImportError:
        print("Module Neo4j non trouvé. Exécution avec des données synthétiques à la place...")
        df = create_synthetic_data(n_samples=200, n_features=8)
    
    print(f"Données préparées: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
    
    # Analyse non supervisée
    unsupervised_results, unsup_df, X_pca = unsupervised_analysis(df)
    
    # Analyse supervisée
    supervised_results, sup_df = supervised_analysis(df)
    
    # Créer les visualisations
    create_visualizations(unsupervised_results, supervised_results, X_pca, df, unsup_df, sup_df)
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    print("1. Modèles non supervisés:")
    print(f"   - Isolation Forest: {unsup_df['IsoForest_anomaly'].sum()} anomalies détectées")
    print(f"   - One-Class SVM: {unsup_df['OCSVM_anomaly'].sum()} anomalies détectées")
    print(f"   - K-Means: {unsup_df['KMeans_anomaly'].sum()} anomalies détectées")
    print(f"   - DBSCAN: {unsup_df['DBSCAN_anomaly'].sum()} anomalies détectées")
    
    if supervised_results is not None:
        from sklearn.metrics import accuracy_score, f1_score
        
        print("\n2. Modèles supervisés:")
        rf_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        rf_f1 = f1_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        
        svm_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        svm_f1 = f1_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        
        xgb_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        xgb_f1 = f1_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        
        print(f"   - Random Forest: Précision = {rf_accuracy:.4f}, F1-Score = {rf_f1:.4f}")
        print(f"   - SVM: Précision = {svm_accuracy:.4f}, F1-Score = {svm_f1:.4f}")
        print(f"   - XGBoost: Précision = {xgb_accuracy:.4f}, F1-Score = {xgb_f1:.4f}")
    
    print("\nCes résultats ont été enregistrés sous forme de visualisations dans le répertoire courant.")
    
    # Afficher les visualisations principales
    plt.figure(figsize=(10, 6))
    plt.imshow(plt.imread('pca_non_supervise.png'))
    plt.axis('off')
    plt.title('Visualisation PCA des algorithmes non supervisés')
    plt.show()    
    plt.figure(figsize=(10, 6))
    plt.imshow(plt.imread('performance_supervise.png'))
    plt.axis('off')
    plt.title('Performance des modèles supervisés')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(plt.imread('comparaison_supervise_non_supervise.png'))
    plt.axis('off')
    plt.title('Comparaison des approches supervisées et non supervisées')
    plt.show()
    
    # Sauvegarder les résultats dans des fichiers CSV
    unsup_df.to_csv('resultats_non_supervise.csv', index=False)
    if sup_df is not None:
        sup_df.to_csv('resultats_supervise.csv', index=False)
        
    # Créer un rapport des performances
    if supervised_results is not None:
        report = pd.DataFrame({
            'Modèle': ['Isolation Forest', 'One-Class SVM', 'K-Means', 'DBSCAN', 'Random Forest', 'SVM', 'XGBoost'],
            'Type': ['Non supervisé', 'Non supervisé', 'Non supervisé', 'Non supervisé', 'Supervisé', 'Supervisé', 'Supervisé'],
            'Anomalies détectées': [
                unsup_df['IsoForest_anomaly'].sum(),
                unsup_df['OCSVM_anomaly'].sum(),
                unsup_df['KMeans_anomaly'].sum(),
                unsup_df['DBSCAN_anomaly'].sum(),
                sup_df['RF_pred'].sum(),
                sup_df['SVM_pred'].sum(),
                sup_df['XGB_pred'].sum()
            ]
        })
        
        # Ajouter les métriques pour les modèles supervisés
        if 'real_anomaly' in sup_df.columns:
            report['Exactitude'] = [
                None, None, None, None,
                rf_accuracy,
                svm_accuracy,
                xgb_accuracy
            ]
            report['F1-Score'] = [
                None, None, None, None,
                rf_f1,
                svm_f1,
                xgb_f1
            ]
        
        report.to_csv('rapport_performances.csv', index=False)
        print("\nRapport des performances sauvegardé dans 'rapport_performances.csv'")


if __name__ == "__main__":
    print("=== COMPARAISON DES MODÈLES SUPERVISÉS ET NON SUPERVISÉS AVEC NEO4J ===\n")
    
    # Connexion à Neo4j
    try:
        from neo4j_setup import connect_to_neo4j
        
        # Connexion à Neo4j
        uri = "bolt://localhost:7689"
        username = "neo4j"
        password = "2004@2004"
        graph = connect_to_neo4j(uri, username, password)
        
        if graph is None:
            print("Impossible de se connecter à Neo4j. Exécution avec des données synthétiques à la place...")
            # Création des données synthétiques
            print("Création de données synthétiques pour la démonstration...")
            df = create_synthetic_data(n_samples=200, n_features=8)
        else:
            # Extraction des caractéristiques pour ML depuis Neo4j
            from neo4j_analysis import extract_features_for_ml
            df = extract_features_for_ml(graph)
            
            # Création d'un jeu de données étiquetées (si nécessaire)
            # Utiliser Isolation Forest pour étiqueter les données si pas d'étiquettes existantes
            if 'is_anomaly' not in df.columns:
                print("Pas d'étiquettes d'anomalies trouvées, création d'étiquettes synthétiques avec Isolation Forest...")
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                X = df[numeric_columns].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                y_pred = iso_forest.fit_predict(X_scaled)
                df['is_anomaly'] = np.where(y_pred == -1, 1, 0)
                
                print(f"Étiquettes générées: {df['is_anomaly'].sum()} anomalies détectées sur {len(df)} utilisateurs")
    
    except ImportError:
        print("Module Neo4j non trouvé. Exécution avec des données synthétiques à la place...")
        df = create_synthetic_data(n_samples=200, n_features=8)
    
    print(f"Données préparées: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
    
    # Analyse non supervisée
    unsupervised_results, unsup_df, X_pca = unsupervised_analysis(df)
    
    # Analyse supervisée
    supervised_results, sup_df = supervised_analysis(df)
    
    # Créer les visualisations
    create_visualizations(unsupervised_results, supervised_results, X_pca, df, unsup_df, sup_df)
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    print("1. Modèles non supervisés:")
    print(f"   - Isolation Forest: {unsup_df['IsoForest_anomaly'].sum()} anomalies détectées")
    print(f"   - One-Class SVM: {unsup_df['OCSVM_anomaly'].sum()} anomalies détectées")
    print(f"   - K-Means: {unsup_df['KMeans_anomaly'].sum()} anomalies détectées")
    print(f"   - DBSCAN: {unsup_df['DBSCAN_anomaly'].sum()} anomalies détectées")
    
    if supervised_results is not None:
        from sklearn.metrics import accuracy_score, f1_score
        
        print("\n2. Modèles supervisés:")
        rf_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        rf_f1 = f1_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        
        svm_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        svm_f1 = f1_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        
        xgb_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        xgb_f1 = f1_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        
        print(f"   - Random Forest: Précision = {rf_accuracy:.4f}, F1-Score = {rf_f1:.4f}")
        print(f"   - SVM: Précision = {svm_accuracy:.4f}, F1-Score = {svm_f1:.4f}")
        print(f"   - XGBoost: Précision = {xgb_accuracy:.4f}, F1-Score = {xgb_f1:.4f}")
    
    # Sauvegarder les visualisations principales
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Visualisation PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                c=df['is_anomaly'] if 'is_anomaly' in df.columns else 'blue',
                cmap='coolwarm', alpha=0.6)
    plt.title('Visualisation PCA des algorithmes non supervisés')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.savefig(os.path.join(output_dir, 'pca_non_supervise.png'))
    plt.close()
    
    # Performance des modèles supervisés
    if supervised_results is not None:
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        values = [
            [rf_accuracy, rf_f1, precision_score(sup_df['real_anomaly'], sup_df['RF_pred']), recall_score(sup_df['real_anomaly'], sup_df['RF_pred'])],
            [svm_accuracy, svm_f1, precision_score(sup_df['real_anomaly'], sup_df['SVM_pred']), recall_score(sup_df['real_anomaly'], sup_df['SVM_pred'])],
            [xgb_accuracy, xgb_f1, precision_score(sup_df['real_anomaly'], sup_df['XGB_pred']), recall_score(sup_df['real_anomaly'], sup_df['XGB_pred'])]
        ]
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.bar(x - width, values[0], width, label='Random Forest')
        plt.bar(x, values[1], width, label='SVM')
        plt.bar(x + width, values[2], width, label='XGBoost')
        
        plt.xlabel('Métriques')
        plt.ylabel('Score')
        plt.title('Performance des modèles supervisés')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_supervise.png'))
        plt.close()
    
    print("\nCes résultats ont été enregistrés sous forme de visualisations dans le répertoire courant.")
    
    # Afficher les visualisations si les fichiers existent
    for viz_file in ['pca_non_supervise.png', 'performance_supervise.png']:
        viz_path = os.path.join(output_dir, viz_file)
        if os.path.exists(viz_path):
            plt.figure(figsize=(10, 6))
            plt.imshow(plt.imread(viz_path))
            plt.axis('off')
            plt.title(viz_file.replace('.png', '').replace('_', ' ').title())
            plt.show()
        else:
            print(f"Attention: {viz_file} n'a pas été trouvé dans le répertoire de sortie.")

    
    # Sauvegarder les résultats dans des fichiers CSV
    unsup_df.to_csv('resultats_non_supervise.csv', index=False)
    if sup_df is not None:
        sup_df.to_csv('resultats_supervise.csv', index=False)
        
    # Créer un rapport des performances
    if supervised_results is not None:
        report = pd.DataFrame({
            'Modèle': ['Isolation Forest', 'One-Class SVM', 'K-Means', 'DBSCAN', 'Random Forest', 'SVM', 'XGBoost'],
            'Type': ['Non supervisé', 'Non supervisé', 'Non supervisé', 'Non supervisé', 'Supervisé', 'Supervisé', 'Supervisé'],
            'Anomalies détectées': [
                unsup_df['IsoForest_anomaly'].sum(),
                unsup_df['OCSVM_anomaly'].sum(),
                unsup_df['KMeans_anomaly'].sum(),
                unsup_df['DBSCAN_anomaly'].sum(),
                sup_df['RF_pred'].sum(),
                sup_df['SVM_pred'].sum(),
                sup_df['XGB_pred'].sum()
            ]
        })
        
        # Ajouter les métriques pour les modèles supervisés
        if 'real_anomaly' in sup_df.columns:
            report['Exactitude'] = [
                None, None, None, None,
                rf_accuracy,
                svm_accuracy,
                xgb_accuracy
            ]
            report['F1-Score'] = [
                None, None, None, None,
                rf_f1,
                svm_f1,
                xgb_f1
            ]
        
        report.to_csv('rapport_performances.csv', index=False)
        print("\nRapport des performances sauvegardé dans 'rapport_performances.csv'")


if __name__ == "__main__":
    print("=== COMPARAISON DES MODÈLES SUPERVISÉS ET NON SUPERVISÉS AVEC NEO4J ===\n")
    
    # Connexion à Neo4j
    try:
        from neo4j_setup import connect_to_neo4j
        
        # Connexion à Neo4j
        uri = "bolt://localhost:7689"
        username = "neo4j"
        password = "2004@2004"
        graph = connect_to_neo4j(uri, username, password)
        
        if graph is None:
            print("Impossible de se connecter à Neo4j. Exécution avec des données synthétiques à la place...")
            # Création des données synthétiques
            print("Création de données synthétiques pour la démonstration...")
            df = create_synthetic_data(n_samples=200, n_features=8)
        else:
            # Extraction des caractéristiques pour ML depuis Neo4j
            from neo4j_analysis import extract_features_for_ml
            df = extract_features_for_ml(graph)
            
            # Création d'un jeu de données étiquetées (si nécessaire)
            # Utiliser Isolation Forest pour étiqueter les données si pas d'étiquettes existantes
            if 'is_anomaly' not in df.columns:
                print("Pas d'étiquettes d'anomalies trouvées, création d'étiquettes synthétiques avec Isolation Forest...")
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                X = df[numeric_columns].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                y_pred = iso_forest.fit_predict(X_scaled)
                df['is_anomaly'] = np.where(y_pred == -1, 1, 0)
                
                print(f"Étiquettes générées: {df['is_anomaly'].sum()} anomalies détectées sur {len(df)} utilisateurs")
    
    except ImportError:
        print("Module Neo4j non trouvé. Exécution avec des données synthétiques à la place...")
        df = create_synthetic_data(n_samples=200, n_features=8)
    
    print(f"Données préparées: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
    
    # Analyse non supervisée
    unsupervised_results, unsup_df, X_pca = unsupervised_analysis(df)
    
    # Analyse supervisée
    supervised_results, sup_df = supervised_analysis(df)
    
    # Créer les visualisations
    create_visualizations(unsupervised_results, supervised_results, X_pca, df, unsup_df, sup_df)
    
    # Résumé des résultats
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    print("1. Modèles non supervisés:")
    print(f"   - Isolation Forest: {unsup_df['IsoForest_anomaly'].sum()} anomalies détectées")
    print(f"   - One-Class SVM: {unsup_df['OCSVM_anomaly'].sum()} anomalies détectées")
    print(f"   - K-Means: {unsup_df['KMeans_anomaly'].sum()} anomalies détectées")
    print(f"   - DBSCAN: {unsup_df['DBSCAN_anomaly'].sum()} anomalies détectées")
    
    if supervised_results is not None:
        from sklearn.metrics import accuracy_score, f1_score
        
        print("\n2. Modèles supervisés:")
        rf_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        rf_f1 = f1_score(sup_df['real_anomaly'], sup_df['RF_pred'])
        
        svm_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        svm_f1 = f1_score(sup_df['real_anomaly'], sup_df['SVM_pred'])
        
        xgb_accuracy = accuracy_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        xgb_f1 = f1_score(sup_df['real_anomaly'], sup_df['XGB_pred'])
        
        print(f"   - Random Forest: Précision = {rf_accuracy:.4f}, F1-Score = {rf_f1:.4f}")
        print(f"   - SVM: Précision = {svm_accuracy:.4f}, F1-Score = {svm_f1:.4f}")
        print(f"   - XGBoost: Précision = {xgb_accuracy:.4f}, F1-Score = {xgb_f1:.4f}")
    
    # Sauvegarder les visualisations principales
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Visualisation PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                c=df['is_anomaly'] if 'is_anomaly' in df.columns else 'blue',
                cmap='coolwarm', alpha=0.6)
    plt.title('Visualisation PCA des algorithmes non supervisés')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.savefig(os.path.join(output_dir, 'pca_non_supervise.png'))
    plt.close()
    
    # Performance des modèles supervisés
    if supervised_results is not None:
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        values = [
            [rf_accuracy, rf_f1, precision_score(sup_df['real_anomaly'], sup_df['RF_pred']), recall_score(sup_df['real_anomaly'], sup_df['RF_pred'])],
            [svm_accuracy, svm_f1, precision_score(sup_df['real_anomaly'], sup_df['SVM_pred']), recall_score(sup_df['real_anomaly'], sup_df['SVM_pred'])],
            [xgb_accuracy, xgb_f1, precision_score(sup_df['real_anomaly'], sup_df['XGB_pred']), recall_score(sup_df['real_anomaly'], sup_df['XGB_pred'])]
        ]
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.bar(x - width, values[0], width, label='Random Forest')
        plt.bar(x, values[1], width, label='SVM')
        plt.bar(x + width, values[2], width, label='XGBoost')
        
        plt.xlabel('Métriques')
        plt.ylabel('Score')
        plt.title('Performance des modèles supervisés')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_supervise.png'))
        plt.close()
    
    print("\nCes résultats ont été enregistrés sous forme de visualisations dans le répertoire courant.")
    
    # Afficher les visualisations si les fichiers existent
    for viz_file in ['pca_non_supervise.png', 'performance_supervise.png']:
        viz_path = os.path.join(output_dir, viz_file)
        if os.path.exists(viz_path):
            plt.figure(figsize=(10, 6))
            plt.imshow(plt.imread(viz_path))
            plt.axis('off')
            plt.title(viz_file.replace('.png', '').replace('_', ' ').title())
            plt.show()
        else:
            print(f"Attention: {viz_file} n'a pas été trouvé dans le répertoire de sortie.")

    
    # Sauvegarder les résultats dans des fichiers CSV
    unsup_df.to_csv('resultats_non_supervise.csv', index=False)
    if sup_df is not None:
        sup_df.to_csv('resultats_supervise.csv', index=False)
        
    # Créer un rapport des performances
    if supervised_results is not None:
        report = pd.DataFrame({
            'Modèle': ['Isolation Forest', 'One-Class SVM', 'K-Means', 'DBSCAN', 'Random Forest', 'SVM', 'XGBoost'],
            'Type': ['Non supervisé', 'Non supervisé', 'Non supervisé', 'Non supervisé', 'Supervisé', 'Supervisé', 'Supervisé'],
            'Anomalies détectées': [
                unsup_df['IsoForest_anomaly'].sum(),
                unsup_df['OCSVM_anomaly'].sum(),
                unsup_df['KMeans_anomaly'].sum(),
                unsup_df['DBSCAN_anomaly'].sum(),
                sup_df['RF_pred'].sum(),
                sup_df['SVM_pred'].sum(),
                sup_df['XGB_pred'].sum()
            ]
        })
        
        # Ajouter les métriques pour les modèles supervisés
        if 'real_anomaly' in sup_df.columns:
            report['Exactitude'] = [
                None, None, None, None,
                rf_accuracy,
                svm_accuracy,
                xgb_accuracy
            ]
            report['F1-Score'] = [
                None, None, None, None,
                rf_f1,
                svm_f1,
                xgb_f1
            ]
        
        report.to_csv('rapport_performances.csv', index=False)
        print("\nRapport des performances sauvegardé dans 'rapport_performances.csv'")
        
        
        
        # Création de l'autoencoder standard pour la détection d'anomalies
def create_autoencoder(input_dim, encoding_dim=3):
    # Définition de l'encodeur
    input_layer = Input(shape=(input_dim,))
    
    # Architecture de l'encodeur - diminution progressive de la dimension
    encoder = Dense(int(input_dim * 0.75), activation='relu')(input_layer)
    encoder = Dense(int(input_dim * 0.5), activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Architecture du décodeur - augmentation progressive de la dimension
    decoder = Dense(int(input_dim * 0.5), activation='relu')(encoder)
    decoder = Dense(int(input_dim * 0.75), activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Création du modèle complet autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Compilation du modèle avec une fonction de perte adaptée à la reconstruction
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Modèle de l'encodeur seul (pour obtenir la représentation comprimée)
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    
    return autoencoder, encoder_model


# Fonction pour entraîner l'autoencoder standard et détecter les anomalies
def autoencoder_anomaly_detection(X, contamination=0.1):
    # Obtenir les dimensions de l'entrée
    input_dim = X.shape[1]
    
    # Création des modèles
    autoencoder, encoder = create_autoencoder(input_dim)
    
    # Division en ensembles d'entraînement et de validation (pour prévenir le surapprentissage)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    # Early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Entraînement de l'autoencoder
    history = autoencoder.fit(
        X_train, X_train,  # L'autoencoder apprend à reproduire l'entrée
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Reconstruction des données d'entrée
    X_pred = autoencoder.predict(X)
    
    # Calcul de l'erreur de reconstruction (MSE)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    
    # Détermination du seuil pour les anomalies basé sur la contamination
    threshold = np.percentile(mse, 100 * (1 - contamination))
    
    # Identification des anomalies (1 pour anomalie, 0 pour normal)
    predictions = np.zeros(len(mse))
    predictions[mse > threshold] = 1
    
    return predictions, mse, autoencoder, encoder, history


# Création de l'autoencoder amélioré pour la détection d'anomalies
def create_improved_autoencoder(input_dim, encoding_dim=3, dropout_rate=0.2):
    # Définition de l'encodeur avec une architecture plus robuste
    input_layer = Input(shape=(input_dim,))
    
    # Architecture de l'encodeur avec dropout pour éviter le surapprentissage
    encoder = Dense(int(input_dim * 0.8), activation='relu')(input_layer)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(int(input_dim * 0.5), activation='relu')(encoder)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Couche de représentation latente (bottleneck)
    bottleneck = encoder
    
    # Architecture du décodeur avec dropout
    decoder = Dense(int(input_dim * 0.5), activation='relu')(bottleneck)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(int(input_dim * 0.8), activation='relu')(decoder)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Création du modèle complet autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Compilation du modèle avec une fonction de perte adaptée à la reconstruction
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Modèle de l'encodeur seul (pour obtenir la représentation comprimée)
    encoder_model = Model(inputs=input_layer, outputs=bottleneck)
    
    return
