import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_anomalies(results):
    """
    Créer des visualisations pour les résultats d'anomalies détectées
    """
    print("Création des visualisations pour les anomalies...")
    
    # Configurer le style de Seaborn
    sns.set(style="whitegrid")
    
    # 1. Visualisation des activités après heures
    if 'after_hours' in results and results['after_hours']:
        df_after_hours = pd.DataFrame(results['after_hours'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='user', y='after_hours_activity_count', data=df_after_hours, color='darkred')
        plt.title('Activités après les heures normales par utilisateur', fontsize=15)
        plt.xlabel('Utilisateur', fontsize=12)
        plt.ylabel('Nombre d\'activités', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('after_hours_activities.png')
        plt.close()
    
    # 2. Visualisation de l'entropie des activités
    if 'entropy' in results and results['entropy']:
        df_entropy = pd.DataFrame(results['entropy'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='user', y='entropy', data=df_entropy, color='darkblue')
        plt.title('Entropie des activités par utilisateur', fontsize=15)
        plt.xlabel('Utilisateur', fontsize=12)
        plt.ylabel('Entropie', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('activity_entropy.png')
        plt.close()
        
        # Créer un graphique combinant entropie et nombre d'activités
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        color = 'tab:blue'
        ax1.set_xlabel('Utilisateur', fontsize=12)
        ax1.set_ylabel('Entropie', fontsize=12, color=color)
        ax1.bar(df_entropy['user'], df_entropy['entropy'], color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.xticks(rotation=45, ha='right')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Nombre d\'activités', fontsize=12, color=color)
        ax2.plot(df_entropy['user'], df_entropy['activity_count'], color=color, marker='o', linestyle='-', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Entropie vs Nombre d\'activités par utilisateur', fontsize=15)
        fig.tight_layout()
        plt.savefig('entropy_vs_activity.png')
        plt.close()

    # 3. Visualisation des utilisateurs à haute activité
    if 'high_activity' in results and results['high_activity']:
        df_high_activity = pd.DataFrame(results['high_activity'])
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='user', y='activity_count', data=df_high_activity, palette='viridis')
        plt.title('Utilisateurs avec activité élevée', fontsize=15)
        plt.xlabel('Utilisateur', fontsize=12)
        plt.ylabel('Nombre d\'activités', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.0f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', 
                      xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig('high_activity_users.png')
        plt.close()
    
    # 4. Visualisation de la centralité
    if 'centrality' in results and results['centrality']:
        df_centrality = pd.DataFrame(results['centrality'])
        
        # Utiliser Matplotlib pour une visualisation en PNG
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(df_centrality)), df_centrality['total_degree'], 
                   s=df_centrality['total_degree']*10, c=df_centrality['total_degree'], 
                   cmap='viridis', alpha=0.7)
        plt.xticks(range(len(df_centrality)), df_centrality['user'], rotation=45, ha='right')
        plt.colorbar(label='Degré total')
        plt.title('Centralité de degré par utilisateur', fontsize=15)
        plt.xlabel('Utilisateur', fontsize=12)
        plt.ylabel('Degré total', fontsize=12)
        plt.tight_layout()
        plt.savefig('user_centrality.png')
        plt.close()

def visualize_features(features_df):
    """
    Créer des visualisations pour les caractéristiques extraites pour ML
    """
    print("Création des visualisations pour les caractéristiques...")
    
    # 1. Heatmap des corrélations entre caractéristiques
    plt.figure(figsize=(12, 10))
    correlation_matrix = features_df.drop('user', axis=1).corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                mask=mask, square=True, linewidths=.5)
    plt.title('Corrélation entre les caractéristiques', fontsize=15)
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()
    
    # 2. Scatter plots pour visualiser les relations entre différentes caractéristiques
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='total_activities', y='activity_entropy', 
                    size='unique_resources', hue='after_hours_ratio',
                    data=features_df, palette='viridis', sizes=(20, 200))
    plt.title('Relation entre activités totales et entropie', fontsize=15)
    plt.xlabel('Nombre total d\'activités', fontsize=12)
    plt.ylabel('Entropie des activités', fontsize=12)
    plt.legend(title='Ratio après heures', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('activity_entropy_scatter.png')
    plt.close()
    
    # 3. Distribution des caractéristiques principales avec matplotlib
    features_to_plot = ['total_activities', 'after_hours_activities', 'weekend_activities', 
                        'unique_systems', 'unique_resources', 'activity_entropy']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Distribution des caractéristiques principales", fontsize=16)
    
    for i, feature in enumerate(features_to_plot):
        row = i // 2
        col = i % 2
        
        axes[row, col].hist(features_df[feature], bins=20)
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Fréquence')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # 4. Box plots pour détecter les valeurs aberrantes
    plt.figure(figsize=(15, 10))
    features_df_melt = pd.melt(features_df.drop('user', axis=1))
    sns.boxplot(x='variable', y='value', data=features_df_melt)
    plt.title('Distribution des caractéristiques et valeurs aberrantes', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_boxplots.png')
    plt.close()

def visualize_resource_entropy(resource_entropy_data):
    """
    Créer des visualisations pour l'entropie des ressources
    """
    print("Création des visualisations pour l'entropie des ressources...")
    
    # Convertir en DataFrame pour faciliter les visualisations
    df = pd.DataFrame(resource_entropy_data)
    
    # 1. Graphique combiné entropie et nombre de ressources uniques
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color = 'tab:purple'
    ax1.set_xlabel('Utilisateur', fontsize=12)
    ax1.set_ylabel('Entropie des ressources', fontsize=12, color=color)
    ax1.bar(df['user'].head(10), df['resource_entropy'].head(10), color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Ressources uniques', fontsize=12, color=color)
    ax2.scatter(df['user'].head(10), df['unique_resources'].head(10), color=color, s=100)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Entropie des ressources vs Nombre de ressources uniques', fontsize=15)
    fig.tight_layout()
    plt.savefig('resource_entropy_vs_count.png')
    plt.close()
    
    # 2. Visualisation avec Matplotlib en format PNG
    plt.figure(figsize=(12, 10))
    plt.scatter(df['resource_entropy'].head(20), df['unique_resources'].head(20), 
               s=df['unique_resources'].head(20)*5, c=df['resource_entropy'].head(20),
               cmap='Purples', alpha=0.7)
    
    # Ajouter des étiquettes pour chaque point
    for i, txt in enumerate(df['user'].head(20)):
        plt.annotate(txt, (df['resource_entropy'].iloc[i], df['unique_resources'].iloc[i]),
                    fontsize=9, ha='center', va='bottom')
    
    plt.colorbar(label='Entropie des ressources')
    plt.title('Relation entre entropie des ressources et nombre de ressources uniques', fontsize=15)
    plt.xlabel('Entropie des ressources', fontsize=12)
    plt.ylabel('Ressources uniques', fontsize=12)
    plt.tight_layout()
    plt.savefig('resource_entropy_scatter.png')
    plt.close()

# Fonction pour créer un dashboard récapitulatif avec Matplotlib
def create_dashboard(results, features_df, resource_entropy_data):
    """
    Créer un tableau de bord récapitulatif en PNG avec Matplotlib
    """
    print("Création du tableau de bord récapitulatif...")
    
    # Convertir les données en DataFrames
    df_after_hours = pd.DataFrame(results['after_hours']) if 'after_hours' in results else pd.DataFrame()
    df_high_activity = pd.DataFrame(results['high_activity']) if 'high_activity' in results else pd.DataFrame()
    df_centrality = pd.DataFrame(results['centrality']) if 'centrality' in results else pd.DataFrame()
    df_entropy = pd.DataFrame(results['entropy']) if 'entropy' in results else pd.DataFrame()
    df_resource = pd.DataFrame(resource_entropy_data)
    
    # Créer une figure avec 6 sous-graphiques
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle("Tableau de bord d'analyse des comportements utilisateurs", fontsize=20)
    
    # 1. Activités après heures
    if not df_after_hours.empty:
        axes[0, 0].bar(df_after_hours['user'], df_after_hours['after_hours_activity_count'], color='darkred')
        axes[0, 0].set_title('Activités après heures')
        axes[0, 0].set_xlabel('Utilisateur')
        axes[0, 0].set_ylabel('Nombre d\'activités')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Utilisateurs à haute activité
    if not df_high_activity.empty:
        axes[0, 1].bar(df_high_activity['user'], df_high_activity['activity_count'], color='darkblue')
        axes[0, 1].set_title('Utilisateurs à haute activité')
        axes[0, 1].set_xlabel('Utilisateur')
        axes[0, 1].set_ylabel('Nombre d\'activités')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Entropie des activités
    if not df_entropy.empty:
        axes[1, 0].bar(df_entropy['user'], df_entropy['entropy'], color='darkgreen')
        axes[1, 0].set_title('Entropie des activités')
        axes[1, 0].set_xlabel('Utilisateur')
        axes[1, 0].set_ylabel('Entropie')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Entropie des ressources
    if not df_resource.empty:
        axes[1, 1].bar(df_resource['user'].head(10), df_resource['resource_entropy'].head(10), color='purple')
        axes[1, 1].set_title('Entropie des ressources')
        axes[1, 1].set_xlabel('Utilisateur')
        axes[1, 1].set_ylabel('Entropie')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 5. Scatter plot activités vs entropie
    if not features_df.empty:
        scatter = axes[2, 0].scatter(features_df['total_activities'], features_df['activity_entropy'], 
                           s=features_df['unique_resources']/2 + 5,
                           c=features_df['after_hours_ratio'], cmap='viridis', alpha=0.7)
        axes[2, 0].set_title('Activités vs Entropie')
        axes[2, 0].set_xlabel('Nombre total d\'activités')
        axes[2, 0].set_ylabel('Entropie des activités')
        plt.colorbar(scatter, ax=axes[2, 0], label='Ratio après heures')
    
    # 6. Centralité des utilisateurs
    if not df_centrality.empty:
        axes[2, 1].plot(range(len(df_centrality)), df_centrality['total_degree'], 'o-', color='orange', markersize=8)
        axes[2, 1].set_title('Centralité des utilisateurs')
        axes[2, 1].set_xlabel('Utilisateur')
        axes[2, 1].set_ylabel('Degré total')
        axes[2, 1].set_xticks(range(len(df_centrality)))
        axes[2, 1].set_xticklabels(df_centrality['user'], rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('user_behavior_dashboard.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Pour tester les fonctions de visualisation
    from neo4j_setup import connect_to_neo4j
    from neo4j_analysis import detect_anomalies_neo4j, extract_features_for_ml, calculate_resource_entropy
    
    # Connexion à Neo4j
    uri = "bolt://localhost:7689"
    username = "neo4j"
    password = "2004@2004"
    graph = connect_to_neo4j(uri, username, password)
    
    # Analyse des comportements suspects
    anomaly_results = detect_anomalies_neo4j(graph)
    
    # Calcul de l'entropie des ressources
    resource_entropy = calculate_resource_entropy(graph)
    
    # Extraction des caractéristiques pour ML
    features = extract_features_for_ml(graph)
    
    # Création des visualisations
    visualize_anomalies(anomaly_results)
    visualize_features(features)
    visualize_resource_entropy(resource_entropy)
    
    # Création du dashboard
    create_dashboard(anomaly_results, features, resource_entropy)
    
    print("\nToutes les visualisations ont été créées avec succès au format PNG!")