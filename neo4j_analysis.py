import pandas as pd
import numpy as np
from py2neo import Graph
import math

def detect_anomalies_neo4j(graph):
    """
    Détecter des comportements potentiellement suspects en utilisant Neo4j
    """
    print("Analyse des comportements avec Neo4j...")
    results = {}
    
    # 1. Activités après les heures normales
    print("Détection des activités après les heures normales...")
    after_hours_query = """
    MATCH (u:User)-[r:PERFORMS]->(a:Activity)
    WHERE r.is_after_hours = 1
    WITH u, count(r) AS after_hours_activity_count
    RETURN u.name AS user, after_hours_activity_count
    ORDER BY after_hours_activity_count DESC
    LIMIT 10
    """
    results['after_hours'] = graph.run(after_hours_query).data()
    print(f"Nombre d'utilisateurs trouvés pour activités après heures: {len(results['after_hours'])}")
    
    # 2. Accès aux ressources inhabituelles
    print("Détection des accès aux ressources inhabituelles...")
    unusual_resource_access = """
    MATCH (u:User)-[:PERFORMS]->(:Activity)-[:INVOLVES]->(r:Resource)
    WITH u, r, count(*) AS access_count
    WITH u, collect({resource: r.name, count: access_count}) AS resource_accesses
    WITH u, resource_accesses, size(resource_accesses) AS unique_resources
    RETURN u.name AS user, unique_resources, resource_accesses
    ORDER BY unique_resources DESC
    LIMIT 10
    """
    results['unusual_resources'] = graph.run(unusual_resource_access).data()
    print(f"Nombre d'utilisateurs trouvés pour accès aux ressources inhabituelles: {len(results['unusual_resources'])}")
    
    # 3. Utilisateurs avec activité élevée
    print("Détection des utilisateurs avec activité élevée...")
    high_activity_users = """
    MATCH (u:User)-[r:PERFORMS]->(a:Activity)
    WITH u, count(r) AS activity_count
    RETURN u.name AS user, activity_count
    ORDER BY activity_count DESC
    LIMIT 10
    """
    results['high_activity'] = graph.run(high_activity_users).data()
    print(f"Nombre d'utilisateurs trouvés avec activité élevée: {len(results['high_activity'])}")
    
    # 4. Patterns d'activité
    print("Analyse des patterns d'activité...")
    activity_patterns = """
    MATCH (u:User)-[r:PERFORMS]->(a:Activity)
    WITH u, a.name AS activity_type, count(*) AS activity_count
    ORDER BY u.name, activity_count DESC
    WITH u, collect({activity: activity_type, count: activity_count}) AS activities
    RETURN u.name AS user, activities
    LIMIT 10
    """
    results['activity_patterns'] = graph.run(activity_patterns).data()
    print(f"Nombre d'utilisateurs trouvés pour patterns d'activité: {len(results['activity_patterns'])}")
    
    # 5. Centralité de degré
    print("Calcul de la centralité de degré...")
    centrality_query = """
    MATCH (u:User)
    CALL {
        WITH u
        MATCH (u)-[r]->()
        RETURN count(r) AS out_degree
    }
    CALL {
        WITH u
        MATCH (u)<-[r]-()
        RETURN count(r) AS in_degree
    }
    WITH u, out_degree + in_degree AS total_degree
    RETURN u.name AS user, total_degree
    ORDER BY total_degree DESC
    LIMIT 10
    """
    results['centrality'] = graph.run(centrality_query).data()
    print(f"Nombre d'utilisateurs trouvés pour centralité de degré: {len(results['centrality'])}")
    
    # 6. Calcul de l'entropie des activités
    print("Calcul de l'entropie des activités...")
    activity_entropy_query = """
    MATCH (u:User)-[:PERFORMS]->(a:Activity)
    WITH u, a.name AS activity, count(*) AS freq
    WITH u, collect({activity: activity, freq: freq}) AS activities, sum(freq) AS total
    RETURN u.name AS user, activities, total
    """
    entropy_results = graph.run(activity_entropy_query).data()
    print(f"Nombre d'utilisateurs trouvés pour calcul d'entropie: {len(entropy_results)}")
    
    # Calcul de l'entropie en Python
    users_entropy = []
    for record in entropy_results:
        user = record['user']
        activities = record['activities']
        total = record['total']
        
        if total > 0:
            entropy = 0
            for act in activities:
                p = act['freq'] / total
                entropy -= p * math.log2(p)
            users_entropy.append({'user': user, 'entropy': entropy, 'activity_count': total})
    
    # Trier par entropie décroissante
    users_entropy.sort(key=lambda x: x['entropy'], reverse=True)
    results['entropy'] = users_entropy[:10]
    
    # Afficher les résultats
    for category, data in results.items():
        print(f"\n=== {category.upper()} ===")
        for item in data:
            print(item)
    
    return results

def extract_features_for_ml(graph):
    """
    Extraire des caractéristiques pour l'apprentissage automatique à partir de Neo4j
    """
    print("Extraction des caractéristiques pour le ML...")
    
    feature_query = """
    MATCH (u:User)
    
    // Nombre total d'activités
    CALL {
        WITH u
        MATCH (u)-[r:PERFORMS]->()
        RETURN count(r) AS total_activities
    }
    
    // Activités après les heures normales
    CALL {
        WITH u
        MATCH (u)-[r:PERFORMS]->()
        WHERE r.is_after_hours = 1
        RETURN count(r) AS after_hours_activities
    }
    
    // Activités le weekend
    CALL {
        WITH u
        MATCH (u)-[r:PERFORMS]->()
        WHERE r.is_weekend = 1
        RETURN count(r) AS weekend_activities
    }
    
    // Nombre de systèmes utilisés
    CALL {
        WITH u
        MATCH (u)-[:USES]->(s:System)
        RETURN count(DISTINCT s) AS unique_systems
    }
    
    // Nombre de ressources accédées
    CALL {
        WITH u
        MATCH (u)-[:PERFORMS]->()-[:INVOLVES]->(r:Resource)
        RETURN count(DISTINCT r) AS unique_resources
    }
    
    // Types d'activités différentes
    CALL {
        WITH u
        MATCH (u)-[:PERFORMS]->(a:Activity)
        RETURN count(DISTINCT a) AS activity_types
    }
    
    RETURN u.name AS user, 
           total_activities,
           after_hours_activities,
           weekend_activities,
           unique_systems,
           unique_resources,
           activity_types,
           1.0 * after_hours_activities / CASE WHEN total_activities > 0 THEN total_activities ELSE 1 END AS after_hours_ratio,
           1.0 * weekend_activities / CASE WHEN total_activities > 0 THEN total_activities ELSE 1 END AS weekend_ratio
    """
    
    features_df = pd.DataFrame(graph.run(feature_query).data())
    print(f"Nombre d'utilisateurs trouvés pour extraction de caractéristiques: {len(features_df)}")
    
    # Ajout du calcul d'entropie
    entropy_query = """
    MATCH (u:User)-[:PERFORMS]->(a:Activity)
    WITH u, a.name AS activity, count(*) AS freq
    WITH u, collect({activity: activity, freq: freq}) AS activities, sum(freq) AS total
    RETURN u.name AS user, activities, total
    """
    
    entropy_data = graph.run(entropy_query).data()
    print(f"Nombre d'utilisateurs trouvés pour calcul d'entropie des activités: {len(entropy_data)}")
    entropy_values = {}
    
    for record in entropy_data:
        user = record['user']
        activities = record['activities']
        total = record['total']
        
        if total > 0:
            entropy = 0
            for act in activities:
                p = act['freq'] / total
                entropy -= p * math.log2(p)
            entropy_values[user] = entropy
    
    # Ajout de l'entropie au DataFrame des caractéristiques
    features_df['activity_entropy'] = features_df['user'].map(entropy_values)
    
    # Calcul de l'entropie temporelle (par heure de la journée)
    temporal_entropy_query = """
    MATCH (u:User)-[r:PERFORMS]->(a:Activity)
    WITH u, r.hour_of_day AS hour, count(*) AS freq
    WITH u, collect({hour: hour, freq: freq}) AS hourly_activities, sum(freq) AS total
    RETURN u.name AS user, hourly_activities, total
    """
    
    temporal_data = graph.run(temporal_entropy_query).data()
    print(f"Nombre d'utilisateurs trouvés pour calcul d'entropie temporelle: {len(temporal_data)}")
    temporal_entropy = {}
    
    for record in temporal_data:
        user = record['user']
        hourly = record['hourly_activities']
        total = record['total']
        
        if total > 0:
            entropy = 0
            for h in hourly:
                p = h['freq'] / total
                entropy -= p * math.log2(p)
            temporal_entropy[user] = entropy
    
    # Ajout de l'entropie temporelle au DataFrame
    features_df['temporal_entropy'] = features_df['user'].map(temporal_entropy)
    
    print(f"Caractéristiques extraites pour {len(features_df)} utilisateurs.")
    
    return features_df

def calculate_resource_entropy(graph):
    """
    Calculer l'entropie des ressources accédées par chaque utilisateur
    """
    print("Calcul de l'entropie des ressources...")
    
    resource_entropy_query = """
    MATCH (u:User)-[:PERFORMS]->()-[:INVOLVES]->(r:Resource)
    WITH u, r.name AS resource, count(*) AS freq
    WITH u, collect({resource: resource, freq: freq}) AS resources, sum(freq) AS total
    RETURN u.name AS user, resources, total
    """
    
    resource_data = graph.run(resource_entropy_query).data()
    print(f"Nombre d'utilisateurs trouvés pour calcul d'entropie des ressources: {len(resource_data)}")
    users_resource_entropy = []
    
    for record in resource_data:
        user = record['user']
        resources = record['resources']
        total = record['total']
        
        if total > 0:
            entropy = 0
            for res in resources:
                p = res['freq'] / total
                entropy -= p * math.log2(p)
            users_resource_entropy.append({
                'user': user, 
                'resource_entropy': entropy, 
                'unique_resources': len(resources)
            })
    
    # Trier par entropie décroissante
    users_resource_entropy.sort(key=lambda x: x['resource_entropy'], reverse=True)
    
    print(f"\n=== RESOURCE ENTROPY ===")
    for item in users_resource_entropy[:10]:
        print(item)
    
    return users_resource_entropy

if __name__ == "__main__":
    # Test du script d'analyse
    from neo4j_setup import connect_to_neo4j
    
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
    print(features.head())