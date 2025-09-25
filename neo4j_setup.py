import pandas as pd
from py2neo import Graph, Node, Relationship
import time
import sys

def connect_to_neo4j(uri, username, password):
    """
    Établir une connexion à Neo4j
    """
    try:
        graph = Graph(uri, auth=(username, password))
        print("Connexion à Neo4j établie avec succès")
        return graph
    except Exception as e:
        print(f"Erreur de connexion à Neo4j: {e}")
        sys.exit(1)

def clear_database(graph):
    """
    Supprimer toutes les données existantes dans la base Neo4j
    """
    try:
        graph.run("MATCH (n) DETACH DELETE n")
        print("Base de données Neo4j nettoyée")
    except Exception as e:
        print(f"Erreur lors du nettoyage de la base: {e}")

def create_graph(graph, df):
    """
    Créer le graphe dans Neo4j à partir du DataFrame
    """
    print("Importation des données dans Neo4j...")
    start_time = time.time()
    
    # Dictionnaires pour stocker les nœuds créés
    users = {}
    systems = {}
    activities = {}
    resources = {}
    
    try:
        # Traiter chaque ligne du DataFrame
        for index, row in df.iterrows():
            # Création des nœuds utilisateur
            user_id = str(row['user'])
            if user_id not in users:
                users[user_id] = Node("User", name=user_id, type="employee")
                graph.create(users[user_id])
            
            # Création des nœuds système
            system_id = str(row['pc'])
            if system_id not in systems:
                systems[system_id] = Node("System", name=system_id, type="workstation")
                graph.create(systems[system_id])
            
            # Création des nœuds activité
            activity_id = str(row['activity'])
            if activity_id not in activities:
                activities[activity_id] = Node("Activity", name=activity_id)
                graph.create(activities[activity_id])
            
            # Création des nœuds ressource
            resource_id = str(row['resource'])
            if resource_id not in resources:
                resources[resource_id] = Node("Resource", name=resource_id, type="file")
                graph.create(resources[resource_id])
            
            # Création des relations
            # Utilisateur -> Activité
            user_activity = Relationship(users[user_id], "PERFORMS", activities[activity_id],
                                         timestamp=row['timestamp'].isoformat(),
                                         hour_of_day=int(row['hour_of_day']),
                                         is_after_hours=int(row['is_after_hours']),
                                         is_weekend=int(row['is_weekend']))
            graph.create(user_activity)
            
            # Système -> Ressource
            system_resource = Relationship(systems[system_id], "ACCESSES", resources[resource_id],
                                          timestamp=row['timestamp'].isoformat())
            graph.create(system_resource)
            
            # Utilisateur -> Système
            user_system = Relationship(users[user_id], "USES", systems[system_id],
                                      timestamp=row['timestamp'].isoformat())
            graph.create(user_system)
            
            # Activité -> Ressource
            activity_resource = Relationship(activities[activity_id], "INVOLVES", resources[resource_id],
                                            timestamp=row['timestamp'].isoformat())
            graph.create(activity_resource)
            
            # Afficher la progression
            if index % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Traité {index} lignes en {elapsed:.2f} secondes")
        
        # Créer des indices pour améliorer les performances
        graph.run("CREATE INDEX user_name IF NOT EXISTS FOR (u:User) ON (u.name)")
        graph.run("CREATE INDEX system_name IF NOT EXISTS FOR (s:System) ON (s.name)")
        graph.run("CREATE INDEX activity_name IF NOT EXISTS FOR (a:Activity) ON (a.name)")
        graph.run("CREATE INDEX resource_name IF NOT EXISTS FOR (r:Resource) ON (r.name)")
        
        total_time = time.time() - start_time
        print(f"Importation terminée en {total_time:.2f} secondes. {len(df)} lignes traitées.")
        
        # Vérifier le nombre de nœuds créés
        user_count = graph.run("MATCH (u:User) RETURN count(u) AS count").data()[0]['count']
        print(f"Nombre d'utilisateurs créés: {user_count}")
        
        return True
    
    except Exception as e:
        print(f"Erreur lors de la création du graphe: {e}")
        return False

if __name__ == "__main__":
    # Test du script d'importation
    from data_preprocessing import preprocess_data
    
    # Configurations
    uri = "bolt://localhost:7689"
    username = "neo4j"
    password = "2004@2004"
    file_path = "/Users/HouDa/Desktop/Projet_NOSQL/demo_multi_activity_logs_10000.csv"
    
    # Prétraitement des données
    df = preprocess_data(file_path)
    
    # Connexion à Neo4j
    graph = connect_to_neo4j(uri, username, password)
    
    # Nettoyage de la base
    clear_database(graph)
    
    # Création du graphe
    success = create_graph(graph, df)
    
    if success:
        print("Importation des données dans Neo4j réussie!")
    else:
        print("Échec de l'importation des données.")