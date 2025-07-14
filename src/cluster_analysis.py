import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json # Import the json module

# --- Funciones de Carga de Objetos ---
def load_object(file_path: str):
    """
    Carga un objeto usando joblib.
    """
    try:
        obj = joblib.load(file_path)
        print(f"Objeto cargado exitosamente desde: {file_path}")
        return obj
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error al cargar el objeto desde {file_path}: {e}")
        return None

def load_data_processed(file_path: str) -> pd.DataFrame:
    """
    Carga el DataFrame preprocesado desde un archivo CSV.
    Este DataFrame YA DEBERÍA ESTAR ESCALADO y con la columna 'bankrupt'.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Datos procesados cargados exitosamente desde: {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en '{file_path}'. Asegúrate de ejecutar data_preprocessing.py primero.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al cargar los datos procesados: {e}")
        return pd.DataFrame()


# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    print("--- Iniciando Análisis de Clústeres ---")

    # Rutas a los archivos
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    models_dir = os.path.join(project_root, 'models')
    reports_dir = os.path.join(project_root, 'reports')
    # El archivo processed_data.csv desde data_preprocessing.py
    processed_data_file = os.path.join(project_root, 'data', 'processed', 'data_processed.csv') 
    
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Cargar modelos y datos
    pca_model_path = os.path.join(models_dir, 'pca_model.pkl')
    kmeans_model_path = os.path.join(models_dir, 'kmeans_model.pkl')
    # model_features.pkl ahora contendrá las 78 características que se usaron para PCA/KMeans
    model_features_path = os.path.join(models_dir, 'model_features.pkl') 

    pca_model = load_object(pca_model_path)
    kmeans_model = load_object(kmeans_model_path)
    original_features = load_object(model_features_path) # Estas son las 78 features escaladas

    df_processed = load_data_processed(processed_data_file)
    
    if pca_model is None or kmeans_model is None or original_features is None or df_processed.empty:
        print("Error: No se pudieron cargar todos los objetos/datos necesarios. Terminando el análisis de clústeres.")
        exit()

    # Separar X e y del DataFrame cargado
    # Importante: X_for_clustering debe ser el mismo conjunto de características (78) que se usó para entrenar PCA y KMeans
    X_for_clustering = df_processed[original_features] # Usa solo las 78 features que el modelo espera
    y_target = df_processed['bankrupt']

    print(f"\nDatos para clustering shape: {X_for_clustering.shape}")

    # 2. Aplicar PCA para obtener los datos en el espacio de componentes principales
    X_pca = pca_model.transform(X_for_clustering)
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    # 3. Predecir los clústeres usando el KMeans que fue entrenado con datos PCA
    cluster_labels = kmeans_model.predict(X_pca)
    
    # 4. Unir etiquetas de clúster con el DataFrame original (o una copia con características originales)
    df_clustered = X_for_clustering.copy() # Usar las 78 features para análisis
    df_clustered['cluster'] = cluster_labels
    df_clustered['bankrupt'] = y_target # Añadir la variable objetivo para analizar la proporción de quiebras por clúster

    print("\n--- Análisis de Composición de Clústeres ---")
    
    # Cluster Counts
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    print("Conteo de empresas por clúster:\n", cluster_counts)
    
    # Save Cluster Counts to JSON
    cluster_counts_path = os.path.join(reports_dir, 'cluster_counts.json')
    with open(cluster_counts_path, 'w') as f:
        json.dump(cluster_counts.to_dict(), f, indent=4)
    print(f"Conteo de clústeres guardado en: {cluster_counts_path}")


    # 5. Caracterización de Clústeres
    cluster_profiles = df_clustered.groupby('cluster')[original_features].mean()
    print("\nMedia de Características (originales, escaladas) por Clúster:\n", cluster_profiles)
    
    # Save Cluster Profiles to JSON
    cluster_profiles_path = os.path.join(reports_dir, 'cluster_profiles.json')
    cluster_profiles_dict = {
        str(idx): row.to_dict() for idx, row in cluster_profiles.iterrows()
    }
    with open(cluster_profiles_path, 'w') as f:
        json.dump(cluster_profiles_dict, f, indent=4)
    print(f"Perfiles de clústeres guardados en: {cluster_profiles_path}")

    
    # 6. Proporción de Quiebras por Clúster
    bankrupt_proportion = df_clustered.groupby('cluster')['bankrupt'].value_counts(normalize=True).unstack(fill_value=0)
    print("\nProporción de Clases (0: No Quiebra, 1: Quiebra) por Clúster:\n", bankrupt_proportion)
    
    # Save Bankrupt Proportions to JSON
    bankrupt_proportion_path = os.path.join(reports_dir, 'bankrupt_proportion_by_cluster.json')
    bankrupt_proportion_dict = {
        str(idx): row.to_dict() for idx, row in bankrupt_proportion.iterrows()
    }
    with open(bankrupt_proportion_path, 'w') as f:
        json.dump(bankrupt_proportion_dict, f, indent=4)
    print(f"Proporción de quiebras por clúster guardada en: {bankrupt_proportion_path}")

    
    if 1 in bankrupt_proportion.columns:
        print("\nProporción de Quiebras (Clase 1) por Clúster (ordenado de mayor a menor):")
        print(bankrupt_proportion[1].sort_values(ascending=False))
    else:
        print("\nNo hay instancias de 'Quiebra' (clase 1) en los clústeres del dataset procesado.")


    # 7. Visualización de Clústeres (en 2D/3D si es posible)
    if df_pca.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=df_pca.iloc[:, 0], y=df_pca.iloc[:, 1], hue=cluster_labels, palette='viridis', legend='full', alpha=0.7)
        plt.title('Clústeres de Empresas (PCA Componentes 1 y 2)')
        plt.xlabel(f'Componente Principal 1 ({pca_model.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'Componente Principal 2 ({pca_model.explained_variance_ratio_[1]*100:.2f}%)')
        plt.grid(True)
        cluster_plot_path = os.path.join(reports_dir, 'clusters_pca_2d.png')
        plt.savefig(cluster_plot_path)
        print(f"Gráfico de Clústeres (PCA 2D) guardado en: {cluster_plot_path}")
        # plt.show() # Keep commented if you don't want the plot to pop up during execution
    else:
        print("\nNo hay suficientes componentes PCA para una visualización 2D.")

    print("\n--- Análisis de Clústeres Finalizado ---")

