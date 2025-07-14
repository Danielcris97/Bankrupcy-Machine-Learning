import pandas as pd
import numpy as np
import os
import joblib
import json # For saving report data like model comparison results
import re # For cleaning column names

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    recall_score, precision_score, f1_score, accuracy_score
)
from tqdm import tqdm

# NUEVAS IMPORTACIONES PARA PCA Y KMEANS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Necesario para escalar


# Importar la utilidad tqdm_joblib para barras de progreso en GridSearchCV
# Asegúrate de que 'utils.py' está en el mismo directorio o en el PYTHONPATH
from utils import tqdm_joblib

# --- Directorios ---
# Define la ruta base del proyecto (un nivel arriba del directorio 'src')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data', 'processed') # Apunta directamente a 'processed'
models_dir = os.path.join(project_root, 'models')
reports_dir = os.path.join(project_root, 'reports')
train_dir = os.path.join(project_root, 'data', 'train') # Para guardar conjuntos de datos de train/test
test_dir = os.path.join(project_root, 'data', 'test')   # Para guardar conjuntos de datos de train/test


# Crear directorios si no existen
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# --- Funciones de Carga y Guardado ---

def load_data_processed(file_path: str) -> pd.DataFrame:
    """
    Carga el DataFrame preprocesado desde un archivo CSV.
    Este DataFrame DEBERÍA NO ESTAR ESCALADO AÚN, y solo tener las columnas limpias y nans/infinitos manejados.
    El escalado se hará en este script de entrenamiento.
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

def save_object(obj, file_path: str):
    """
    Guarda un objeto (modelo, X_train, etc.) usando joblib.
    Crea el directorio si no existe.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        joblib.dump(obj, file_path)
        print(f"Objeto guardado exitosamente con joblib en: {file_path}")
    except Exception as e:
        print(f"Error al guardar el objeto en {file_path}: {e}")

def load_object(file_path: str):
    """
    Carga un objeto (modelo, X_train, etc.) usando joblib.
    """
    try:
        obj = joblib.load(file_path)
        print(f"Objeto cargado exitosamente con joblib desde: {file_path}")
        return obj
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error al cargar el objeto desde {file_path}: {e}")
        return None

# --- Función para limpiar nombres de columnas (replica de data_preprocessing.py) ---
def limpiar_nombres_columnas(columnas):
    """
    Limpia una lista de nombres de columnas.
    Asegura que los nombres sean compatibles con el acceso a columnas de pandas.
    """
    columnas_limpias = []
    for col in columnas:
        col = str(col).strip()
        col = col.lower()
        col = col.replace(' ', '_')
        col = col.replace('%', 'percent')
        col = col.replace('(', '')
        col = col.replace(')', '')
        col = col.replace('-', '_')
        col = col.replace('/', '_')
        col = re.sub(r'[^a-z0-9_]', '', col)
        col = col.replace('___', '_').replace('__', '_') # Reducir múltiples underscores
        col = col.strip('_') # Eliminar underscores al principio/final
        columnas_limpias.append(col)
    return columnas_limpias


# --- Función para ejecutar GridSearchCV con barra de progreso ---
def run_gridsearch_with_tqdm(model, param_grid, X_train_data, y_train_data, scoring_metric, cv_folds=5, model_name="Modelo"):
    """
    Ejecuta GridSearchCV con una barra de progreso de tqdm y retorna el mejor modelo.
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scoring_metric,
        cv=cv_folds,
        n_jobs=-1
    )

    # Calcular el número total de iteraciones para la barra de progreso
    num_combinations = 1
    for values in param_grid.values():
        num_combinations *= len(values)
    total_iterations = num_combinations * cv_folds

    print(f"\n--- Iniciando GridSearchCV para {model_name} (Total iteraciones: {total_iterations}) ---")
    with tqdm_joblib(tqdm(desc=f"GridSearchCV {model_name}", total=total_iterations)):
        grid_search.fit(X_train_data, y_train_data)

    print(f"\n--- GridSearchCV para {model_name} Completado ---")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_

# --- Función para evaluar y mostrar resultados ---
def evaluate_model(model, X_test_data, y_test_data, model_name="Modelo"):
    """
    Evalúa un modelo y imprime la matriz de confusión, el reporte de clasificación y el ROC AUC.
    Retorna un diccionario con métricas clave.
    """
    y_pred = model.predict(X_test_data)
    y_proba = model.predict_proba(X_test_data)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\n--- Evaluación de {model_name} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test_data, y_pred))
    print("\nClassification Report:\n", classification_report(y_test_data, y_pred, target_names=["No Quiebra", "Quiebra"]))
    
    # Calcular métricas para la clase minoritaria (Quiebra)
    recall_quiebra = recall_score(y_test_data, y_pred, pos_label=1, zero_division=0)
    precision_quiebra = precision_score(y_test_data, y_pred, pos_label=1, zero_division=0)
    f1_quiebra = f1_score(y_test_data, y_pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, y_proba) if y_proba is not None else np.nan

    print(f"Recall (Quiebra): {recall_quiebra:.4f}")
    print(f"Precision (Quiebra): {precision_quiebra:.4f}")
    print(f"F1-score (Quiebra): {f1_quiebra:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC: N/A (modelo no soporta predict_proba)")

    return {
        "recall_quiebra": recall_quiebra,
        "precision_quiebra": precision_quiebra,
        "f1_score_quiebra": f1_quiebra,
        "accuracy": accuracy,
        "roc_auc": roc_auc
    }


# --- Bloque principal de ejecución cuando el script es llamado directamente ---
if __name__ == "__main__":
    print("--- Iniciando el pipeline de Entrenamiento de Modelos ---")

    # Define la ruta al archivo data_processed.csv
    processed_data_file = os.path.join(data_dir, 'data_processed.csv') # Archivo desde data_preprocessing.py
    
    # Cargar los parámetros de preprocesamiento, especialmente las columnas a eliminar por correlación
    preprocessor_params_path = os.path.join(models_dir, 'preprocessor_params.pkl')
    preprocessor_params = load_object(preprocessor_params_path)
    if preprocessor_params is None:
        print("Error: No se pudieron cargar los parámetros de preprocesamiento. Asegúrate de ejecutar data_preprocessing.py primero.")
        exit()
    
    cols_to_drop_corr = preprocessor_params.get('columns_dropped_correlation', [])
    print(f"Columnas a eliminar por correlación: {len(cols_to_drop_corr)}")

    # 1. Cargar datos procesados (este DF ya debería tener NaNs/infinitos manejados y net_income_flag eliminada)
    df_raw = load_data_processed(processed_data_file)
    if df_raw.empty:
        print("No se pudieron cargar los datos procesados. Terminando el entrenamiento.")
        exit()

    # 2. Separar características (X) y variable objetivo (y)
    X = df_raw.drop(columns=["bankrupt"])
    y = df_raw["bankrupt"]

    # Asegurarse de que los nombres de las columnas en X estén limpios
    X.columns = limpiar_nombres_columnas(X.columns)

    # 3. Eliminar las 24 características altamente correlacionadas
    # ESTO SE HACE AQUÍ para que el escalador y PCA/KMeans trabajen con las 78 features.
    X_after_corr_drop = X.copy()
    for col in cols_to_drop_corr:
        if col in X_after_corr_drop.columns:
            X_after_corr_drop.drop(columns=[col], inplace=True)
    
    print(f"Dimensiones de X después de eliminar correlacionadas: {X_after_corr_drop.shape}") # Debería ser (N, 78)


    # 4. División de datos en conjuntos de entrenamiento y prueba
    # Esta es la división inicial para los datos de 78 características
    X_train_full_78_features, X_test_full_78_features, y_train, y_test = train_test_split(
        X_after_corr_drop, y, test_size=0.2, random_state=10, stratify=y
    )
    print(f"\nDatos divididos. X_train_full_78_features: {X_train_full_78_features.shape}, X_test_full_78_features: {X_test_full_78_features.shape}")
    print(f"Conteo de clases en y_train original:\n{y_train.value_counts()}")

    # 5. Escalado de Características
    # El scaler se ajusta con las 78 características.
    scaler = StandardScaler()
    X_train_scaled_full_df = pd.DataFrame(scaler.fit_transform(X_train_full_78_features), 
                                        columns=X_train_full_78_features.columns,
                                        index=X_train_full_78_features.index)
    X_test_scaled_full_df = pd.DataFrame(scaler.transform(X_test_full_78_features),
                                       columns=X_test_full_78_features.columns,
                                       index=X_test_full_78_features.index)

    # Guardar el scaler
    save_object(scaler, os.path.join(models_dir, 'scaler_for_all_features.pkl'))

    # ¡CRUCIAL! Guardar la lista de características que el scaler fue ajustado
    # Esto es lo que 'streamlit_app.py' espera para el 'scaler_features.pkl'
    save_object(X_train_full_78_features.columns.tolist(), os.path.join(models_dir, 'scaler_features.pkl'))
    print(f"Lista de características del scaler (78 features) guardada en {os.path.join(models_dir, 'scaler_features.pkl')}")


    # --- Entrenamiento de modelos no supervisados (PCA y KMeans) ---
    # Estos modelos usan las 78 características escaladas
    
    # PCA
    print("\n--- Aplicando PCA ---")
    pca = PCA().fit(X_train_scaled_full_df) # Ajustar PCA en las 78 características escaladas
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
    
    pca_model = PCA(n_components=n_components_95, random_state=42)
    X_train_pca = pca_model.fit_transform(X_train_scaled_full_df)
    X_test_pca = pca_model.transform(X_test_scaled_full_df) # Transformar también el conjunto de prueba
    save_object(pca_model, os.path.join(models_dir, 'pca_model.pkl'))
    print(f"Modelo PCA guardado en: {os.path.join(models_dir, 'pca_model.pkl')}")
    print(f"Número de componentes PCA resultantes para 95% de varianza: {pca_model.n_components_}")

    # KMeans
    print("\n--- Aplicando KMeans ---")
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_model.fit(X_train_pca) # KMeans se ajusta sobre los datos transformados por PCA
    save_object(kmeans_model, os.path.join(models_dir, 'kmeans_model.pkl'))
    print(f"Modelo KMeans guardado en: {os.path.join(models_dir, 'kmeans_model.pkl')}")
    print("\n--- Modelos no supervisados completados ---")


    # --- Preparación de datos para los modelos Supervisados ---
    # AHORA: X_train_final y X_test_final serán las 78 características escaladas.
    # NO eliminamos las 5 características menos importantes aquí para los modelos supervisados.
    # Esto asegura que RandomForest_ClassWeighted pueda alcanzar su máximo recall.
    X_train_final = X_train_scaled_full_df.copy()
    X_test_final = X_test_scaled_full_df.copy()

    print(f"\nDimensiones de X_train_final (para modelos supervisados, 78 features): {X_train_final.shape}")
    print(f"Dimensiones de X_test_final (para modelos supervisados, 78 features): {X_test_final.shape}")

    # Guardar las características finales que los modelos supervisados esperan (ahora 78 features)
    save_object(X_train_final.columns.tolist(), os.path.join(models_dir, 'model_features.pkl'))
    print(f"Lista de características del modelo final (78 features) guardada en {os.path.join(models_dir, 'model_features.pkl')}")


    # Aplicar SMOTE a los datos de entrenamiento para balancear los modelos supervisados
    print("\n--- Aplicando SMOTE con sampling_strategy=0.25 a X_train_final, y_train ---")
    smote_0_2 = SMOTE(random_state=42, sampling_strategy=0.25)
    X_train_smote, y_train_smote = smote_0_2.fit_resample(X_train_final, y_train)
    print("Conteo de clases después de SMOTE (0.25):\n", y_train_smote.value_counts())
    
    # Guardar los X_test_final y y_test (para uso en la app y consistencia)
    save_object(X_test_final, os.path.join(test_dir, 'X_test_final.pkl'))
    save_object(y_test, os.path.join(test_dir, 'y_test_final.pkl'))


    # --- Opciones de Entrenamiento de Modelos ---
    all_model_results = [] # Para almacenar los resultados de todos los modelos

    # --- RandomForest con class_weight='balanced' (sin SMOTE) ---
    print("\n--- Entrenamiento: RandomForest con class_weight='balanced' (sin SMOTE) ---")
    rf_cw_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    
    param_grid_rf_cw = {
        "n_estimators": [500, 2000],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2]
    }
    
    best_rf_cw, best_params_rf_cw = run_gridsearch_with_tqdm(
        rf_cw_model, param_grid_rf_cw, X_train_final, y_train, # X_train_final ahora tiene 78 features
        scoring_metric="recall", cv_folds=5, model_name="RandomForest_ClassWeighted"
    )
    
    metrics_rf_cw = evaluate_model(best_rf_cw, X_test_final, y_test, "RandomForest_ClassWeighted")
    
    model_result_rf_cw = {
        "model": "RandomForest_ClassWeighted",
        "best_params": best_params_rf_cw,
        **metrics_rf_cw
    }
    all_model_results.append(model_result_rf_cw)
    save_object(best_rf_cw, os.path.join(models_dir, 'best_rf_classweighted_model.pkl'))


    # --- Entrenamiento de otros modelos con SMOTE y las 78 features ---

    # 1. RandomForest con SMOTE
    print("\n--- Entrenamiento: RandomForest con SMOTE (sampling_strategy=0.25) ---")
    rf_smote_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    param_grid_rf_smote = {
        "n_estimators": [500, 1000,2000],
        "max_depth": [None, 6, 10],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2]
    }
    
    best_rf_smote, best_params_rf_smote = run_gridsearch_with_tqdm(
        rf_smote_model, param_grid_rf_smote, X_train_smote, y_train_smote, # X_train_smote ahora tiene 78 features
        scoring_metric="recall", cv_folds=5, model_name="RandomForest_SMOTE_Final"
    )

    metrics_rf_smote = evaluate_model(best_rf_smote, X_test_final, y_test, "RandomForest_SMOTE_Final")
    
    model_result_rf_smote = {
        "model": "RandomForest_SMOTE_Final",
        "best_params": best_params_rf_smote,
        **metrics_rf_smote
    }
    all_model_results.append(model_result_rf_smote)
    save_object(best_rf_smote, os.path.join(models_dir, 'final_model_randomforest.pkl'))


    # 2. AdaBoost
    print("\n--- Entrenamiento: AdaBoost ---")
    ada_model = AdaBoostClassifier(random_state=42)
    param_grid_ada = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.5, 1.0, 1.5]
    }
    best_ada, best_params_ada = run_gridsearch_with_tqdm(
        ada_model, param_grid_ada, X_train_smote, y_train_smote, # X_train_smote ahora tiene 78 features
        scoring_metric="recall", model_name="AdaBoost"
    )
    metrics_ada = evaluate_model(best_ada, X_test_final, y_test, "AdaBoost")
    
    model_result_ada = {
        "model": "AdaBoost",
        "best_params": best_params_ada,
        **metrics_ada
    }
    all_model_results.append(model_result_ada)
    save_object(best_ada, os.path.join(models_dir, 'final_model_adaboost.pkl')) # Nombre del archivo para AdaBoost

    # 3. GradientBoosting
    print("\n--- Entrenamiento: GradientBoosting ---")
    gb_model = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {
        "n_estimators": [200, 300,],
        "learning_rate": [0.1, 0.2,],
        "max_depth": [5, 7]
    }
    best_gb, best_params_gb = run_gridsearch_with_tqdm(
        gb_model, param_grid_gb, X_train_smote, y_train_smote, # X_train_smote ahora tiene 78 features
        scoring_metric="recall", model_name="GradientBoosting"
    )
    metrics_gb = evaluate_model(best_gb, X_test_final, y_test, "GradientBoosting")
    
    model_result_gb = {
        "model": "GradientBoosting",
        "best_params": best_params_gb,
        **metrics_gb
    }
    all_model_results.append(model_result_gb)
    save_object(best_gb, os.path.join(models_dir, 'final_model_gradientboosting.pkl'))

    # 4. XGBoost
    print("\n--- Entrenamiento: XGBoost ---")
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False, n_jobs=-1)
    param_grid_xgb = {
        "n_estimators": [200, 300],
        "learning_rate": [0.1, 0.2, 0.3],
        "max_depth": [5, 7],
    }
    best_xgb, best_params_xgb = run_gridsearch_with_tqdm(
        xgb_model, param_grid_xgb, X_train_smote, y_train_smote, # X_train_smote ahora tiene 78 features
        scoring_metric="recall", model_name="XGBoost"
    )
    metrics_xgb = evaluate_model(best_xgb, X_test_final, y_test, "XGBoost")
    
    model_result_xgb = {
        "model": "XGBoost",
        "best_params": best_params_xgb,
        **metrics_xgb
    }
    all_model_results.append(model_result_xgb)
    save_object(best_xgb, os.path.join(models_dir, 'final_model_xgboost.pkl'))

    # 5. LightGBM
    print("\n--- Entrenamiento: LightGBM ---")
    lgbm_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    param_grid_lgbm = {
        "n_estimators": [200],
        "learning_rate": [0.1, 0.2, 0.3],
        "num_leaves": [20, 31, 50],
        "max_depth": [7,-1],
    }
    best_lgbm, best_params_lgbm = run_gridsearch_with_tqdm(
        lgbm_model, param_grid_lgbm, X_train_smote, y_train_smote, # X_train_smote ahora tiene 78 features
        scoring_metric="recall", model_name="LightGBM"
    )
    metrics_lgbm = evaluate_model(best_lgbm, X_test_final, y_test, "LightGBM")
    
    model_result_lgbm = {
        "model": "LightGBM",
        "best_params": best_params_lgbm,
        **metrics_lgbm
    }
    all_model_results.append(model_result_lgbm)
    save_object(best_lgbm, os.path.join(models_dir, 'final_model_lightgbm.pkl'))


    # --- Resumen Final de Resultados ---
    print("\n--- Resumen de los mejores modelos encontrados (ordenado por recall en el test set) ---")
    df_results_summary = pd.DataFrame(all_model_results)
    df_results_summary_sorted = df_results_summary.sort_values(by="recall_quiebra", ascending=False)
    print(df_results_summary_sorted.to_string())

    # Guardar el resumen de resultados
    df_results_summary_sorted.to_csv(os.path.join(reports_dir, 'model_comparison_results.csv'), index=False)
    print(f"\nResultados de la comparación de modelos guardados en: {os.path.join(reports_dir, 'model_comparison_results.csv')}")

    print("\n--- Pipeline de Entrenamiento de Modelos Finalizado ---")