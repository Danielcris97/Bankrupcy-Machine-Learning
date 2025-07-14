import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)

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
        print(f"Error: Archivo no encontrado en '{file_path}'. Por favor, verifica la ruta y que el archivo exista.")
        return None
    except Exception as e:
        print(f"Error al cargar el objeto desde {file_path}: {e}")
        return None

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    print("--- Iniciando Evaluación Detallada del Modelo Predictivo ---")

    # --- Definición de Rutas Absolutas ---
    # Obtener el directorio actual del script (src/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir un nivel para llegar a la raíz del proyecto
    project_root = os.path.dirname(current_script_dir)

    # Definir rutas a los directorios usando la raíz del proyecto
    models_dir = os.path.join(project_root, 'models')
    test_dir = os.path.join(project_root, 'data', 'test')
    reports_dir = os.path.join(project_root, 'reports')

    os.makedirs(reports_dir, exist_ok=True) # Asegurarse de que el directorio de reportes exista

    # 1. Cargar el mejor modelo (RandomForest_ClassWeighted) y los datos de prueba
   
    best_model_path = os.path.join(models_dir, 'best_rf_classweighted_model.pkl') 
    X_test_path = os.path.join(test_dir, 'X_test_final.pkl')
    y_test_path = os.path.join(test_dir, 'y_test_final.pkl')
    # model_features.pkl ahora contendrá las 78 características
    model_features_path = os.path.join(models_dir, 'model_features.pkl')

    model = load_object(best_model_path)
    X_test = load_object(X_test_path)
    y_test = load_object(y_test_path)
    model_features = load_object(model_features_path) # Cargar los nombres de las características

    if model_features is None or X_test is None or y_test is None or model is None:
        print("Error: No se pudieron cargar todos los objetos necesarios. Terminando la evaluación.")
        exit()

    print(f"\nModelo seleccionado para evaluación: {type(model).__name__}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 2. Evaluación con el umbral por defecto (0.5)
    print("\n--- Evaluación con Umbral por Defecto (0.5) ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["No Quiebra", "Quiebra"]))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # 3. Curva Precision-Recall
    print("\n--- Generando Curva Precision-Recall ---")
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall para Detección de Quiebra')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    # Opcional: Mostrar el punto actual (umbral 0.5)
    closest_idx = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(recall[closest_idx], precision[closest_idx], 'ro', markersize=8, label=f'Umbral 0.5 (P={precision[closest_idx]:.2f}, R={recall[closest_idx]:.2f})')
    plt.legend()

    pr_curve_path = os.path.join(reports_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    print(f"Curva Precision-Recall guardada en: {pr_curve_path}")
    # plt.show() # Descomentar si quieres que la gráfica aparezca al ejecutar

    # 4. Análisis de Importancia de Características
    print("\n--- Análisis de Importancia de Características ---")
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        features_df = pd.DataFrame({'Feature': model_features, 'Importance': feature_importances})
        features_df = features_df.sort_values(by='Importance', ascending=False)
        print("Top 15 Características Más Importantes:\n", features_df.head(15))

        # Visualización de las Top N características
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=features_df.head(15))
        # CAMBIO: Actualizar el título del gráfico
        plt.title('Importancia de las Características (RandomForest_ClassWeighted)') 
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.tight_layout()
        feature_importance_path = os.path.join(reports_dir, 'feature_importance_randomforest_classweighted.png') # Nombre de archivo actualizado
        plt.savefig(feature_importance_path)
        print(f"Gráfico de Importancia de Características guardado en: {feature_importance_path}")
        # plt.show() # Descomentar si quieres que la gráfica aparezca al ejecutar
    else:
        print("El modelo seleccionado no tiene un atributo 'feature_importances_'.")

    print("\n--- Evaluación Detallada del Modelo Predictivo Finalizada ---")