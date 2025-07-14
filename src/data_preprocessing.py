import pandas as pd
import numpy as np
import os
import joblib # Usaremos joblib para guardar el scaler y los parámetros del preprocesador
import pickle # Aunque joblib es preferido para scikit-learn, pickle funciona para preprocessor_params si es un dict

# --- Funciones Auxiliares ---

def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los nombres de las columnas de un DataFrame para hacerlos compatibles.
    Convierte a minúsculas, reemplaza espacios y caracteres especiales por guiones bajos.
    """
    import re
    nuevos_nombres = []
    for col in df.columns:
        col = str(col).strip()  # Eliminar espacios al inicio/final
        col = col.lower()       # Convertir a minúsculas
        col = col.replace(' ', '_')
        col = col.replace('%', 'percent')
        col = col.replace('(', '')
        col = col.replace(')', '')
        col = col.replace('-', '_')
        col = col.replace('/', '_')
        col = re.sub(r'[^a-z0-9_]', '', col) # Eliminar cualquier otro caracter no alfanumérico/guion bajo
        col = col.replace('___', '_').replace('__', '_') # Unificar múltiples guiones bajos
        col = col.strip('_')    # Eliminar guiones bajos al inicio/final si los hubiera
        nuevos_nombres.append(col)
    df.columns = nuevos_nombres
    return df

def save_object(obj, file_path: str):
    """
    Guarda un objeto Python en un archivo usando joblib (preferido para modelos/escaladores de sklearn).
    Crea los directorios necesarios si no existen.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        joblib.dump(obj, file_path)
        print(f"Objeto guardado exitosamente en: {file_path}")
    except Exception as e:
        print(f"Error al guardar el objeto: {e}")

# --- Bloque Principal de Preprocesamiento ---

if __name__ == "__main__":
    print("--- Iniciando el pipeline de Preprocesamiento de Datos ---")

    # Rutas de archivos (Asegúrate de que estas rutas sean correctas para tu estructura de proyecto)
    raw_data_path = '../data/raw/data.csv'
    processed_data_dir = '../data/processed'
    models_dir = '../models'
    
    # Crear directorios si no existen
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Cargar datos
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Datos raw cargados. Shape inicial: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Archivo raw no encontrado en '{raw_data_path}'. Por favor, verifica la ruta.")
        exit() # Sale del script si el archivo no se encuentra

    # 2. Limpiar nombres de columnas (¡PRIMER PASO CRUCIAL!)
    df = limpiar_nombres_columnas(df)
    print("Nombres de columnas limpiados.")
    
    # Renombrar explícitamente 'bankrupt?' a 'bankrupt' si existe después de la limpieza
    # Esto asegura consistencia ya que la limpieza puede variar ligeramente
    if 'bankrupt' not in df.columns and 'bankrupt?' in df.columns:
        df.rename(columns={'bankrupt?': 'bankrupt'}, inplace=True)
        print("Columna 'bankrupt?' renombrada a 'bankrupt'.")
    elif 'bankrupt?' in df.columns: # Si ya se limpia a 'bankrupt', no hace falta el rename
         print("La columna 'bankrupt?' ya fue manejada por la función de limpieza.")


    # 3. Eliminar columnas con un solo valor único
    # Excluir la columna 'bankrupt' del chequeo si es el caso (es 0 o 1)
    cols_to_drop_single_value = [col for col in df.columns if df[col].nunique() == 1 and col != 'bankrupt']
    if cols_to_drop_single_value:
        df.drop(columns=cols_to_drop_single_value, inplace=True)
        print(f"Columnas con un solo valor único eliminadas: {cols_to_drop_single_value}")
    else:
        print("No se encontraron columnas con un solo valor único (excluyendo el target) para eliminar.")

    # 4. Manejar valores faltantes (NaN y Infinitos)
    # Primero, reemplazar valores infinitos con NaN para que puedan ser imputados
    initial_inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    if initial_inf_count > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"Se reemplazaron {initial_inf_count} valores infinitos con NaN.")
    else:
        print("No se encontraron valores infinitos en el DataFrame.")

    # Identificar columnas numéricas para imputación (excluyendo el target)
    numeric_cols_for_imputation = df.select_dtypes(include=np.number).columns.tolist()
    if 'bankrupt' in numeric_cols_for_imputation:
        numeric_cols_for_imputation.remove('bankrupt')

    # Imputar NaNs en columnas numéricas con la mediana
    nan_cols_imputed = []
    for col in numeric_cols_for_imputation:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            nan_cols_imputed.append(col)
    
    if nan_cols_imputed:
        print(f"NaNs en {len(nan_cols_imputed)} columnas numéricas rellenados con la mediana.")
    else:
        print("No se encontraron NaNs en columnas numéricas para imputar.")
    
    print(f"Shape después de limpieza y manejo de NaNs/Inf: {df.shape}")

    # 5. Detección y eliminación de características altamente correlacionadas
    # Excluir la variable objetivo del cálculo de correlación
    X_for_correlation = df.drop(columns=['bankrupt']) if 'bankrupt' in df.columns else df.copy()
    
    # Calcular la matriz de correlación de valor absoluto
    corr_matrix = X_for_correlation.corr().abs()
    
    # Seleccionar el triángulo superior de la matriz de correlación para evitar duplicados
    # y comparaciones de una columna consigo misma
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Encontrar columnas con correlación mayor a un umbral (ej. 0.95)
    to_drop_highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    if to_drop_highly_correlated:
        df.drop(columns=to_drop_highly_correlated, inplace=True)
        print(f"Columnas altamente correlacionadas eliminadas (umbral 0.95): {to_drop_highly_correlated}")
    else:
        print("No se encontraron columnas altamente correlacionadas para eliminar.")

    # Guardar las columnas eliminadas por correlación para replicar en nuevos datos
    preprocessor_params = {'columns_dropped_correlation': to_drop_highly_correlated}
    save_object(preprocessor_params, os.path.join(models_dir, 'preprocessor_params.pkl'))
    print(f"Parámetros del preprocesador (columnas correlacionadas eliminadas) guardados en: {os.path.join(models_dir, 'preprocessor_params.pkl')}")
    print(f"Shape después de eliminar columnas correlacionadas: {df.shape}")


    # --- EL PASO CRUCIAL: ESCALADO UNIVERSAL DE TODAS LAS CARACTERÍSTICAS NUMÉRICAS ---
    from sklearn.preprocessing import StandardScaler # Asegúrate de importar esto

    # Identificar todas las columnas que son características (no el target 'bankrupt')
    # y que son de tipo numérico (int, float).
    features_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    if 'bankrupt' in features_to_scale:
        features_to_scale.remove('bankrupt') # Excluir la variable objetivo del escalado

    print(f"\nIniciando escalado de {len(features_to_scale)} características...")

    # Instanciar el escalador.
   
    scaler = StandardScaler() 
    
    # Ajustar el escalador a los datos y transformarlos
    # Esto modificará las columnas del DataFrame in-place
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    print("Todas las características numéricas han sido escaladas.")

    # Guardar el objeto scaler. 
    save_object(scaler, os.path.join(models_dir, 'scaler_for_all_features.pkl'))
    print(f"Scaler universal guardado en: {os.path.join(models_dir, 'scaler_for_all_features.pkl')}")

    # 6. Guardar el DataFrame preprocesado final
    output_path = os.path.join(processed_data_dir, 'data_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDataFrame preprocesado final guardado en: {output_path}. Shape final: {df.shape}")
    print("--- Preprocesamiento de Datos Completado ---")