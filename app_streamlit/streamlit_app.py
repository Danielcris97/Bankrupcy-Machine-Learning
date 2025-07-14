import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
import json
from groq import Groq # Importar la librería Groq
import requests # Importar la librería requests para hacer llamadas HTTP a Flask

# --- 1. Constantes y Configuración de Rutas ---
# Rutas a los directorios de modelos y reportes
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')

# URL de tu backend Flask para guardar datos
# IMPORTANTE: Si despliegas Flask en un servidor diferente, esta URL deberá cambiar
# Para Docker Compose, usa el nombre del servicio Flask: http://flask_backend:5000/save_company_data
FLASK_BACKEND_SAVE_URL = "http://flask_backend:5000/save_company_data" 

# Características importantes para mostrar en el perfil del clúster
DISPLAY_FEATURES_FOR_CLUSTERS = [
    'roaa_before_interest_and_percent_after_tax',
    'operating_gross_margin',
    'cash_flow_rate',
    'debt_ratio_percent',
    'equity_to_liability',
    'working_capital_to_total_assets',
    'current_liability_to_assets',
    'total_asset_growth_rate',
    'retained_earnings_to_total_assets'
]

# --- 2. Carga Segura de la Clave API de Groq ---
# Accede a la clave API de Groq de forma segura usando st.secrets
# Esto requiere un archivo .streamlit/secrets.toml con [groq] api_key = "tu_clave"
try:
    GROQ_API_KEY_VALUE = st.secrets["groq"]["api_key"]
    client = Groq(api_key=GROQ_API_KEY_VALUE)
except KeyError:
    st.error("Error: La clave API de Groq no se encontró en st.secrets. Asegúrate de configurar .streamlit/secrets.toml.")
    st.stop()
except Exception as e:
    st.error(f"Error al inicializar Groq API. Detalles: {e}")
    st.stop()


# --- 3. Funciones de Carga de Objetos (Cacheadas para Rendimiento) ---
@st.cache_resource
def load_object(file_path: str):
    """
    Carga un objeto usando joblib.
    """
    try:
        obj = joblib.load(file_path)
        return obj
    except FileNotFoundError:
        st.error(f"Error: Archivo no encontrado en '{file_path}'. Por favor, asegúrate de que el archivo exista y la ruta sea correcta.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el objeto desde {file_path}: {e}")
        return None

@st.cache_data
def load_json_data(file_path: str):
    """
    Carga datos desde un archivo JSON.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Advertencia: Archivo JSON no encontrado en '{file_path}'.")
        return {}
    except json.JSONDecodeError:
        st.warning(f"Advertencia: Error al decodificar el archivo JSON en '{file_path}'. Podría estar corrupto o vacío.")
        return {}
    except Exception as e:
        st.warning(f"Error inesperado al cargar el JSON desde {file_path}: {e}")
        return {}


# --- 4. Funciones de Preprocesamiento (Replicando la lógica de data_processing.py) ---

def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza de los nombres de las columnas de un DataFrame.
    """
    new_columns = []
    for col in df.columns:
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
        new_columns.append(col)
    df.columns = new_columns
    return df

def apply_full_preprocessing_pipeline(df_raw: pd.DataFrame, preprocessor_params: dict, scaler: object, expected_features_for_scaler: list) -> pd.DataFrame:
    """
    Aplica el pipeline de preprocesamiento completo a un DataFrame de entrada.
    """
    df_processed = df_raw.copy()

    # 1. Limpiar nombres de columnas
    df_processed = limpiar_nombres_columnas(df_processed)

    # 2. Convertir a numérico, manejar infinitos
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Eliminar 'net_income_flag' si existe
    if 'net_income_flag' in df_processed.columns:
        df_processed.drop(columns=['net_income_flag'], inplace=True)

    # 4. Manejar valores atípicos con clipping
    clipping_bounds = preprocessor_params.get('clipping_bounds', {})
    for col, bounds in clipping_bounds.items():
        if col in df_processed.columns and df_processed[col].dtype in ['float64', 'int64']:
            df_processed[col] = np.clip(df_processed[col], bounds['lower'], bounds['upper'])

    # 5. Imputar valores faltantes con la mediana
    imputation_means = preprocessor_params.get('imputation_means', {})
    for col, mean_val in imputation_means.items():
        if col in df_processed.columns and df_processed[col].isnull().any():
            df_processed[col].fillna(mean_val, inplace=True)
    
    # 6. Eliminar características altamente correlacionadas
    cols_to_drop_corr = preprocessor_params.get('columns_dropped_correlation', [])
    for col in cols_to_drop_corr:
        if col in df_processed.columns:
            df_processed.drop(columns=[col], inplace=True)

    # 7. Asegurar que el DataFrame tiene las columnas esperadas por el scaler y en el orden correcto
    df_final_features = df_processed.reindex(columns=expected_features_for_scaler, fill_value=0.0)

    # Verificación final de NaNs antes de escalar
    if df_final_features.isnull().sum().sum() > 0:
        st.warning("¡Advertencia! Se encontraron valores nulos después del preprocesamiento. Imputando con 0.0 para evitar errores en el escalador.")
        df_final_features.fillna(0.0, inplace=True)

    # 8. Aplicar el escalador
    try:
        scaled_array = scaler.transform(df_final_features)
        df_scaled = pd.DataFrame(scaled_array, columns=expected_features_for_scaler)
    except Exception as e:
        st.error(f"Error al aplicar el escalado a los datos de entrada: {e}")
        st.warning("Verifica que las columnas de entrada coincidan con las esperadas por el escalador.")
        st.write("Columnas esperadas por el escalador:", expected_features_for_scaler)
        st.write("Columnas en la entrada después de preprocesamiento:", df_final_features.columns.tolist())
        st.stop() # Detener la ejecución si el escalado falla

    return df_scaled


# --- 5. Lógica de la Aplicación (Predicción y Análisis) ---

# Definición de los SYSTEM PROMPTS para la IA (tal como los enviaste)
SYSTEM_PROMPT_INITIAL_ANALYSIS = "Eres un experto en finanzas, economía y negocios. Tu función es analizar información financiera y de mercado. Solo puedes responder a preguntas relacionadas con estos temas y analizar la información brindada."
SYSTEM_PROMPT_CHAT = "Eres un experto en finanzas, economía y negocios. Tu función es analizar información financiera y de mercado. Solo puedes responder a preguntas relacionadas con estos temas.Puedes responder algunas preguntas sobre el proyecto ml, sobre el método de predicción y funcionalidad básica del modelo, no debes entrar en detalles técnicos de programación porque no es tu especialidad, los temas más especializados de programación se deben consultar al desarrollador. Si te preguntan sobre cualquier otro tema, debes responder: 'Lo siento, soy un experto en finanzas y no puedo ayudarte con ese tema. '"


def get_llm_response(messages: list, system_prompt: str): # Añadido system_prompt como argumento
    """
    Llama a la API de Groq para obtener una respuesta de texto.
    """
    try:
        # Asegurarse de que el system_prompt se inserta al principio de los mensajes
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        chat_completion = client.chat.completions.create(
            messages=full_messages, # Usar full_messages
            model="llama3-8b-8192", # Puedes cambiar a "llama3-70b-8192" para un modelo más potente
            stream=False,
            temperature=0.7, # Controla la creatividad (0.0 a 2.0)
            max_tokens=700 # Limita la longitud de la respuesta
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error al comunicarse con la API de Groq: {e}")
        return "Análisis no disponible debido a un error de comunicación con la IA."

def generate_initial_analysis_prompt_content(prediction_result, prediction_proba, cluster_id, prop_quiebra_cluster, cluster_profile_data, original_input_df_head):
    """
    Genera el contenido del prompt inicial detallado para el análisis de la IA.
    """
    prompt = f"""
    Aquí tienes información sobre un proyecto de Machine Learning para predecir la quiebra de empresas y los resultados de una empresa específica. El modelo fue entrenado y validado para ser robusto.

    **Contexto del Proyecto de ML:**
    - Objetivo: Predecir la quiebra empresarial minimizando falsos negativos (priorizando Recall).
    - Dataset original de entrenamiento: 6,819 registros, 95 variables financieras.
    - Preprocesamiento: Incluyó limpieza, manejo de NaN/outliers, eliminación de 24 características correlacionadas, estandarización (StandardScaler) a 78 características finales.
    - Modelo de Clasificación Final: RandomForest (Class Weighted).
        - Métricas clave en el conjunto de prueba: Recall (Quiebra) = 0.659, Precision (Quiebra) = 0.246, F1-Score (Quiebra) = 0.358, ROC AUC = 0.898.
        - La elección de este modelo prioriza capturar la mayoría de las quiebras reales, aceptando un mayor número de falsas alarmas, ya que el costo de no detectar una quiebra es más alto.
    - Análisis de Clústeres (KMeans sobre PCA): Identifica 3 perfiles de riesgo distintos:
        - Clúster 0 (Bajo Riesgo): 0.14% de quiebras históricas. Representa empresas muy saludables.
        - Clúster 1 (Riesgo Extremo): 75.00% de quiebras históricas (aunque es un clúster muy pequeño, es una señal de alarma crítica). Representa empresas en crisis severa.
        - Clúster 2 (Riesgo Moderado): 4.52% de quiebras históricas. Representa empresas con un perfil promedio.

    **Información de la Empresa a Analizar:**
    - Predicción del modelo: {'ALTA PROBABILIDAD de QUiebra' if prediction_result == 1 else 'BAJA PROBABILIDAD de Quiebra'}
    - Probabilidad de quiebra: {prediction_proba*100:.2f}%
    - Clúster asignado: {cluster_id}
    - Proporción histórica de quiebras en este clúster: {prop_quiebra_cluster*100:.2f}%
    - Perfil financiero típico del Clúster {cluster_id} (valores escalados para las características clave):
    {json.dumps(cluster_profile_data, indent=2)}
    - Primeros valores del DataFrame de entrada original de la empresa (para contexto):
    {original_input_df_head.to_dict() if not original_input_df_head.empty else "No disponible"}

    Por favor, proporciona un análisis financiero detallado de esta empresa, interpretando su predicción y su pertenencia al clúster. Incluye:
    1.  Un resumen del riesgo general de la empresa y su salud financiera.
    2.  Las implicaciones financieras clave de su predicción y su clúster.
    3.  Recomendaciones o consideraciones clave para un analista de negocios o inversor basadas en este perfil.
    4.  Menciona si hay alguna inconsistencia notable entre la predicción del modelo y el clúster asignado (ej. modelo dice 'Baja Probabilidad' pero está en Clúster 1).
    """
    return prompt


# --- 6. Carga de Artefactos al inicio de la aplicación ---
# Cargar el modelo final (RandomForest_ClassWeighted)
final_model = load_object(os.path.join(MODELS_DIR, 'best_rf_classweighted_model.pkl'))

# Cargar el scaler universal
scaler = load_object(os.path.join(MODELS_DIR, 'scaler_for_all_features.pkl'))

# Cargar las características que el SCALER ESPERA (las 78 features después de correlación y net_income_flag)
scaler_expected_features = load_object(os.path.join(MODELS_DIR, 'scaler_features.pkl'))

# Cargar los parámetros del preprocesamiento (clipping_bounds, imputation_means, columns_dropped_correlation)
preprocessor_params = load_object(os.path.join(MODELS_DIR, 'preprocessor_params.pkl'))

# Cargar los modelos PCA y KMeans
pca_model = load_object(os.path.join(MODELS_DIR, 'pca_model.pkl'))
kmeans_model = load_object(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))

# Cargar perfiles de clústeres para la interpretación
cluster_profiles = load_json_data(os.path.join(REPORTS_DIR, 'cluster_profiles.json'))
bankrupt_proportions = load_json_data(os.path.join(REPORTS_DIR, 'bankrupt_proportion_by_cluster.json'))


# Verificar que todos los recursos necesarios se cargaron correctamente
if not all([final_model, scaler, scaler_expected_features, preprocessor_params, pca_model, kmeans_model, cluster_profiles, bankrupt_proportions]):
    st.error("Error crítico: No se pudieron cargar uno o más recursos necesarios. Asegúrate de que todos los scripts de entrenamiento y análisis se ejecutaron correctamente y los archivos `.pkl` y `.json` están en las rutas esperadas (`/models/` y `/reports/`).")
    st.stop()


# --- 7. Interfaz de Usuario de Streamlit ---
st.set_page_config(page_title="Predicción de Quiebra Empresarial", layout="centered")

st.title("💸 Predicción de Quiebra Empresarial")
st.markdown("""
Esta aplicación predice la probabilidad de quiebra de una empresa basándose en sus datos financieros.
Por favor, sube un archivo CSV con las características de la empresa que deseas analizar.
**El archivo CSV debe contener una sola fila de datos para una empresa y todas las 95 columnas esperadas del dataset original.**
""")

st.subheader("Subir Datos de la Empresa")
uploaded_file = st.file_uploader("Arrastra o selecciona un archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        input_df_raw = pd.read_csv(uploaded_file)
        st.write("Datos cargados exitosamente:")
        st.dataframe(input_df_raw)

        if input_df_raw.shape[0] > 1:
            st.warning("Advertencia: El archivo CSV contiene múltiples filas. Solo se procesará la primera fila para la predicción.")
            input_df_raw = input_df_raw.head(1)
        
        # Guardar una copia de la cabecera del DF original para el prompt del LLM
        original_input_df_head = input_df_raw.iloc[0, :5] # Primeros 5 valores para contexto

        st.subheader("Realizando Predicción y Análisis...")

        # --- Aplicar Pipeline de Preprocesamiento ---
        # df_scaled ahora contiene las 78 características escaladas
        df_scaled = apply_full_preprocessing_pipeline(input_df_raw.copy(), preprocessor_params, scaler, scaler_expected_features)

        # Preparar para el modelo supervisado (RandomForest_ClassWeighted)
        # Asegurarse de que las columnas estén en el orden correcto para el modelo
        processed_input_for_rf = df_scaled.reindex(columns=load_object(os.path.join(MODELS_DIR, 'model_features.pkl')), fill_value=0.0)


        # --- Realizar Predicciones y Análisis ---

        # 1. Predicción de Clasificación (RandomForest_ClassWeighted)
        prediction = None
        prediction_proba = None
        try:
            prediction = final_model.predict(processed_input_for_rf)[0]
            prediction_proba = final_model.predict_proba(processed_input_for_rf)[0, 1] # Probabilidad de quiebra (clase 1)

            st.subheader("Resultados de la Predicción de Quiebra (RandomForest_ClassWeighted)")
            if prediction == 1:
                st.error(f"**PREDICCIÓN: ¡La empresa tiene ALTA PROBABILIDAD de QUiebra!**")
                st.metric(label="Probabilidad de Quiebra", value=f"{prediction_proba*100:.2f}%", delta_color="inverse")
            else:
                st.success(f"**PREDICCIÓN: La empresa tiene BAJA PROBABILIDAD de Quiebra.**")
                st.metric(label="Probabilidad de Quiebra", value=f"{prediction_proba*100:.2f}%", delta_color="normal")

            st.markdown(f"*(Umbral de clasificación por defecto del modelo: > 0.5 para Quiebra)*")
        except Exception as e:
            st.error(f"Error al realizar la predicción del RandomForest_ClassWeighted: {e}")
            st.warning("Asegúrate de que los datos de entrada están preparados correctamente para el modelo.")
            st.write("Columnas esperadas por RandomForest_ClassWeighted:", load_object(os.path.join(MODELS_DIR, 'model_features.pkl')))
            st.write("Columnas en entrada final para RandomForest_ClassWeighted:", processed_input_for_rf.columns.tolist())
            st.stop()


        st.write("---")
        st.subheader("Análisis Adicional (Modelos No Supervisados)")

        # 2. Análisis PCA
        pca_transformed_data = None
        if pca_model:
            try:
                pca_transformed_data = pca_model.transform(df_scaled)
                st.write(f"**Análisis PCA:**")
                st.write(f"La empresa se representa en {pca_transformed_data.shape[1]} componentes principales.")
                st.write(f"Primeros componentes: {pca_transformed_data[0, :5]}")
            except Exception as e:
                st.warning(f"No se pudo aplicar PCA a los datos de entrada. Error: {e}")
                st.write("Columnas en entrada procesada y escalada para PCA:", df_scaled.columns.tolist())
                # No st.stop() aquí para permitir que KMeans intente ejecutarse si PCA falló por otra razón

        # 3. Análisis KMeans
        cluster_id = None
        prop_quiebra_cluster = 0
        cluster_profile_data = {}

        if kmeans_model and pca_transformed_data is not None:
            try:
                kmeans_cluster = kmeans_model.predict(pca_transformed_data)[0]
                cluster_id = str(kmeans_cluster)

                st.write(f"**Análisis KMeans:**")
                st.write(f"La empresa pertenece al clúster: **{cluster_id}**")

                if cluster_id in bankrupt_proportions:
                    prop_quiebra_cluster = bankrupt_proportions[cluster_id].get('1', 0)
                    st.info(f"Este clúster (Clúster {cluster_id}) tuvo una proporción de quiebras del **{prop_quiebra_cluster*100:.2f}%** en los datos de entrenamiento.")

                    if cluster_id in cluster_profiles:
                        st.markdown("**Perfil Financiero Típico de este Clúster (Valores Escalados):**")
                        profile_df = pd.DataFrame([cluster_profiles[cluster_id]])
                        profile_df.index = [f"Clúster {cluster_id} Media"]

                        actual_display_features = [f for f in DISPLAY_FEATURES_FOR_CLUSTERS if f in profile_df.columns]
                        if actual_display_features:
                            st.dataframe(profile_df[actual_display_features].T.style.format("{:.3f}"))
                            st.markdown("""
                            *(**Interpretación de valores escalados:** Un valor positivo significa que la empresa en este clúster tiende a tener un valor más alto para esta característica en comparación con la media de todas las empresas. Un valor negativo significa que tiende a tener un valor más bajo.)*
                            """)
                            cluster_profile_data = {f: profile_df[f].iloc[0] for f in actual_display_features} # Para el LLM
                        else:
                            st.info("No se pudieron mostrar características clave para este clúster (las características seleccionadas no coinciden).")
                    else:
                        st.info("No se encontró el perfil de características para este clúster.")
                else:
                    st.markdown("*(No se encontró información de proporciones de quiebra para este clúster. Puede que el archivo JSON esté incompleto o no se haya generado.)*")

            except Exception as e:
                st.warning(f"No se pudo aplicar KMeans a los datos de entrada. Error: {e}")
                if pca_transformed_data is not None:
                    st.write("Datos pasados a KMeans (primeros 5 valores de las primeras componentes PCA):", pca_transformed_data[0, :5].tolist())
                    st.write("Número de características pasadas a KMeans:", pca_transformed_data.shape[1])
                st.stop()
        else:
            st.warning("Modelo KMeans no disponible o PCA falló.")


        # --- Sección de Análisis por IA (Groq) y Chat Interactivo ---
        st.write("---")
        st.subheader("Análisis de Riesgo Detallado por IA (Groq) y Consultas")

        # Inicializar historial de chat si no existe
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Botón para generar el análisis inicial (solo si no se ha generado ya)
        if not st.session_state.messages:
            if st.button("Generar Análisis de Riesgo con IA"):
                with st.spinner("Generando análisis inicial con IA..."):
                    initial_prompt_content = generate_initial_analysis_prompt_content(
                        prediction, prediction_proba, cluster_id, prop_quiebra_cluster,
                        cluster_profile_data, original_input_df_head
                    )
                    
                    # El primer mensaje al LLM usa el prompt menos restrictivo para el análisis
                    messages_for_llm = [
                        {"role": "system", "content": SYSTEM_PROMPT_INITIAL_ANALYSIS}, 
                        {"role": "user", "content": initial_prompt_content}
                    ]
                    
                    ai_response = get_llm_response(messages_for_llm, SYSTEM_PROMPT_INITIAL_ANALYSIS) # Pasa el prompt de sistema aquí
                    
                    # Añadir la respuesta inicial de la IA al historial
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # Añadir una pregunta inicial de la IA para invitar al diálogo
                    st.session_state.messages.append({"role": "assistant", "content": "¿Tienes alguna duda sobre esta empresa, alguna de las métricas o del proyecto en sí?"})
                    st.rerun() # Recargar para mostrar el chat

        # Mostrar historial de chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Entrada de chat para nuevas preguntas
        if prompt := st.chat_input("Haz una pregunta a la IA experta en finanzas..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("La IA está pensando..."):
                    # Prepara los mensajes para el LLM, incluyendo el contexto del sistema y el historial
                    # Para el chat, usa el prompt más restrictivo
                    messages_to_send = st.session_state.messages # El historial ya tiene los roles correctos

                    ai_response = get_llm_response(messages_to_send, SYSTEM_PROMPT_CHAT) # Pasa el prompt de sistema aquí
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
        # --- Fin de la sección LLM y Chat Interactivo ---

        st.write("---")
        st.subheader("Guardar Datos de Empresa en Base de Datos")
        if st.button("Guardar esta empresa en la Base de Datos"):
            if not input_df_raw.empty:
                # Convertir el DataFrame a un diccionario para enviar a Flask
                # Tomamos la primera (y única) fila y la convertimos a dict
                company_data_to_save = input_df_raw.iloc[0].to_dict()
                
                # --- AÑADIR PREDICCIONES Y CLÚSTER AL DICCIONARIO ---
                company_data_to_save['prediction_bankrupt'] = int(prediction) # 0 o 1
                company_data_to_save['prediction_probability'] = float(prediction_proba) # Probabilidad
                company_data_to_save['assigned_cluster_id'] = cluster_id # ID del clúster
                # --- FIN DE AÑADIR DATOS ---

                try:
                    response = requests.post(FLASK_BACKEND_SAVE_URL, json=company_data_to_save)
                    if response.status_code == 201:
                        st.success(f"Datos de la empresa guardados exitosamente en la base de datos. ID: {response.json().get('id')}")
                    else:
                        st.error(f"Error al guardar datos en la base de datos: {response.status_code} - {response.json().get('error', 'Error desconocido')}")
                except requests.exceptions.ConnectionError:
                    st.error("Error de conexión: Asegúrate de que el servidor Flask esté corriendo en la URL especificada (ej. http://flask_backend:5000).")
                except Exception as e:
                    st.error(f"Ocurrió un error al enviar datos a Flask: {e}")
            else:
                st.warning("No hay datos de empresa cargados para guardar.")


        st.write("---")
        st.subheader("Detalles Técnicos (Datos Procesados y Escalados)")
        st.dataframe(df_scaled)


    except Exception as e:
        st.error(f"Ocurrió un error general durante el procesamiento o la predicción: {e}")
        st.info("Asegúrate de que el archivo CSV tiene el formato correcto, contiene datos numéricos válidos, y sus nombres de columna son consistentes con tus datos de entrenamiento originales (95 columnas).")

st.markdown("---")
st.info("Para más información, consulta el código fuente o contacta al desarrollador.")
