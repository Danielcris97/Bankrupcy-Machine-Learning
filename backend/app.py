# backend/app.py
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json # Importar json para la depuración

app = Flask(__name__)

# --- Configuración de Firebase ---
# Ruta al archivo de credenciales de Firebase Admin SDK
# Asegúrate de que este archivo JSON está en backend/credentials/
# Verifica que 'machine-learning-empesas-firebase-adminsdk-fbsvc-f902914f59.json' es el nombre EXACTO
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials', 'machine-learning-empesas-firebase-adminsdk-fbsvc-f902914f59.json')
 
# --- DEPURACIÓN: Añadir estas líneas ---
print(f"DEBUG: Intentando cargar credenciales desde: {CREDENTIALS_PATH}")
if os.path.exists(CREDENTIALS_PATH):
    print(f"DEBUG: El archivo de credenciales EXISTE en la ruta especificada.")
    try:
        with open(CREDENTIALS_PATH, 'r') as f:
            content = f.read(100) # Leer solo los primeros 100 caracteres para no exponer el secreto
        print(f"DEBUG: Contenido inicial del archivo: {content}...")
        # Intenta parsear el JSON para ver si es válido
        json.loads(content + '{}') # Agrega '}' para hacer un JSON válido si solo leíste el inicio
        print("DEBUG: El archivo parece ser un JSON válido.")
    except json.JSONDecodeError:
        print("DEBUG: ADVERTENCIA: El archivo NO es un JSON válido o está incompleto.")
    except Exception as file_read_e:
        print(f"DEBUG: Error al leer/inspeccionar el archivo: {file_read_e}")
else:
    print(f"DEBUG: ERROR: El archivo de credenciales NO EXISTE en la ruta: {CREDENTIALS_PATH}")
# --- FIN DE DEPURACIÓN ---

# Inicializar la app de Firebase
try:
    cred = credentials.Certificate(CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase inicializado correctamente.")
except FileNotFoundError:
    print(f"Error: Archivo de credenciales no encontrado en '{CREDENTIALS_PATH}'.")
    print("Asegúrate de haber descargado el archivo JSON de Firebase y de haberlo colocado en la ruta correcta.")
    exit(1) # Salir si las credenciales no se encuentran
except Exception as e:
    print(f"Error al inicializar Firebase: {e}")
    exit(1) # Salir si hay otro error de inicialización


@app.route('/')
def home():
    return "Backend de Predicción de Quiebra - ¡Funcionando!"

@app.route('/save_company_data', methods=['POST'])
def save_company_data():
    """
    Endpoint para recibir y guardar datos de una empresa en Firestore.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    company_data = request.get_json()

    if not company_data:
        return jsonify({"error": "No data provided"}), 400

    try:
        doc_ref = db.collection('companies_analyzed').document()
        doc_ref.set(company_data) # Guarda el diccionario directamente como un documento

        print(f"Datos de empresa guardados con ID: {doc_ref.id}")
        return jsonify({"message": "Company data saved successfully", "id": doc_ref.id}), 201

    except Exception as e:
        print(f"Error al guardar datos en Firestore: {e}")
        return jsonify({"error": f"Failed to save data: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)