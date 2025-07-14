# backend/app.py
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = Flask(__name__)

# --- Configuración de Firebase ---
# Ruta al archivo de credenciales de Firebase Admin SDK
# Asegúrate de que este archivo JSON está en backend/credentials/
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials', 'machine-learning-empesas-firebase-adminsdk-fbsvc-f902914f59.json')
 
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
        # Puedes usar un campo único como ID del documento, o dejar que Firestore genere uno.
        # Si la empresa tiene un ID único (ej. CIF/NIF), podrías usarlo:
        # doc_id = company_data.get('cif_nif_column_name') # Asegúrate de que exista y sea único
        # if doc_id:
        #     doc_ref = db.collection('companies_analyzed').document(doc_id)
        # else:
        #     doc_ref = db.collection('companies_analyzed').document() # Firestore genera ID

        # Para simplicidad y si no hay un ID único garantizado, deja que Firestore genere el ID
        doc_ref = db.collection('companies_analyzed').document()
        doc_ref.set(company_data) # Guarda el diccionario directamente como un documento

        print(f"Datos de empresa guardados con ID: {doc_ref.id}")
        return jsonify({"message": "Company data saved successfully", "id": doc_ref.id}), 201

    except Exception as e:
        print(f"Error al guardar datos en Firestore: {e}")
        return jsonify({"error": f"Failed to save data: {e}"}), 500

if __name__ == '__main__':
    # Ejecutar Flask en un puerto diferente al de Streamlit (ej. 5000)
    # y accesible desde cualquier IP para que Streamlit pueda conectarse.
    app.run(debug=True, host='0.0.0.0', port=5000)