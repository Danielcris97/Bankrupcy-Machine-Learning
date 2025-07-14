# Usa una imagen base de Python ligera
FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación Streamlit
COPY . .

# Copia las carpetas de modelos y reportes desde la raíz del proyecto
# Asegúrate de que las rutas relativas sean correctas desde donde se ejecutará el build context (la raíz del proyecto)
# Cuando construyas la imagen de Streamlit, el contexto de construcción debe ser la raíz de tu proyecto.
# Si el Dockerfile está en app_streamlit, el contexto DEBE ser app_streamlit.
# Por lo tanto, necesitamos ajustar las rutas para acceder a 'models' y 'reports'
# Asumiremos que el contexto de construcción es `app_streamlit/`
# Por lo tanto, las carpetas `models` y `reports` están "fuera" de este contexto.
# Para simplificar, copiaremos las carpetas `models` y `reports` directamente en el Dockerfile principal del proyecto,
# o las montaremos como volúmenes.
# Para este Dockerfile específico de Streamlit, si está en app_streamlit/,
# y models/ y reports/ están en la raíz del proyecto, necesitarás copiarlos desde un nivel superior.
# La forma más sencilla es copiar todo el proyecto a /app y luego trabajar desde allí.

# Vamos a asumir que el contexto de construcción es la raíz del proyecto,
# y que este Dockerfile se copiará a la raíz temporalmente para el build.
# O, si el Dockerfile está en app_streamlit, necesitamos copiar desde ../models y ../reports

# Opción 1: Si el Dockerfile está en app_streamlit y el contexto de construcción es app_streamlit
# Esto requiere que models y reports estén en el mismo nivel que app_streamlit
# Si models y reports están en la raíz del proyecto, necesitas copiarlos así:
# COPY ../models ./models
# COPY ../reports ./reports

# Para simplificar y hacer el Dockerfile más robusto, vamos a estructurarlo para que
# el contexto de construcción sea la raíz del proyecto y luego copiemos las subcarpetas.

# --- REVISIÓN: El Dockerfile debe estar en la raíz del contexto de construcción ---
# Para que esto funcione bien, el Dockerfile de Streamlit debería estar en la raíz de tu proyecto,
# o bien, el comando docker build debe especificar el contexto correctamente.
# La forma más limpia es tener un Dockerfile por servicio en su propia carpeta,
# y luego usar docker-compose para construir desde la raíz del proyecto.

# Vamos a asumir que este Dockerfile se construirá desde la raíz del proyecto,
# y que los archivos se copiarán desde sus subcarpetas.

# Copia los archivos de la aplicación Streamlit
# Esto copiará app_streamlit/app.py, app_streamlit/requirements.txt, etc.
COPY app_streamlit/ .

# Copia los modelos y reportes (asumiendo que están en la raíz del proyecto)
COPY models/ ./models/
COPY reports/ ./reports/

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]