# Usa una imagen base de Python consistente.
FROM python:3.11-slim-buster

# Establece el directorio de trabajo principal del contenedor.
# Los modelos y reportes estarán en /app/models y /app/reports.
WORKDIR /app

# Copia el archivo de requisitos desde su ubicación correcta en el contexto de construcción
# a un subdirectorio dentro del WORKDIR.
COPY ./app_streamlit/requirements.txt ./streamlit_code/
RUN pip install --no-cache-dir -r ./streamlit_code/requirements.txt

# Copia el resto del código de la aplicación Streamlit a su subdirectorio.
# Esto es principalmente para la construcción de la imagen. En tiempo de ejecución con docker-compose,
# el volumen montado en docker-compose.yml tendrá prioridad y sobrescribirá este contenido.
COPY ./app_streamlit/ ./streamlit_code/

# Expone el puerto por defecto de Streamlit.
EXPOSE 8501

# Comando para ejecutar la aplicación.
# app.py ahora se encuentra en /app/streamlit_code/app.py
ENTRYPOINT ["streamlit", "run", "./streamlit_code/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
