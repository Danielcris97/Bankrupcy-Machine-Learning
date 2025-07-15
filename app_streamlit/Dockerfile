# app_streamlit/Dockerfile
FROM python:3.11

# Crea el directorio de trabajo principal
WORKDIR /app

# Copia el código de Streamlit en su subdirectorio
COPY ./app_streamlit /app/streamlit_code

# Copia modelos y reportes al nivel correcto (para que Streamlit los encuentre)
COPY ./models /app/models
COPY ./reports /app/reports

# Copia e instala dependencias
COPY ./app_streamlit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expone el puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "streamlit_code/app.py", "--server.port=8501", "--server.address=0.0.0.0"]