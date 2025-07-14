# Dockerfile en ./app_streamlit/

FROM python:3.11

# Crea el directorio de trabajo
WORKDIR /app/streamlit_code

# Copia el código de Streamlit
COPY ./app_streamlit /app/streamlit_code

# Copia modelos y reportes al nivel esperado
COPY ./models /app/models
COPY ./reports /app/reports

# Instala dependencias
COPY ./app_streamlit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expone el puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py"]
