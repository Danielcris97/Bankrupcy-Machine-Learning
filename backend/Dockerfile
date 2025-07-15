# backend/Dockerfile
FROM python:3.11

WORKDIR /app

# Copia todo el código del backend
COPY ./backend /app

# Incluye credenciales dentro de la imagen
# Aunque docker-compose.yml montará el volumen, esta línea es útil
# si construyes la imagen por separado (ej. para Render) y no usas un volumen.
COPY ./backend/credentials /app/credentials

# Instala dependencias
COPY ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5000

# Comando para ejecutar Gunicorn (servidor WSGI para Flask)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]