# Dockerfile en ./backend/

FROM python:3.11

WORKDIR /app

# Copia todo el código del backend
COPY ./backend /app

# Incluye credenciales dentro de la imagen
COPY ./backend/credentials /app/credentials

# Instala dependencias
COPY ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
