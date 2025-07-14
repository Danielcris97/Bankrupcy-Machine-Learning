# Usa una imagen base de Python ligera
FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos e instálalos
# La ruta es relativa al contexto de construcción (la raíz del proyecto)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación Flask y la carpeta de credenciales
# La ruta es relativa al contexto de construcción
COPY backend/ .

# Expone el puerto por defecto de Flask
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
# Usa Gunicorn para producción, pero para desarrollo Flask es suficiente.
# Para producción: CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
CMD ["python", "app.py"]
