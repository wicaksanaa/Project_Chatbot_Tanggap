# Gunakan image dasar python
FROM python:3.11

# Set environment variable
ENV MODEL True

# Set working directory
ENV APP_HOME /back-end
WORKDIR $APP_HOME

# Copy semua file ke working directory
COPY . ./

# Upgrade pip dan install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port 8080
EXPOSE 8080

# Jalankan aplikasi
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
