# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей с полным Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    wget \
    curl \
    gnupg \
    git \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Проверка версий
RUN node --version && npm --version

# Установка N8N через npm с конкретной версией
RUN npm install -g n8n@1.33.0

# Копирование зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . .

# Создание директорий
RUN mkdir -p \
    models \
    data \
    n8n_workflows \
    n8n_data \
    logs

# Экспорт портов
EXPOSE 8000 5678

# Скрипт запуска
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]