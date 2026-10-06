FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying the app to reuse Docker's build cache.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .

# The run command's -p flag publishes this container port on the host.
EXPOSE 8000

# Listen on every container interface so the published port reaches the app.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
