FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -r <(pipx run poetry export -f requirements.txt --without-hashes) || pip install --no-cache-dir fastapi uvicorn requests pydantic
COPY app/ /app/app/
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
