FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install poetry && poetry install --no-root
COPY src/ ./src/
CMD ["poetry", "run", "python", "src/main.py"]
