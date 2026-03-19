FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry
COPY pyproject.toml ./
RUN poetry install --no-root
COPY src/ ./src/
CMD ["poetry", "run", "python", "src/main.py"]
