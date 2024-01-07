FROM python:3.11-buster
ENV PYTHONUNBUFFERED=1
WORKDIR /api
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
COPY . .
RUN poetry config virtualenvs.in-project true
RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV AZURE_SEARCH_SERVICE_ENDPOINT=${AZURE_SEARCH_SERVICE_ENDPOINT}
ENV AZURE_SEARCH_API_KEY=${AZURE_SEARCH_API_KEY}
ENTRYPOINT ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--reload"]
