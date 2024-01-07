イメージのbuild
```
docker compose build
```

```
curl -sSL https://install.python-poetry.org | python3 -
open ~/.zshrc
必要なコードをコピー(パスを通す)
source ~/.zshrc
vscodeを開き直す
```

```
poetry init
poetry add fastapi
poetry add uvicorn
```

```
docker compose run --entrypoint "poetry init --name demo-app --dependency fastapi --dependency uvicorn[standard]" demo-app
docker compose run --entrypoint "poetry install --no-root" demo-app
docker compose build --no-cache
docker compose up
```

```
docker-compose --env-file .env up
```
