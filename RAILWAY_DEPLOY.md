# Deploy no Railway

Este projeto esta preparado para deploy como app Python/FastAPI no Railway.

## Configuracao

O Railway deve detectar o projeto via Nixpacks usando:

- `nixpacks.toml` para forcar build Python
- `requirements.txt` para instalar dependencias
- `railway.json` para start command e healthcheck

## Start command

```bash
python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers
```

O Railway define a variavel `PORT` automaticamente.

## Healthcheck

```text
/health
```

## Arquivos ignorados no deploy

`.railwayignore` remove Android, Node/Capacitor e arquivos auxiliares do contexto de deploy. O container precisa apenas de:

- API FastAPI
- `www/`
- banco V3
- modelo `DorOrofacialAI`
- dependencias Python
