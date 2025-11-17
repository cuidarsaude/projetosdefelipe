# projetosdefelipe

Repository for automation projects delivered to Cuidar Saude. The first module published here is the **Codex ChatHub Plugin**, which bridges the Codex agent with both n8n and Flowise.

## Structure

```
projetosdefelipe/
├─ codex_chathub_plugin/
│  ├─ codex_plugin_service.py
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ README.md
│  └─ …
└─ n8n_flow_agent.py
```

- `codex_chathub_plugin/`: FastAPI service compatible with the OpenAI API. It exposes two models to the ChatHub: `codex-flow-builder` (n8n) and `codex-flowise-builder` (Flowise).
- `n8n_flow_agent.py`: CLI helper leveraged by the plugin to generate and publish workflows to n8n via REST.

## Requirements

- Python 3.11+
- Docker (optional, but recommended for deployment)
- n8n with ChatHub enabled
- Flowise (to leverage the Flowise-specific agent)

## Environment variables

`codex_chathub_plugin/.env.example` lists every required variable. Copy it to `.env` and fill in:

- n8n credentials (`N8N_BASE_URL`, `N8N_API_KEY`, `N8N_CHAT_USER`, `N8N_CHAT_PASSWORD`, ...).
- The OpenAI key/model used to draft flows.
- Flowise parameters (`FLOWISE_BASE_URL`, `FLOWISE_API_KEY`, `FLOWISE_CHAT_MODEL_*`, `CODEX_FLOWISE_MODEL_ID`).

## Running locally

```bash
cd codex_chathub_plugin
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn codex_plugin_service:app --reload --port 4010
```

Or via Docker:

```bash
cd codex_chathub_plugin
docker build -t codex-plugin:latest .
docker run -d --name codex-plugin --env-file .env -p 4010:4010 codex-plugin:latest
```

## ChatHub integration

1. Create an “OpenAI” credential in n8n pointing to `http://codex-plugin:4010/v1` and select the `codex-flow-builder` model.
2. Create another credential for Flowise pointing to the same endpoint, but using the `codex-flowise-builder` model.
3. Configure the ChatHub agents so each one uses the proper model (n8n or Flowise).

The “n8n” agent keeps publishing fully executable workflows inside n8n, while the “Flowise” agent generates and saves new chatflows/agentflows directly in Flowise.

## Contact

Questions or suggestions? Open an issue or contact cuidarsaude.ia@gmail.com.
