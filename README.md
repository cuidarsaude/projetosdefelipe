# projetosdefelipe

Repositório com os projetos de automação entregues para a Cuidar Saude. O primeiro módulo publicado aqui é o **Codex ChatHub Plugin**, responsável por integrar o n8n e o Flowise ao agente Codex.

## Estrutura

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

- `codex_chathub_plugin/`: serviço FastAPI compatível com a API da OpenAI, expõe os modelos `codex-flow-builder` (n8n) e `codex-flowise-builder` (Flowise) para o ChatHub.
- `n8n_flow_agent.py`: utilitário usado internamente pelo plugin para gerar e publicar workflows no n8n via API.

## Pré-requisitos

- Python 3.11+
- Docker (opcional, mas recomendado para subir o plugin em container)
- n8n com ChatHub habilitado
- Flowise (para usar o agente dedicado ao Flowise)

## Variáveis de ambiente

O arquivo `codex_chathub_plugin/.env.example` lista todas as variáveis necessárias. Copie-o para `.env` e preencha:

- Credenciais do n8n (`N8N_BASE_URL`, `N8N_API_KEY`, `N8N_CHAT_USER`, `N8N_CHAT_PASSWORD`...).
- OpenAI / modelo utilizado internamente.
- Parâmetros do Flowise (`FLOWISE_BASE_URL`, `FLOWISE_API_KEY`, `FLOWISE_CHAT_MODEL_*`, `CODEX_FLOWISE_MODEL_ID`).

## Executando o plugin localmente

```bash
cd codex_chathub_plugin
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn codex_plugin_service:app --reload --port 4010
```

Ou via Docker:

```bash
cd codex_chathub_plugin
docker build -t codex-plugin:latest .
docker run -d --name codex-plugin --env-file .env -p 4010:4010 codex-plugin:latest
```

## Integração com o ChatHub

1. Crie uma credencial "OpenAI" no n8n apontando para `http://codex-plugin:4010/v1` e selecione o modelo `codex-flow-builder`.
2. Crie outra credencial para o Flowise apontando para o mesmo endpoint, mas usando o modelo `codex-flowise-builder`.
3. Configure os agentes do ChatHub para usar cada modelo conforme necessário.

O agente "n8n" continua publicando workflows completos diretamente no n8n, enquanto o agente "Flowise" gera e salva novos chatflows/agentflows dentro do Flowise.

## Contato

Dúvidas ou melhorias: abrir uma issue ou falar com cuidarsaude.ia@gmail.com.
