# Codex ChatHub Plugin

Plugin compatível com a API do OpenAI que permite usar o ChatHub do n8n para
solicitar a criação automática de workflows. O serviço expõe os endpoints
`/v1/models` e `/v1/chat/completions`, interpreta os prompts recebidos,
gera o workflow via `n8n_flow_agent` e publica o fluxo usando a API REST
do n8n. A resposta enviada de volta para o ChatHub confirma a criação do
workflow e informa o ID para consulta.

## Estrutura

```
plugins/codex_chathub/
├─ codex_plugin_service.py  # FastAPI com compatibilidade OpenAI
├─ requirements.txt         # Dependências Python
├─ Dockerfile               # Build da imagem (porta 4010)
└─ .env.example             # Variáveis necessárias
```

### Variáveis de ambiente

| Nome                  | Descrição                                            |
|-----------------------|------------------------------------------------------|
| `N8N_BASE_URL`        | URL pública do n8n (ex.: `https://dominio.trycloudflare.com`) |
| `N8N_API_KEY`         | API key do n8n com permissão para criar workflows    |
| `N8N_WORKFLOWS_ENDPOINT` | Endpoint REST (normalmente `/api/v1/workflows`)  |
| `N8N_VERIFY_SSL`      | Defina `false` quando estiver usando Cloudflare túnel |
| `OPENAI_API_KEY`      | API key oficial usada para gerar o workflow          |
| `OPENAI_MODEL`        | Modelo do OpenAI (`gpt-4o-mini`, `gpt-4.1-mini`, etc.) |
| `OPENAI_TEMPERATURE`  | Temperatura usada na geração do fluxo                |
| `CODEX_PLUGIN_MODEL_ID` | Identificador exposto ao ChatHub (`codex-flow-builder`) |

## Build e execução via Docker

1. Copie `.env.example` para `.env` e preencha os valores reais.
2. Faça o build da imagem:
   ```bash
   docker build -t codex-plugin:latest plugins/codex_chathub
   ```
3. Suba o container (expondo a porta 4010 e conectando-o à mesma rede do n8n):
   ```bash
   docker run -d \
     --name codex-plugin \
     --restart unless-stopped \
     --env-file /path/para/.env \
     --network n8n_default \
     -p 4010:4010 \
     codex-plugin:latest
   ```

Após o serviço estar rodando, crie uma credencial `OpenAI` no n8n apontando o
`Base URL` para `http://codex-plugin:4010/v1` (a chave pode ser qualquer
valor) e selecione o modelo `codex-flow-builder`. Ao criar um agente no
ChatHub usando essa credencial, cada mensagem enviada vai disparar a geração
de um workflow no n8n automaticamente.

Envie `manual` ou `ajuda` no chat para receber uma lista das ações suportadas
e exemplos de uso diretamente na conversa.*** End Patch
