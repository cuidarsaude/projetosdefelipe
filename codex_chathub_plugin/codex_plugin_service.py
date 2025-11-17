import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from n8n_flow_agent import AgentConfig, call_llm, deploy_workflow

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("codex-plugin")

env_path = Path(__file__).resolve().parent.parent.parent / ".env.n8n"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

try:
    AGENT_CONFIG = AgentConfig.from_env()
except Exception as exc:  # pragma: no cover - configuration issues must stop startup
    logger.error("Falha ao carregar configuracao do agente: %s", exc)
    raise

PLUGIN_MODEL_ID = os.getenv("CODEX_PLUGIN_MODEL_ID", "codex-flow-builder")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
CHAT_USER = os.getenv("N8N_CHAT_USER", os.getenv("N8N_EMAIL", ""))
CHAT_PASSWORD = os.getenv("N8N_CHAT_PASSWORD", os.getenv("N8N_PASSWORD", ""))
PROJECT_ID = os.getenv("N8N_PROJECT_ID", "oceyE1aX1AtYkrAd")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY n�o definido nas vari�veis de ambiente.")
if not CHAT_USER or not CHAT_PASSWORD:
    raise RuntimeError("Defina N8N_CHAT_USER e N8N_CHAT_PASSWORD para executar a��es administrativas.")

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

FLOWISE_MODEL_ID = os.getenv("CODEX_FLOWISE_MODEL_ID", "codex-flowise-builder")
FLOWISE_BASE_URL = os.getenv("FLOWISE_BASE_URL")
FLOWISE_PUBLIC_URL = os.getenv("FLOWISE_PUBLIC_URL", FLOWISE_BASE_URL or "")
FLOWISE_API_KEY = os.getenv("FLOWISE_API_KEY")
FLOWISE_CHAT_MODEL_NAME = os.getenv("FLOWISE_CHAT_MODEL_NAME", "chatOpenAI")
FLOWISE_CHAT_MODEL_VERSION = float(os.getenv("FLOWISE_CHAT_MODEL_VERSION", "8.3"))
FLOWISE_CHAT_MODEL_TEMPERATURE = os.getenv("FLOWISE_CHAT_MODEL_TEMPERATURE", "0.2")
FLOWISE_CHAT_MODEL_CREDENTIAL_ID = os.getenv("FLOWISE_CHAT_MODEL_CREDENTIAL_ID")
FLOWISE_ENABLED = bool(
    FLOWISE_BASE_URL and FLOWISE_API_KEY and FLOWISE_CHAT_MODEL_CREDENTIAL_ID
)

HELP_KEYWORDS = (
    "manual",
    "ajuda",
    "help",
    "como usar",
    "como funciona",
    "o que voce faz",
    "o que você faz",
    "quais acoes",
    "quais ações",
    "instrucoes",
    "instruções",
    "documentacao",
    "documentação",
)

N8N_MANUAL_TEXT = (
    "### Manual do agente Codex Flow Builder (n8n)\n"
    "\n"
    "**O que ele faz**\n"
    "- Cria workflows n8n completos a partir de prompts em texto livre.\n"
    "- Cria pastas no projeto atual.\n"
    "- Move workflows entre pastas ou para a raiz.\n"
    "- Arquiva ou restaura workflows existentes.\n"
    "\n"
    "**Exemplos de pedidos**\n"
    "1. `Crie um workflow chamado Monitor Leads com webhook, set e HTTP Request.`\n"
    "2. `Crie uma pasta Atendimento IA dentro de Operacoes`.\n"
    "3. `Mova o workflow Monitor Leads para a pasta Atendimento IA`.\n"
    "4. `Arquive o workflow Lead Antigo` ou `Restaure o workflow Lead Antigo`.\n"
    "\n"
    "**Boas práticas**\n"
    "- Detalhe entradas, saídas e integrações (URLs, métodos, campos).\n"
    "- Cite nomes exatos de workflows/pastas ao mover ou arquivar.\n"
    "- Informe credenciais já criadas no n8n quando necessárias.\n"
    "- Peça ajustes adicionais descrevendo o que mudar.\n"
    "\n"
    "Envie `manual` ou `ajuda` para reabrir estas instruções."
)

FLOWISE_MANUAL_TEXT = (
    "### Manual do agente Codex Flow Builder (Flowise)\n"
    "\n"
    "**O que ele faz**\n"
    "- Gera novos chatflows/agentflows dentro do Flowise usando o modelo selecionado.\n"
    "- Preenche nodes automaticamente (Start, Agent, HTTP, Banco de Memória etc.).\n"
    "- Lista rapidamente os flows existentes quando solicitado.\n"
    "\n"
    "**Como pedir**\n"
    "1. `Crie um Flowise para capturar leads e enviar via HTTP POST...`\n"
    "2. `Monte um agente com RAG que consulte a base XYZ.`\n"
    "3. `Liste os flows ativos` para visualizar IDs.\n"
    "\n"
    "**Notas**\n"
    "- Use frases objetivas descrevendo entradas, integrações e destino.\n"
    "- Após criar, abra o link fornecido para ajustar nodes no Flowise.\n"
    "- Por enquanto, este agente foca em criação e listagem; exclusões/edições complexas devem ser feitas manualmente.\n"
    "\n"
    "Envie `manual` ou `ajuda` para reabrir estas instruções."
)

class Message(BaseModel):
    role: str
    content: Union[str, List[Union[str, dict]]]

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        parts: List[str] = []
        for chunk in self.content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                if "text" in chunk and isinstance(chunk["text"], str):
                    parts.append(chunk["text"])
                else:
                    parts.append(json.dumps(chunk))
        return "\n".join(parts)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    user: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: dict
    finish_reason: str = Field(default="stop")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: List[Choice]
    usage: dict


def _latest_user_text(messages: List["Message"]) -> str:
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.as_text().strip()
    return ""


def _should_show_manual(messages: List["Message"]) -> bool:
    if not messages:
        return False
    text = _latest_user_text(messages).lower()
    if not text:
        return False
    if text in {"manual", "ajuda", "help"}:
        return True
    return any(keyword in text for keyword in HELP_KEYWORDS)


def _build_chat_response(model: str, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message={"role": "assistant", "content": content})],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def _flowise_headers() -> Dict[str, str]:
    if not FLOWISE_ENABLED:
        raise HTTPException(status_code=400, detail="Integração com Flowise não configurada.")
    return {
        "Authorization": f"Bearer {FLOWISE_API_KEY}",
        "Content-Type": "application/json",
    }


def _flowise_request(method: str, path: str, **kwargs) -> Dict[str, object]:
    headers = kwargs.pop("headers", {})
    headers.update(_flowise_headers())
    url = f"{FLOWISE_BASE_URL.rstrip('/')}{path}"
    response = requests.request(method, url, headers=headers, timeout=kwargs.pop("timeout", 60), **kwargs)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Erro ao falar com Flowise ({response.status_code}): {response.text}",
        )
    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=f"Flowise retornou dados inválidos: {exc}") from exc


def _flowise_chat_model_template() -> Dict[str, object]:
    return {
        "id": "chatOpenAI_codex",
        "label": "ChatOpenAI",
        "version": FLOWISE_CHAT_MODEL_VERSION,
        "name": FLOWISE_CHAT_MODEL_NAME,
        "type": "ChatOpenAI",
        "baseClasses": ["ChatOpenAI", "BaseChatModel", "BaseLanguageModel"],
        "category": "Chat Models",
        "description": "Wrapper around OpenAI chat endpoints",
        "inputParams": [],
        "inputAnchors": [],
        "outputAnchors": [],
        "outputs": {},
        "inputs": {
            "modelName": os.getenv("FLOWISE_CHAT_MODEL", "gpt-4o-mini"),
            "temperature": FLOWISE_CHAT_MODEL_TEMPERATURE,
        },
        "credential": FLOWISE_CHAT_MODEL_CREDENTIAL_ID,
    }


def _flowise_generate_agentflow(question: str) -> Dict[str, object]:
    payload = {
        "question": question.strip(),
        "selectedChatModel": _flowise_chat_model_template(),
    }
    return _flowise_request("POST", "/api/v1/agentflowv2-generator/generate", json=payload)


def _flowise_list_chatflows(limit: int = 10) -> str:
    data = _flowise_request("GET", "/api/v1/chatflows")
    if not data:
        return "Não encontrei chatflows no Flowise."

    lines: List[str] = []
    for item in data[:limit]:
        lines.append(f"- **{item.get('name')}** (`{item.get('id')}`)")
    if len(data) > limit:
        lines.append(f"\nMostrando {limit} de {len(data)} flows.")
    return "\n".join(lines)


def _flowise_find_flow_by_name(name: str) -> Optional[Dict[str, object]]:
    if not name:
        return None
    normalized = name.strip().lower()
    if not normalized:
        return None
    data = _flowise_request("GET", "/api/v1/chatflows")
    for item in data:
        candidate = (item.get("name") or "").strip().lower()
        if candidate == normalized:
            return item
    return None


def _flowise_flowdata_payload(nodes: List[Dict[str, object]], edges: List[Dict[str, object]]) -> str:
    canvas = {
        "nodes": nodes,
        "edges": edges,
        "viewport": {"x": 0, "y": 0, "zoom": 0.9},
    }
    return json.dumps(canvas, ensure_ascii=False)


def _sanitize_name_fragment(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text or "")
    words = [w.capitalize() for w in cleaned.split() if w]
    if not words:
        return "Codex Flow"
    return " ".join(words[:6])


def _flowise_suggest_name(prompt: str) -> str:
    base = _sanitize_name_fragment(prompt)
    return base if base else "Codex Flow"


def _flowise_public_link(flow_id: str) -> str:
    if FLOWISE_PUBLIC_URL:
        base = FLOWISE_PUBLIC_URL.rstrip("/")
        return f"{base}/chatflows/{flow_id}"
    if FLOWISE_BASE_URL:
        base = FLOWISE_BASE_URL.rstrip("/")
        return f"{base}/chatflows/{flow_id}"
    return flow_id


def _flowise_create_or_update_flow(
    name: str,
    nodes: List[Dict[str, object]],
    edges: List[Dict[str, object]],
) -> Dict[str, object]:
    payload = {
        "name": name,
        "flowData": _flowise_flowdata_payload(nodes, edges),
        "type": "AGENTFLOW",
        "deployed": False,
        "isPublic": False,
    }
    existing = _flowise_find_flow_by_name(name)
    if existing:
        response = _flowise_request("PUT", f"/api/v1/chatflows/{existing['id']}", json=payload)
        response["__action__"] = "atualizado"
        return response
    response = _flowise_request("POST", "/api/v1/chatflows", json=payload)
    response["__action__"] = "criado"
    return response


def _handle_flowise_completion(messages: List["Message"]) -> str:
    if not FLOWISE_ENABLED:
        raise HTTPException(status_code=400, detail="Integração com Flowise não está configurada.")

    latest_text = _latest_user_text(messages)
    lowered = (latest_text or "").lower()
    if lowered and ("list" in lowered or "listar" in lowered or "liste" in lowered) and (
        "flow" in lowered or "chatflow" in lowered or "workflow" in lowered
    ):
        listings = _flowise_list_chatflows()
        return f"Últimos chatflows:\n{listings}"

    question = _compose_prompt(messages)
    response = _flowise_generate_agentflow(question)
    if "error" in response:
        raise HTTPException(status_code=502, detail=f"Flowise não conseguiu gerar um fluxo: {response['error']}")
    nodes = response.get("nodes") or []
    edges = response.get("edges") or []
    if not isinstance(nodes, list) or not nodes:
        raise HTTPException(status_code=502, detail="Flowise não retornou nodes válidos.")

    name = _flowise_suggest_name(latest_text or question)
    created = _flowise_create_or_update_flow(name, nodes, edges)
    flow_id = created.get("id", "desconhecido")
    action = created.get("__action__", "criado")
    link = _flowise_public_link(flow_id)
    logger.info("Flowise flow %s (%s) %s.", flow_id, name, action)
    return (
        f"Flowise **{name}** {action} (ID `{flow_id}`).\n"
        f"Acesse {link} para abrir no editor e ajustar detalhes ou credenciais."
    )


class FolderRequest(BaseModel):
    name: str
    parentFolderId: Optional[str] = Field(default=None, alias="parentFolderId")
    parentFolderName: Optional[str] = Field(default=None, alias="parentFolderName")


class WorkflowSelector(BaseModel):
    workflowId: Optional[str] = None
    workflowName: Optional[str] = None


class MoveWorkflowRequest(BaseModel):
    workflows: List[WorkflowSelector]
    targetFolderId: Optional[str] = None
    targetFolderName: Optional[str] = None
    toRoot: bool = False


class ArchiveWorkflowRequest(BaseModel):
    workflows: List[WorkflowSelector]
    action: str = Field(default="archive")


ACTION_SCHEMA = {
    "name": "codex_tool_plan",
    "schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_workflow", "create_folder", "move_workflow", "archive_workflow"],
            },
            "arguments": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parentFolderId": {"type": "string"},
                    "parentFolderName": {"type": "string"},
                    "workflows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "workflowId": {"type": "string"},
                                "workflowName": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "targetFolderId": {"type": "string"},
                    "targetFolderName": {"type": "string"},
                    "toRoot": {"type": "boolean"},
                    "action": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


app = FastAPI(
    title="Codex ChatHub Plugin",
    version="1.0.0",
    description="Plugin OpenAI-compatÃ­vel para o ChatHub gerar workflows n8n automaticamente.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class N8NRestClient:
    def __init__(self, base_url: str, user: str, password: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.user = user
        self.password = password
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self._logged_in = False

    def _ensure_login(self) -> None:
        if self._logged_in:
            return
        response = self.session.post(
            f"{self.base_url}/rest/login",
            json={"emailOrLdapLoginId": self.user, "password": self.password},
            timeout=30,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        self._logged_in = True

    def request(self, method: str, path: str, *, expect_json: bool = True, **kwargs) -> Dict[str, object]:
        timeout = kwargs.pop("timeout", 45)
        self._ensure_login()
        url = f"{self.base_url}{path}"
        response = self.session.request(
            method=method,
            url=url,
            timeout=timeout,
            verify=self.verify_ssl,
            **kwargs,
        )
        if response.status_code == 401:
            self._logged_in = False
            self._ensure_login()
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout,
                verify=self.verify_ssl,
                **kwargs,
            )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = getattr(exc.response, "text", str(exc))
            logger.error("Falha ao chamar %s %s: %s", method, url, detail)
            raise HTTPException(status_code=500, detail=f"Falha ao chamar n8n: {detail}")
        if not expect_json or not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}


REST_CLIENT = N8NRestClient(
    base_url=AGENT_CONFIG.n8n_base_url,
    user=CHAT_USER,
    password=CHAT_PASSWORD,
    verify_ssl=AGENT_CONFIG.verify_ssl,
)


def _workflow_api_headers() -> Dict[str, str]:
    return {
        "X-N8N-API-KEY": AGENT_CONFIG.n8n_api_key,
        "Content-Type": "application/json",
    }


def _workflow_api_url(workflow_id: Optional[str] = None, action: Optional[str] = None) -> str:
    base = f"{AGENT_CONFIG.n8n_base_url}{AGENT_CONFIG.workflows_endpoint}".rstrip("/")
    if workflow_id:
        base = f"{base}/{workflow_id}"
    if action:
        base = f"{base}/{action.lstrip('/')}"
    return base


def _extract_response_text(response) -> str:
    output = getattr(response, "output", None) or []
    for block in output:
        for item in getattr(block, "content", []) or []:
            content = getattr(item, "text", None)
            if isinstance(content, str) and content.strip():
                return content
    # fallback for SDKs returning plain string
    if hasattr(response, "output_text"):
        joined = "".join(getattr(response, "output_text"))
        if joined.strip():
            return joined
    raise ValueError("Não consegui extrair o texto estruturado da resposta do modelo.")


def _plan_action(prompt: str) -> Dict[str, object]:
    instructions = (
        "Você analisa o pedido do usuário e decide qual ação executar: "
        "create_workflow (para gerar um novo workflow n8n completo), "
        "create_folder (criar/organizar pastas), move_workflow (mover workflows existentes) "
        "ou archive_workflow (arquivar/restaurar). "
        "Preencha o campo 'arguments' com os dados necessários (nomes/IDs). "
        "Quando o usuário pedir apenas para criar uma pasta, escolha create_folder; "
        "se ele quiser mover algo, use move_workflow, etc. "
        "Retorne somente JSON."
    )
    completion = OPENAI_CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=min(OPENAI_TEMPERATURE, 0.4),
        response_format={"type": "json_schema", "json_schema": ACTION_SCHEMA},
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
    )
    raw_text = completion.choices[0].message.content or ""
    try:
        plan = json.loads(raw_text)
        if not isinstance(plan, dict):
            raise ValueError("Plano inválido.")
        return plan
    except Exception as exc:
        logger.error("Falha ao interpretar plano de ação: %s", exc)
        return {"action": "create_workflow", "arguments": {}}


def _format_tool_result(result: Dict[str, object]) -> str:
    message = result.get("message")
    if message:
        return str(message)
    return json.dumps(result, ensure_ascii=False)


def _compose_prompt(messages: List[Message]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.role.upper()
        content = msg.as_text().strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_workflow(prompt: str) -> str:
    workflow = call_llm(
        prompt=prompt,
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=OPENAI_TEMPERATURE,
    )
    workflow_name = (workflow.get("name") or "Fluxo sem nome").strip() or "Fluxo sem nome"
    existing = _find_workflow_by_name(workflow_name)
    created: Dict[str, object]
    action_label: str
    if existing:
        workflow_id = existing.get("id", "desconhecido")
        created = _update_workflow_via_api(workflow_id, workflow)
        action_label = "atualizado"
        logger.info("Workflow %s (%s) atualizado via plugin.", workflow_id, workflow_name)
    else:
        created = deploy_workflow(AGENT_CONFIG, workflow, activate=False)
        workflow_id = created.get("id", "desconhecido")
        action_label = "criado"
        logger.info("Workflow %s (%s) criado via plugin.", workflow_id, workflow_name)

    activation_error = _activate_workflow_via_api(workflow_id)
    url_hint = (
        f"{AGENT_CONFIG.n8n_base_url.rstrip('/')}/workflow/{workflow_id}"
        if workflow_id != "desconhecido"
        else AGENT_CONFIG.n8n_base_url
    )
    if activation_error:
        status_line = (
            f"Workflow **{workflow_name}** {action_label} (ID `{workflow_id}`), "
            f"porém a ativação automática falhou ({activation_error}). "
            f"Abra {url_hint} para revisar e ativar manualmente."
        )
    else:
        status_line = (
            f"Workflow **{workflow_name}** {action_label} e ativado com sucesso (ID `{workflow_id}`). "
            f"Revise em {url_hint} caso precise de ajustes."
        )
    return status_line


@app.get("/health")
def health():
    return {"status": "ok", "model": PLUGIN_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": PLUGIN_MODEL_ID,
                "object": "model",
                "owned_by": "codex-plugin",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def create_completion(payload: ChatCompletionRequest):
    supported_models = {PLUGIN_MODEL_ID, FLOWISE_MODEL_ID}
    if payload.model not in supported_models:
        raise HTTPException(status_code=400, detail=f"Modelo {payload.model} não suportado.")

    if not payload.messages:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem fornecida.")

    if _should_show_manual(payload.messages):
        manual = FLOWISE_MANUAL_TEXT if payload.model == FLOWISE_MODEL_ID else N8N_MANUAL_TEXT
        return _build_chat_response(payload.model, manual)

    if payload.model == FLOWISE_MODEL_ID:
        answer = _handle_flowise_completion(payload.messages)
        return _build_chat_response(payload.model, answer)

    prompt = _compose_prompt(payload.messages)
    logger.debug("Prompt recebido: %s", prompt.replace("\n", " | "))

    plan = _plan_action(prompt)
    logger.info("Plano selecionado: %s", plan.get("action"))

    try:
        answer = _execute_plan(plan, prompt)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Falha ao executar plano %s: %s", plan, exc)
        answer = f"Não consegui concluir a ação solicitada: {exc}"

    return _build_chat_response(payload.model, answer)


def _list_folders() -> List[Dict[str, object]]:
    response = REST_CLIENT.request("GET", f"/rest/projects/{PROJECT_ID}/folders")
    return response.get("data", [])


def _list_workflows() -> List[Dict[str, object]]:
    response = REST_CLIENT.request("GET", "/rest/workflows", params={"limit": 250})
    return response.get("data", [])


def _find_workflow_by_name(name: Optional[str]) -> Optional[Dict[str, object]]:
    if not name:
        return None
    normalized = name.strip().lower()
    if not normalized:
        return None
    return next(
        (wf for wf in _list_workflows() if (wf.get("name") or "").strip().lower() == normalized),
        None,
    )


def _update_workflow_via_api(workflow_id: str, workflow: Dict[str, object]) -> Dict[str, object]:
    allowed = {"name", "nodes", "connections", "settings", "pinData", "staticData", "tags"}
    payload = {key: workflow[key] for key in allowed if key in workflow}
    updated = REST_CLIENT.request("PATCH", f"/rest/workflows/{workflow_id}", json=payload)
    return updated.get("data", updated)


def _activate_workflow_via_api(workflow_id: str) -> Optional[str]:
    if not workflow_id:
        return "ID do workflow não encontrado."
    url = _workflow_api_url(workflow_id, "activate")
    response = requests.post(
        url,
        headers=_workflow_api_headers(),
        timeout=30,
        verify=AGENT_CONFIG.verify_ssl,
    )
    if response.status_code >= 300:
        return f"{response.status_code}: {response.text}"
    return None


def _resolve_folder_id(folder_id: Optional[str], folder_name: Optional[str]) -> Optional[str]:
    if folder_id:
        return folder_id
    if not folder_name:
        return None
    normalized = folder_name.strip().lower()
    matches = [item for item in _list_folders() if (item.get("name") or "").lower() == normalized]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Pasta '{folder_name}' não encontrada.")
    if len(matches) > 1:
        raise HTTPException(status_code=400, detail=f"Pasta '{folder_name}' é ambígua; informe o ID.")
    return matches[0].get("id")


def _resolve_workflows(entries: List[WorkflowSelector]) -> List[Dict[str, str]]:
    if not entries:
        raise HTTPException(status_code=400, detail="Informe ao menos um workflow.")
    available = _list_workflows()
    resolved: List[Dict[str, str]] = []
    for entry in entries:
        candidate = None
        if entry.workflowId:
            candidate = next((wf for wf in available if wf.get("id") == entry.workflowId), None)
            if not candidate:
                raise HTTPException(status_code=404, detail=f"Workflow '{entry.workflowId}' não encontrado.")
        elif entry.workflowName:
            normalized = entry.workflowName.strip().lower()
            matches = [wf for wf in available if (wf.get("name") or "").lower() == normalized]
            if not matches:
                raise HTTPException(status_code=404, detail=f"Workflow '{entry.workflowName}' não encontrado.")
            if len(matches) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workflow '{entry.workflowName}' é ambíguo; informe o ID.",
                )
            candidate = matches[0]
        else:
            raise HTTPException(
                status_code=400,
                detail="Cada workflow precisa de 'workflowId' ou 'workflowName'.",
            )
        resolved.append({"id": candidate["id"], "name": candidate.get("name", candidate["id"])})
    return resolved


def _perform_create_folder(payload: FolderRequest) -> Dict[str, object]:
    parent_id = _resolve_folder_id(payload.parentFolderId, payload.parentFolderName)
    body: Dict[str, object] = {"name": payload.name}
    if parent_id:
        body["parentFolderId"] = parent_id
    created = REST_CLIENT.request("POST", f"/rest/projects/{PROJECT_ID}/folders", json=body)
    folder = created.get("data", created)
    return {
        "message": f"Pasta '{folder.get('name')}' criada (ID {folder.get('id')}).",
        "folderId": folder.get("id"),
        "parentFolderId": folder.get("parentFolderId"),
    }


def _perform_move_workflows(payload: MoveWorkflowRequest) -> Dict[str, object]:
    workflows = _resolve_workflows(payload.workflows)
    if payload.toRoot:
        target_folder_id = None
        target_label = "raiz"
    else:
        target_folder_id = _resolve_folder_id(payload.targetFolderId, payload.targetFolderName)
        if not target_folder_id:
            raise HTTPException(status_code=400, detail="Informe a pasta destino ou defina toRoot=true.")
        target = next((f for f in _list_folders() if f.get("id") == target_folder_id), {})
        target_label = target.get("name", target_folder_id)
    summary: List[str] = []
    for wf in workflows:
        REST_CLIENT.request(
            "PATCH",
            f"/rest/workflows/{wf['id']}",
            json={"parentFolderId": target_folder_id},
        )
        summary.append(f"{wf['name']} → {target_label}")
    return {
        "message": "Workflows movidos:\n" + "\n".join(summary),
        "movedCount": len(summary),
        "targetFolderId": target_folder_id,
    }


def _perform_archive_workflows(payload: ArchiveWorkflowRequest) -> Dict[str, object]:
    action = payload.action.lower()
    if action not in {"archive", "restore"}:
        raise HTTPException(status_code=400, detail="Ação inválida. Use 'archive' ou 'restore'.")
    workflows = _resolve_workflows(payload.workflows)
    summary: List[str] = []
    for wf in workflows:
        endpoint = "/archive" if action == "archive" else "/restore"
        try:
            REST_CLIENT.request("POST", f"/rest/workflows/{wf['id']}{endpoint}")
            verb = "arquivado" if action == "archive" else "restaurado"
            summary.append(f"{wf['name']} → {verb}.")
        except HTTPException as exc:
            summary.append(f"{wf['name']} → falha ({exc.detail}).")
    return {"message": "\n".join(summary), "action": action, "total": len(summary)}


def _execute_plan(plan: Dict[str, object], prompt: str) -> str:
    action = (plan.get("action") or "create_workflow").lower()
    args = plan.get("arguments") or {}
    try:
        if action == "create_folder":
            request = FolderRequest(**args)
            result = _perform_create_folder(request)
            return _format_tool_result(result)
        if action == "move_workflow":
            if "workflows" not in args:
                raise HTTPException(status_code=400, detail="Informe os workflows a mover.")
            request = MoveWorkflowRequest(**args)
            result = _perform_move_workflows(request)
            return _format_tool_result(result)
        if action == "archive_workflow":
            if "workflows" not in args:
                raise HTTPException(status_code=400, detail="Informe os workflows a arquivar/restaurar.")
            request = ArchiveWorkflowRequest(**args)
            result = _perform_archive_workflows(request)
            return _format_tool_result(result)
        # default to workflow generation
        return _build_workflow(prompt)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Dados inválidos: {exc}") from exc



@app.post("/internal/folders/create")
def internal_create_folder(payload: FolderRequest):
    return _perform_create_folder(payload)


@app.post("/internal/workflows/move")
def internal_move_workflows(payload: MoveWorkflowRequest):
    return _perform_move_workflows(payload)


@app.post("/internal/workflows/archive")
def internal_archive_workflows(payload: ArchiveWorkflowRequest):
    return _perform_archive_workflows(payload)





