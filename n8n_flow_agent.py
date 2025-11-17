#!/usr/bin/env python3
"""
Ferramenta para gerar fluxos n8n a partir de texto livre usando um LLM
e publicar via API do n8n.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


ENV_HINT_PATH = Path(".env.n8n")


def _load_env_hint(path: Path = ENV_HINT_PATH) -> None:
    """Populate os.environ with key/value pairs from a simple .env file if present."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class ConfigError(RuntimeError):
    """Represents missing or invalid configuration."""


@dataclass
class AgentConfig:
    n8n_base_url: str
    n8n_api_key: str
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    verify_ssl: bool = True
    workflows_endpoint: str = "/api/v1/workflows"

    @classmethod
    def from_env(
        cls,
        overrides: Optional[Dict[str, Optional[str]]] = None,
    ) -> "AgentConfig":
        overrides = overrides or {}
        _load_env_hint()

        def pick(key: str) -> Optional[str]:
            if overrides.get(key):
                return overrides[key]
            return os.environ.get(key)

        missing = [
            key
            for key, value in [
                ("N8N_BASE_URL", pick("N8N_BASE_URL")),
                ("N8N_API_KEY", pick("N8N_API_KEY")),
                ("OPENAI_API_KEY", pick("OPENAI_API_KEY")),
            ]
            if not value
        ]
        if missing:
            raise ConfigError(
                f"As variaveis obrigatorias faltantes: {', '.join(missing)}. "
                "Defina-as ou crie um arquivo .env.n8n."
            )

        verify_ssl = os.environ.get("N8N_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

        workflows_endpoint = pick("N8N_WORKFLOWS_ENDPOINT") or "/api/v1/workflows"
        if not workflows_endpoint.startswith("/"):
            workflows_endpoint = f"/{workflows_endpoint}"

        return cls(
            n8n_base_url=pick("N8N_BASE_URL").rstrip("/"),
            n8n_api_key=pick("N8N_API_KEY"),
            openai_api_key=pick("OPENAI_API_KEY"),
            openai_model=pick("OPENAI_MODEL") or "gpt-4o-mini",
            verify_ssl=verify_ssl,
            workflows_endpoint=workflows_endpoint,
        )


def read_prompt(prompt_arg: Optional[str], prompt_file: Optional[str]) -> str:
    """Resolve prompt precedence: --prompt-file > arg > stdin."""
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    if prompt_arg and prompt_arg != "-":
        return prompt_arg.strip()
    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("Nenhum prompt informado.")
    return data


def strip_code_fence(payload: str) -> str:
    """Remove marcacoes ```json ...``` do retorno do modelo."""
    fenced = payload.strip()
    if fenced.startswith("```"):
        fenced = re.sub(r"^```(?:json)?", "", fenced, flags=re.IGNORECASE).strip()
        if fenced.endswith("```"):
            fenced = fenced[: -3].strip()
    return fenced


def ensure_uuid(value: Optional[str] = None) -> str:
    return value or str(uuid.uuid4())


def ensure_workflow_defaults(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Garante campos minimos requeridos pelo n8n."""
    workflow.setdefault("settings", {})
    workflow.setdefault("pinData", {})
    workflow.setdefault("staticData", None)
    workflow.setdefault("tags", [])
    workflow.setdefault("connections", {})
    workflow["name"] = workflow.get("name") or f"Fluxo Gerado {uuid.uuid4().hex[:8]}"
    workflow["versionId"] = ensure_uuid(workflow.get("versionId"))

    nodes: List[Dict[str, Any]] = workflow.get("nodes") or []
    if not nodes:
        raise ValueError("O modelo nao retornou nenhum node.")

    for index, node in enumerate(nodes):
        node["id"] = ensure_uuid(node.get("id"))
        node["name"] = node.get("name") or f"Node {index + 1}"
        node["typeVersion"] = node.get("typeVersion") or 1
        node.setdefault("parameters", {})
        if "position" not in node:
            node["position"] = [index * 200, 200]

    return workflow


def call_llm(prompt: str, model: str, api_key: str, temperature: float = 0.2) -> Dict[str, Any]:
    """Gera um workflow n8n usando o modelo especificado."""
    client = OpenAI(api_key=api_key)
    system_prompt = textwrap.dedent(
        """
        Voce e um arquiteto de automacoes n8n. Converta o pedido do usuario em um workflow valido
        retornando APENAS um objeto JSON. Siga as regras:
        - Inclua pelo menos um node de gatilho (ex: Webhook ou Schedule).
        - Todos os nodes precisam de: id UUID, name, type, typeVersion, parameters, position [x, y].
        - Configure conexoes usando o formato padrao do n8n (objeto connections -> <node> -> main -> [[{node, type, index}]])
        - Use somente credenciais ja existentes referenciando pelo nome (campo parameters.credentials).
        - Evite textos explicativos; somente JSON.
        - Preferir nodes oficiais (n8n-nodes-base.*) a nao ser que o pedido exija outro pacote.
        - Se precisar de codigo, utilize nodes \"code\" ou \"function\" e descreva o JS em jsCode.
        - Utilize expressoes n8n quando necessario (={{ ... }}).
        """
    ).strip()

    user_prompt = (
        "Leia a solicitacao abaixo e produza um workflow n8n completo, pronto para ser criado via API.\n"
        "Requisitos:\n"
        "- Ajuste o campo \"name\" para descrever o fluxo.\n"
        "- Limite-se aos nodes necessarios.\n"
        "- Caso seja util, inclua tags (array de strings).\n"
        "- Retorne SOMENTE JSON valido.\n\n"
        f"Solicitacao:\n{prompt.strip()}"
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("O modelo nao retornou conteudo.")
    parsed = json.loads(strip_code_fence(content))
    return ensure_workflow_defaults(parsed)


def deploy_workflow(
    config: AgentConfig,
    workflow: Dict[str, Any],
    activate: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Publica o workflow via API REST do n8n."""
    allowed_fields = {"name", "nodes", "connections", "settings"}
    payload = {k: workflow[k] for k in allowed_fields if k in workflow}

    url = f"{config.n8n_base_url}{config.workflows_endpoint}"
    headers = {
        "X-N8N-API-KEY": config.n8n_api_key,
        "Content-Type": "application/json",
    }
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
        verify=config.verify_ssl,
    )
    if response.status_code >= 300:
        raise RuntimeError(
            f"Falha ao criar workflow ({response.status_code}): {response.text}"
        )
    created = response.json()

    if activate:
        workflow_id = created.get("id")
        if not workflow_id:
            raise RuntimeError("Workflow criado mas sem ID retornado; não é possível ativar.")
        activate_url = f"{config.n8n_base_url}{config.workflows_endpoint}/{workflow_id}/activate"
        activate_response = requests.post(
            activate_url,
            headers=headers,
            timeout=timeout,
            verify=config.verify_ssl,
        )
        if activate_response.status_code >= 300:
            raise RuntimeError(
                f"Workflow criado ({workflow_id}) mas falhou ao ativar ({activate_response.status_code}): {activate_response.text}"
            )
        created = activate_response.json()

    return created


def save_output(path: str, data: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera e publica workflows n8n a partir de texto livre."
    )
    parser.add_argument("prompt", nargs="?", help="Descricao em texto do fluxo.")
    parser.add_argument(
        "--prompt-file",
        help="Arquivo contendo a descricao completa (substitui o argumento).",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Ativa o workflow apos criar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nao envia para o n8n; apenas imprime o JSON.",
    )
    parser.add_argument(
        "--output-json",
        help="Salva o JSON gerado no caminho especificado.",
    )
    parser.add_argument(
        "--model",
        help="Modelo OpenAI a utilizar (padrao: gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura do modelo (default 0.2).",
    )
    parser.add_argument(
        "--base-url",
        dest="N8N_BASE_URL",
        help="Override para N8N_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        dest="N8N_API_KEY",
        help="Override para N8N_API_KEY.",
    )
    parser.add_argument(
        "--openai-key",
        dest="OPENAI_API_KEY",
        help="Override para OPENAI_API_KEY.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        prompt = read_prompt(args.prompt, args.prompt_file)
    except Exception as exc:
        print(f"Erro ao ler prompt: {exc}", file=sys.stderr)
        return 1

    try:
        config = AgentConfig.from_env(
            {
                "N8N_BASE_URL": args.N8N_BASE_URL,
                "N8N_API_KEY": args.N8N_API_KEY,
                "OPENAI_API_KEY": args.OPENAI_API_KEY,
            }
        )
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    model = args.model or config.openai_model

    try:
        workflow = call_llm(
            prompt=prompt,
            model=model,
            api_key=config.openai_api_key,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"Erro ao gerar workflow via LLM: {exc}", file=sys.stderr)
        return 3

    if args.output_json:
        save_output(args.output_json, workflow)

    if args.dry_run:
        print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return 0

    try:
        created = deploy_workflow(config, workflow, activate=args.activate)
    except Exception as exc:
        print(f"Erro ao publicar workflow no n8n: {exc}", file=sys.stderr)
        return 4

    workflow_id = created.get("id")
    name = created.get("name") or workflow.get("name")
    status = "ativado" if created.get("active") else "criado"
    print(f"Workflow '{name}' ({workflow_id}) {status} com sucesso.")
    if created.get("versionId"):
        print(f"VersionId: {created['versionId']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
