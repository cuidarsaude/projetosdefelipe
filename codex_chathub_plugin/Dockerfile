FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY codex_plugin_service.py .
COPY ../../n8n_flow_agent.py ./n8n_flow_agent.py

EXPOSE 4010

CMD ["uvicorn", "codex_plugin_service:app", "--host", "0.0.0.0", "--port", "4010"]
