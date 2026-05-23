# AI Transcriber v2 — три сервиса

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Backend     │     │  vLLM        │     │  Frontend    │
│  :8001       │────>│  :11435      │────>│  :8080       │
│  FastAPI     │     │  Qwen2.5-7B  │     │  Gradio      │
│  ASR+диариз. │     │  (GPU)       │     │  (без GPU)   │
└──────────────┘     └──────────────┘     └──────────────┘
     GPU                    GPU                CPU
```

## Запуск (три терминала)

### 1. vLLM (LLM сервер)
```bash
conda activate test-env
cd /home/user1/transcriber-v2/backend
source .env
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 11435 --host 0.0.0.0 \
  --max-model-len 8192 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 --trust-remote-code
```

### 2. Backend (FastAPI + ASR + диаризация)
```bash
conda activate test-env
cd /home/user1/transcriber-v2/backend
source .env
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 3. Frontend (Gradio)
```bash
conda activate test-env
cd /home/user1/transcriber-v2/frontend
BACKEND_URL=http://127.0.0.1:8001 uvicorn main:app --host 0.0.0.0 --port 8080
```

Открыть: **http://localhost:8080**

## Остановка
```bash
fuser -k 11435/tcp  # vLLM
fuser -k 8001/tcp   # backend
fuser -k 8080/tcp   # frontend
```

## Возможности

| Функция | Что делает |
|---------|-----------|
| **🎧 Расшифровка** | Загрузить аудио → Whisper medium + pyannote → транскрипт (формат Егора) |
| **📝 Протокол (LLM)** | Без шаблона: отчёт с задачами, ответственными, дедлайнами. С шаблоном: заполняет шаблон |
| **⬇️ Скачать .docx** | Готовый протокол в формате Word |
| **💬 Чат** | Задать вопрос по содержимому совещания. Работает только после расшифровки |

## API

| Путь | Метод | Описание |
|------|-------|----------|
| `/health` | GET | Статус |
| `/transcribe` | POST | Аудио → транскрипт |
| `/protocol` | POST | Транскрипт → протокол (LLM) |
| `/protocol/download` | POST | Транскрипт → .docx |
| `/chat` | POST | Вопрос по транскрипту → ответ (LLM) |
