# AI Transcriber v2

Три сервиса: vLLM (LLM), Backend (ASR + диаризация + API), Frontend (Gradio).

```
Backend :8001 ──→ vLLM :11435 ──→ Frontend :8080
  (GPU)            (GPU)              (CPU)
```

## Окружения

**Система:** Ubuntu 24.04, CUDA 13.2

Два conda-окружения:

| Окружение | Назначение | requirements | Зависимости системы |
|-----------|-----------|-------------|-------------------|
| `transcriber` | Backend + Frontend | `requirements.txt` | `pandoc ffmpeg` |
| `vllm` | LLM-сервер | `requirements-vllm.txt` | CUDA 12.x/13.x |

## Установка

```bash
# Системные пакеты (Ubuntu 24.04 / 22.04)
sudo apt-get install -y pandoc ffmpeg

# 1. Backend + Frontend
conda create -n transcriber python=3.12 -y
conda activate transcriber
pip install -r requirements.txt

# 2. vLLM
conda create -n vllm python=3.12 -y
conda activate vllm
pip install -r requirements-vllm.txt

# 3. .env с HF_TOKEN
echo 'HF_TOKEN=hf_...' > backend/.env
```

## Запуск (три терминала)

### 1. vLLM
```bash
conda activate vllm
source backend/.env
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 11435 --host 0.0.0.0 \
  --max-model-len 8192 --quantization fp8 \
  --gpu-memory-utilization 0.55 --trust-remote-code
```

### 2. Backend
```bash
conda activate transcriber
cd backend && source .env
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 3. Frontend
```bash
conda activate transcriber
cd frontend
BACKEND_URL=http://127.0.0.1:8001 python run.py
```

Открыть: **http://localhost:8080**

## API

| Путь | Метод | Описание |
|------|-------|----------|
| `/health` | GET | Статус |
| `/transcribe` | POST | Аудио → транскрипт |
| `/protocol/download` | POST | Транскрипт (+ шаблон) → .docx |
| `/chat` | POST | Вопрос → ответ LLM |

## Возможности

- Загрузка аудио → Whisper large-v3 + PyAnnote → транскрипт с диаризацией
- LLM-протокол: с шаблоном (.docx) или без — отчёт с задачами, ответственными, сроками
- Скачивание готового протокола в .docx
- Оконная суммаризация длинных транскриптов для 8K контекста
