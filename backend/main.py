"""FastAPI backend — точки входа для транскрибации и LLM."""

import os
import uuid
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.models import (
    UploadResponse,
    TranscribeResponse,
    ProtocolRequest,
    ProtocolResponse,
    ChatRequest,
    ChatResponse,
)
from app.pipeline import run_pipeline
from app.llm import fill_protocol
from app.llm_stub import chat_response

# ── Пути ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_DIR = BASE_DIR / "templates"
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Transcriber — Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "asr_model": "medium", "llm_model": "Qwen2.5-7B-Instruct (vLLM)"}


# ── Загрузка аудио ─────────────────────────────────────────────────────
@app.post("/upload-audio", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix if file.filename else ".wav"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        import av
        container = av.open(str(save_path))
        duration = float(
            container.streams.audio[0].duration * container.streams.audio[0].time_base
        )
        container.close()
    except Exception:
        duration = 0.0

    return UploadResponse(
        filename=unique_name,
        duration_sec=round(duration, 1),
        status="uploaded",
    )


# ── Транскрибация ──────────────────────────────────────────────────────
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix if file.filename else ".wav"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        result = run_pipeline(str(save_path))
        os.remove(save_path)
        return TranscribeResponse(**result)
    except Exception as e:
        if save_path.exists():
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/{filename}", response_model=TranscribeResponse)
async def transcribe_uploaded(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    try:
        result = run_pipeline(str(file_path))
        return TranscribeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── LLM: протокол ──────────────────────────────────────────────────────
import asyncio


@app.post("/protocol", response_model=ProtocolResponse)
async def fill_protocol_endpoint(req: ProtocolRequest):
    """Заполнить протокол через gemma3:12b (Ollama)."""
    template_content = None
    if req.template_filename:
        tmpl_path = TEMPLATE_DIR / req.template_filename
        if tmpl_path.exists():
            try:
                from markitdown import MarkItDown

                md = MarkItDown(enable_plugins=False)
                template_content = md.convert(str(tmpl_path)).text_content
            except Exception as e:
                print(f"[TEMPLATE] Error reading {req.template_filename}: {e}")

    try:
        # Запускаем блокирующий LLM в отдельном потоке, не блокируя event loop
        protocol = await asyncio.to_thread(fill_protocol, req.transcript, template_content)
        return ProtocolResponse(protocol_text=protocol, status="ok")
    except Exception as e:
        return ProtocolResponse(
            protocol_text=f"⚠️ Ошибка LLM: {e}\n\n---\n\n*Не удалось получить ответ от gemma3:12b*",
            status=f"error: {e}",
        )


@app.post("/protocol/download")
async def download_protocol(req: ProtocolRequest):
    """Заполнить протокол и вернуть .docx файл."""
    template_content = req.template_text  # prefer inline text
    if not template_content and req.template_filename:
        tmpl_path = TEMPLATE_DIR / req.template_filename
        if tmpl_path.exists():
            try:
                from markitdown import MarkItDown
                md = MarkItDown(enable_plugins=False)
                template_content = md.convert(str(tmpl_path)).text_content
            except Exception:
                pass

    try:
        protocol_md = await asyncio.to_thread(fill_protocol, req.transcript, template_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Конвертируем markdown → .docx через pypandoc
    try:
        import pypandoc

        tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        tmp_path = tmp.name
        tmp.close()

        pypandoc.convert_text(
            protocol_md,
            format="md",
            to="docx",
            outputfile=tmp_path,
        )

        return FileResponse(
            tmp_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="protocol.docx",
            headers={"Content-Disposition": "attachment; filename=protocol.docx"},
        )
    except ImportError:
        # Если pypandoc не установлен — возвращаем текст
        return ProtocolResponse(protocol_text=protocol_md, status="ok (pypandoc not installed)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX error: {e}")


# ── LLM: чат ───────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Ответить на вопрос по транскрипту через Qwen2.5 (vLLM)."""
    try:
        from app.llm import chat_about_transcript
        reply = await asyncio.to_thread(
            chat_about_transcript, req.message, req.transcript, req.history
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        return ChatResponse(reply=f"⚠️ Ошибка LLM: {e}")


# ── Загрузка шаблона ───────────────────────────────────────────────────
@app.post("/upload-template")
async def upload_template(file: UploadFile = File(...)):
    save_path = TEMPLATE_DIR / (file.filename or "template.docx")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    return {"filename": save_path.name, "status": "uploaded"}
