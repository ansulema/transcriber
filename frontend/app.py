"""
Frontend: Gradio UI for AI Transcriber.
"""

from __future__ import annotations

import os
import requests

import gradio as gr

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001")


# ── API helpers ──────────────────────────────────────────────────────────

def api_transcribe(file_path: str) -> tuple[str, str]:
    """Transcribe audio -> (transcript_text, filename)."""
    if not file_path:
        return "Файл не выбран.", ""
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/transcribe",
            files={"file": f},
            timeout=600,
        )
    if resp.status_code != 200:
        return f"Ошибка: {resp.text}", ""
    data = resp.json()
    return data["transcript_text"], data.get("original_filename", data["filename"])


def api_download_docx(transcript: str, template_path: str | None, audio_filename: str) -> str | None:
    if not transcript.strip():
        return None

    template_text = None
    if template_path:
        try:
            from markitdown import MarkItDown
            md = MarkItDown(enable_plugins=False)
            template_text = md.convert(template_path).text_content
        except Exception:
            pass

    resp = requests.post(
        f"{BACKEND_URL}/protocol/download",
        json={
            "transcript": transcript,
            "template_text": template_text,
            "original_filename": audio_filename,
        },
        timeout=300,
    )
    print(f"[FRONTEND] download: status={resp.status_code} len={len(resp.content)}")
    if resp.status_code != 200:
        print(f"[FRONTEND] download error: {resp.text[:200]}")
        return None

    tmp_path = "/tmp/report.docx"
    with open(tmp_path, "wb") as f:
        f.write(resp.content)
    print(f"[FRONTEND] saved to: {tmp_path}")
    return tmp_path


# ── Theme ────────────────────────────────────────────────────────────────

theme = gr.Theme.from_hub("harsh8001/skymist")


# ── CSS ──────────────────────────────────────────────────────────────────

css = """
.gradio-container { max-width: 100% !important; padding: 0.5rem 1rem !important; }
footer { display: none !important; }
"""

# ── UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Transcriber") as demo:
    gr.Markdown("## AI Transcriber")

    filename_state = gr.State("")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=250):
            audio_input = gr.File(
                file_types=["audio", "video"],
                file_count="single",
                type="filepath",
                label="Аудио",
                height=120,
            )
            audio_button = gr.Button(
                "Расшифровать", variant="primary"
            )
            gr.Markdown("---")
            template_input = gr.File(
                file_types=[".docx", ".md", ".txt"],
                file_count="single",
                type="filepath",
                label="Шаблон протокола",
                height=120,
            )
            protocol_button = gr.Button(
                "Заполнить и скачать протокол (.docx)", variant="primary"
            )

        with gr.Column(scale=2, min_width=400):
            audio_answer = gr.Textbox(
                label="Расшифровка",
                lines=22,
            )
            docx_output = gr.File(label="Готовый протокол", visible=True, height=80)

    audio_button.click(
        fn=api_transcribe,
        inputs=[audio_input],
        outputs=[audio_answer, filename_state],
    )
    protocol_button.click(
        fn=api_download_docx,
        inputs=[audio_answer, template_input, filename_state],
        outputs=[docx_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        theme=theme,
        css=css,
    )
