"""
Frontend: Gradio UI for AI Transcriber.
"""

from __future__ import annotations

import os
import tempfile
import requests

import gradio as gr

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001")


# ── API helpers ──────────────────────────────────────────────────────────

def api_transcribe(file_path: str) -> str:
    if not file_path:
        return "Файл не выбран."
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/transcribe",
            files={"file": f},
            timeout=600,
        )
    if resp.status_code != 200:
        return f"Ошибка: {resp.text}"
    return resp.json()["transcript_text"]


def api_download_docx(transcript: str, template_path: str | None) -> str | None:
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
        json={"transcript": transcript, "template_text": template_text},
        timeout=300,
    )
    if resp.status_code != 200:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


# ── Theme ────────────────────────────────────────────────────────────────

theme = gr.themes.Default(
    primary_hue=gr.themes.utils.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.cyan,
    radius_size="lg",
).set(
    button_primary_background_fill="#4ad9d9",
    button_primary_background_fill_hover="#32a1a1",
    body_background_fill="#f0fdfa",
)


# ── CSS ──────────────────────────────────────────────────────────────────

css = """
.gradio-container { max-width: 100% !important; padding: 0.5rem 1rem !important; }
footer { display: none !important; }
"""

# ── UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Transcriber") as demo:
    gr.Markdown("## AI Transcriber")

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
        outputs=[audio_answer],
    )
    protocol_button.click(
        fn=api_download_docx,
        inputs=[audio_answer, template_input],
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
