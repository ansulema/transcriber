"""Gradio frontend — ходит к backend API по HTTP."""

import os
import tempfile
import gradio as gr
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001")


def _health_check():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        data = r.json()
        return f"✅ Сервер жив | Версия: {data['version']} | ASR: {data['asr_model']} | LLM: {data['llm_model']}"
    except Exception as e:
        return f"❌ Сервер недоступен: {e}"


def _transcribe_audio(file_obj):
    if file_obj is None:
        return "⚠️ Сначала загрузи аудиофайл.", ""

    try:
        with open(file_obj.name, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/transcribe",
                files={"file": (os.path.basename(file_obj.name), f)},
                timeout=600,
            )
        if r.status_code != 200:
            return f"❌ Ошибка сервера: {r.text}", ""
        data = r.json()
        transcript = data["transcript_text"]
        summary = (
            f"📊 **Статистика:**\n"
            f"- Файл: `{data['filename']}`\n"
            f"- Длительность: {data['duration_sec']:.0f} сек ({data['duration_sec']/60:.1f} мин)\n"
            f"- Сегментов: {data['num_segments']}\n"
        )
        return summary, transcript
    except requests.exceptions.ConnectionError:
        return "❌ Не удалось подключиться к бекенду. Запущен ли backend на :8001?", ""
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", ""


def _fill_protocol_click(transcript, template_file):
    if not transcript.strip():
        return "⚠️ Сначала получи транскрипт аудио.", None

    template_name = None
    if template_file is not None:
        try:
            with open(template_file.name, "rb") as f:
                r = requests.post(
                    f"{BACKEND_URL}/upload-template",
                    files={"file": (os.path.basename(template_file.name), f)},
                    timeout=30,
                )
            if r.status_code == 200:
                template_name = r.json()["filename"]
        except Exception:
            pass

    try:
        r = requests.post(
            f"{BACKEND_URL}/protocol",
            json={"transcript": transcript, "template_filename": template_name},
            timeout=300,  # 5 минут на LLM
        )
        if r.status_code != 200:
            return f"❌ Ошибка: {r.text}", None
        data = r.json()
        return data["protocol_text"], None
    except requests.exceptions.ConnectionError:
        return "❌ Сервер недоступен.", None
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None


def _download_docx_click(transcript, template_file):
    """Скачать протокол как .docx."""
    if not transcript.strip():
        return None

    template_name = None
    if template_file is not None:
        try:
            with open(template_file.name, "rb") as f:
                r = requests.post(
                    f"{BACKEND_URL}/upload-template",
                    files={"file": (os.path.basename(template_file.name), f)},
                    timeout=30,
                )
            if r.status_code == 200:
                template_name = r.json()["filename"]
        except Exception:
            pass

    try:
        r = requests.post(
            f"{BACKEND_URL}/protocol/download",
            json={"transcript": transcript, "template_filename": template_name},
            timeout=300,
        )
        if r.status_code != 200:
            return None

        # Сохраняем .docx во временный файл
        tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except Exception:
        return None


def _chat_respond(message, history, transcript):
    """Ответить на вопрос по транскрипту через LLM."""
    if not message.strip():
        return "", history

    # Защита от None history
    if history is None:
        history = []

    if not transcript.strip():
        history.append((message, "⚠️ Сначала расшифруй аудио — чат работает по содержимому транскрипта."))
        return "", history

    try:
        # Gradio 6 history: list of (user_msg, bot_msg) tuples
        formatted_history = []
        for user_msg, bot_msg in history:
            formatted_history.append({"role": "user", "content": user_msg})
            formatted_history.append({"role": "assistant", "content": bot_msg})

        r = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "message": message,
                "transcript": transcript,
                "history": formatted_history,
            },
            timeout=120,
        )

        reply_raw = r.json()
        if isinstance(reply_raw, dict) and "reply" in reply_raw:
            reply = reply_raw["reply"]
        elif isinstance(reply_raw, dict) and "detail" in reply_raw:
            reply = f"❌ Ошибка сервера: {reply_raw['detail']}"
        else:
            reply = f"❌ Неожиданный ответ: {str(reply_raw)[:200]}"

        history.append((message, reply))
        return "", history
    except requests.exceptions.ConnectionError:
        reply = "❌ Сервер недоступен."
        history.append((message, reply))
        return "", history
    except Exception as e:
        reply = f"❌ Ошибка: {str(e)}"
        history.append((message, reply))
        return "", history


# ── Тема ───────────────────────────────────────────────────────────────
_emerald = gr.themes.utils.colors.emerald
_blue = gr.themes.colors.blue
_lime = gr.themes.colors.lime

theme = gr.themes.Default(
    primary_hue=_emerald,
    secondary_hue=_blue,
    neutral_hue=_lime,
    radius_size="lg",
).set(
    body_background_fill="linear-gradient(180deg, #b2f288, #ffffff)",
    button_primary_background_fill="#52e35e",
    button_primary_background_fill_hover="#38c944",
)


# ── Интерфейс ─────────────────────────────────────────────────────────
with gr.Blocks(title="AI Transcriber — Протоколы совещаний") as demo:
    gr.Markdown(
        """
        # 🎙️ AI Transcriber
        ### Whisper medium → pyannote 3.1 → **Qwen2.5-7B (vLLM)**
        """
    )

    with gr.Row():
        health_btn = gr.Button("🩺 Проверить сервер", size="sm", scale=1)
        health_output = gr.Markdown("_нажми для проверки_")
    health_btn.click(fn=_health_check, inputs=[], outputs=[health_output])

    # ── Состояние (текущий транскрипт для чата) ─────────────────────
    transcript_state = gr.State("")

    # ── Таб 1: Транскрибация аудио ─────────────────────────────────────────
    with gr.Tab("🎧 Транскрибация аудио"):
        gr.Markdown("### Загрузи аудиозапись совещания")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                audio_input = gr.File(
                    file_types=["audio", "video"],
                    file_count="single",
                    type="filepath",
                    label="Аудиофайл (.mp3, .wav, .flac, .webm)",
                )
                transcribe_btn = gr.Button(
                    "🎯 Расшифровать аудиозапись",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("---")
                gr.Markdown("### Шаблон протокола")
                template_input = gr.File(
                    file_types=[".docx", ".md", ".txt"],
                    file_count="single",
                    type="filepath",
                    label="Загрузи шаблон протокола (.docx/.md)",
                )
                with gr.Row():
                    protocol_btn = gr.Button(
                        "📝 Заполнить протокол (LLM)",
                        variant="primary",
                        size="lg",
                    )
                    download_btn = gr.Button(
                        "⬇️ Скачать .docx",
                        size="lg",
                    )

            with gr.Column(scale=2):
                stats_output = gr.Markdown("_Статистика появится после расшифровки_")
                transcript_output = gr.Textbox(
                    label="📄 Расшифровка",
                    lines=20,
                    max_lines=40,
                )

        with gr.Row():
            protocol_output = gr.Markdown(
                "_Здесь будет заполненный протокол (gemma3:12b)_"
            )
            docx_download = gr.File(label="Готовый протокол .docx", visible=True)

        transcribe_btn.click(
            fn=_transcribe_audio,
            inputs=[audio_input],
            outputs=[stats_output, transcript_output],
        ).then(
            fn=lambda stats, text: text,
            inputs=[stats_output, transcript_output],
            outputs=[transcript_state],
        )
        protocol_btn.click(
            fn=_fill_protocol_click,
            inputs=[transcript_output, template_input],
            outputs=[protocol_output, docx_download],
        )
        download_btn.click(
            fn=_download_docx_click,
            inputs=[transcript_output, template_input],
            outputs=[docx_download],
        )

    # ── Таб 2: Чат ───────────────────────────────────────────────────
    with gr.Tab("💬 Чат с LLM"):
        gr.Markdown(
            """
            ### 🧠 Чат-ассистент по содержимому совещания
            Задай вопрос по расшифрованному транскрипту.
            """
        )

        chatbot = gr.Chatbot(label="Диалог")
        with gr.Row():
            msg_input = gr.Textbox(
                label="Твой вопрос",
                placeholder="Спроси что-нибудь о совещании...",
                scale=4,
            )
            send_btn = gr.Button("💬 Отправить", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Очистить", scale=1)

        def _on_send(message, history, transcript):
            return _chat_respond(message, history, transcript)

        send_btn.click(
            fn=_on_send,
            inputs=[msg_input, chatbot, transcript_state],
            outputs=[msg_input, chatbot],
        )
        msg_input.submit(
            fn=_on_send,
            inputs=[msg_input, chatbot, transcript_state],
            outputs=[msg_input, chatbot],
        )
        clear_btn.click(fn=lambda: [], inputs=[], outputs=[chatbot])

    gr.Markdown(
        "---\n"
        "⚡ **AI Transcriber v2** | Whisper medium + pyannote 3.1 + **Qwen2.5-7B (vLLM)** | "
        "Backend :8001 → vLLM :11435 → Frontend :8080"
    )


if __name__ == "__main__":
    demo.launch(server_port=8080, theme=theme)
