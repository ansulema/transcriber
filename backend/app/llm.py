"""LLM модуль — ходит к vLLM серверу через OpenAI-совместимый API.

vLLM запущен как отдельный сервис на :11435.
Использует Qwen2.5-7B-Instruct.
"""

import os
import re
import json
import tiktoken
from openai import OpenAI

# ── Конфиг ─────────────────────────────────────────────────────────────
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:11435/v1")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONTEXT_WINDOW = 8000  # используем 8K из max-model-len
TOKENIZER = tiktoken.get_encoding("cl100k_base")

FILLER_WORDS = r"ну, |как бы|соответственно|совсем|вообще|так, |в общем|в принципе|просто|вот|тогда|, да, |, да|    "

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
    return _client


def post_process(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(FILLER_WORDS, "", text, flags=re.IGNORECASE)
    return text


# ── Промпты ────────────────────────────────────────────────────────────

SYSTEM_MAIN = """Ты — ассистент, который заполняет протоколы совещаний на основе расшифровки.
Твоя задача — прочитать транскрипт, проанализировать его и заполнить пустой шаблон протокола.
Отвечай только на русском языке. Не выдумывай информацию, которой нет в транскрипте."""


def _call_llm(messages: list[dict], max_tokens: int = 4096) -> str:
    """Отправить запрос к vLLM и получить ответ."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
    )
    return response.choices[0].message.content or ""


# ── Заполнение протокола ───────────────────────────────────────────────


def fill_protocol(transcript: str, template_content: str | None = None) -> str:
    """Заполнить протокол через LLM (vLLM).

    Args:
        transcript: текст транскрипта в формате Егора
        template_content: шаблон протокола в markdown (или None)

    Returns:
        заполненный протокол в markdown
    """
    cleaned = post_process(transcript)

    if not template_content or not template_content.strip():
        # Без шаблона — LLM сама формирует отчёт
        template_content = (
            "# Отчёт по совещанию\n\n"
            "**Тема совещания:** \n"
            "**Дата проведения:** \n"
            "**Участники:** \n\n"
            "---\n\n"
            "### Обсуждённые вопросы\n"
            "1. \n"
            "2. \n"
            "3. \n\n"
            "### Принятые решения\n"
            "- \n"
            "- \n\n"
            "### Задачи\n\n"
            "| № | Задача | Ответственный | Срок | Статус |\n"
            "|---|--------|---------------|------|--------|\n"
            "| 1 | | | | |\n"
            "| 2 | | | | |\n"
            "| 3 | | | | |\n\n"
            "---\n"
        )

    # Проверяем длину — если большой транскрипт, режем первую половину
    # (vLLM с max-model-len=8192 справится с большинством транскриптов)
    n_tokens = len(TOKENIZER.encode(cleaned))
    if n_tokens > CONTEXT_WINDOW - 1500:
        # Обрезаем до ~6000 токенов (оставляем место для шаблона и ответа)
        tokens = TOKENIZER.encode(cleaned)
        tokens = tokens[:6000]
        cleaned = TOKENIZER.decode(tokens) + "\n\n... [транскрипт обрезан]"

    # Один запрос — модель читает транскрипт и сразу заполняет протокол
    messages = [
        {"role": "system", "content": SYSTEM_MAIN},
        {
            "role": "user",
            "content": (
                "Прочитай транскрипт совещания и заполни пустой протокол. "
                "Не выдумывай информацию. Если данных нет — напиши 'не указано'.\n\n"
                f"=== ТРАНСКРИПТ ===\n{cleaned}\n\n"
                f"=== ПУСТОЙ ПРОТОКОЛ ===\n{template_content}\n\n"
                "Верни ТОЛЬКО заполненный протокол, без лишних комментариев."
            ),
        },
    ]

    print(f"[LLM] Sending request ({n_tokens} tokens)...")
    result = _call_llm(messages)
    print(f"[LLM] Response: {len(result)} chars")
    return result


# ── Чат по транскрипту ────────────────────────────────────────────────

SYSTEM_CHAT = """Ты — ассистент, который отвечает на вопросы по содержанию совещания.
Используй только информацию из транскрипта. Если ответа нет в транскрипте — скажи об этом."""


def chat_about_transcript(message: str, transcript: str, history: list[dict] | None = None) -> str:
    """Ответить на вопрос пользователя о совещании.

    Args:
        message: вопрос пользователя
        transcript: текст транскрипта
        history: история чата

    Returns:
        ответ LLM
    """
    if history is None:
        history = []

    # Ограничиваем транскрипт если слишком длинный
    cleaned = post_process(transcript)
    n_tokens = len(TOKENIZER.encode(cleaned))
    if n_tokens > 4000:
        tokens = TOKENIZER.encode(cleaned)
        tokens = tokens[:4000]
        cleaned = TOKENIZER.decode(tokens)

    messages = [
        {"role": "system", "content": SYSTEM_CHAT},
        {"role": "user", "content": f"Вот транскрипт совещания:\n\n{cleaned}"},
    ]

    for h in history[-6:]:  # последние 6 сообщений для контекста
        messages.append(h)

    messages.append({"role": "user", "content": message})

    result = _call_llm(messages, max_tokens=2048)
    return result
