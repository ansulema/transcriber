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
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
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


def strip_timestamps(text: str) -> str:
    """Remove timestamps from format: 'speaker | MM:SS-MM:SS | text' -> 'speaker: text'."""
    lines = text.split("\n")
    result = []
    for line in lines:
        # Match: "N | MM:SS-MM:SS | text" -> "Speaker N: text"
        m = re.match(r"^(\d+)\s*\|\s*\d{2,}:\d{2}-\d{2,}:\d{2}\s*\|\s*(.+)$", line.strip())
        if m:
            result.append(f"Speaker {m.group(1)}: {m.group(2)}")
        else:
            result.append(line)
    return "\n".join(result)


# ── Промпты ────────────────────────────────────────────────────────────

SYSTEM_MAIN = """Ты — ассистент, который заполняет протоколы совещаний на основе расшифровки.
Твоя задача — взять ПУСТОЙ ШАБЛОН протокола и заполнить его данными из транскрипта.
Сохраняй структуру шаблона полностью — заголовки, таблицы, разделы.
Заменяй пустые места и маркеры на конкретные данные из транскрипта.
Если каких-то данных в транскрипте нет — пиши 'не указано'.
Не добавляй ничего от себя. Не меняй формат шаблона.
Отвечай только на русском языке."""


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
    cleaned = post_process(strip_timestamps(transcript))

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
                "Возьми ПУСТОЙ ШАБЛОН протокола ниже и заполни его на основе ТРАНСКРИПТА. "
                "Важно: сохрани СТРУКТУРУ шаблона — все заголовки, таблицы, колонки. "
                "Только замени пустые поля на данные из транскрипта.\n\n"
                f"=== ТРАНСКРИПТ ===\n{cleaned}\n\n"
                f"=== ПУСТОЙ ШАБЛОН ===\n{template_content}\n\n"
                "Верни ТОЛЬКО заполненный шаблон, без комментариев."
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
    cleaned = post_process(strip_timestamps(transcript))
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
