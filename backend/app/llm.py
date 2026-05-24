"""LLM модуль — ходит к vLLM серверу через OpenAI-совместимый API.

vLLM запущен как отдельный сервис на :11435.
Использует Qwen3-4B-Instruct-2507.
"""

import os
import re
import tiktoken
from openai import OpenAI

# ── Конфиг ─────────────────────────────────────────────────────────────
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:11435/v1")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CONTEXT_WINDOW = 8000  # max-model-len
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ~5000 токенов на окно (оставляем ~3000 на промпт + шаблон + ответ)
CHUNK_SIZE = 5000
# Запас на промпт, шаблон и ответ в финальном запросе
FINAL_OVERHEAD = 3000

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
        # Match: "Спикер N | MM:SS-MM:SS | text" -> "Спикер N: text"
        m = re.match(r"^Спикер\s+(\d+)\s*\|\s*\d{2,}:\d{2}-\d{2,}:\d{2}\s*\|\s*(.+)$", line.strip())
        if m:
            result.append(f"Спикер {m.group(1)}: {m.group(2)}")
        else:
            result.append(line)
    return "\n".join(result)


def _count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


# ── Промпты ────────────────────────────────────────────────────────────

SYSTEM_MAIN = """Ты — ассистент, который заполняет протоколы совещаний на основе расшифровки.
Твоя задача — взять ПУСТОЙ ШАБЛОН протокола и заполнить его данными из транскрипта.
Сохраняй структуру шаблона полностью — заголовки, таблицы, разделы.
Заменяй пустые места и маркеры на конкретные данные из транскрипта.
Если каких-то данных в транскрипте нет — пиши 'не указано'.
Не добавляй ничего от себя. Не меняй формат шаблона.
Если шаблон требует список участников (ФИО) — перечисли всех, кто говорит в транскрипте.
Отвечай только на русском языке."""

SYSTEM_SUMMARIZE = """Ты — ассистент, который делает краткую выжимку из фрагмента расшифровки совещания.
Выдели только ключевые факты: темы, решения, задачи, ответственных, сроки.
Пиши кратко, телеграфным стилем, только по-русски.
Не добавляй ничего от себя."""


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


# ── Оконная суммаризация ───────────────────────────────────────────────

def _summarize_chunk(chunk: str, chunk_num: int, total: int) -> str:
    """Суммаризировать один фрагмент транскрипта."""
    messages = [
        {"role": "system", "content": SYSTEM_SUMMARIZE},
        {"role": "user", "content": (
            f"Фрагмент {chunk_num}/{total} расшифровки совещания. "
            f"Сделай краткую выжимку: темы, решения, задачи, ответственные, сроки.\n\n"
            f"{chunk}"
        )},
    ]
    print(f"[LLM] Summarizing chunk {chunk_num}/{total} ({_count_tokens(chunk)} tokens)...")
    result = _call_llm(messages, max_tokens=1024)
    print(f"[LLM] Chunk {chunk_num} summary: {len(result)} chars")
    return result


def _sliding_window_summarize(text: str) -> str:
    """Разбить текст на окна, суммаризировать каждое, склеить."""
    tokens = TOKENIZER.encode(text)
    n_total = len(tokens)

    if n_total <= CHUNK_SIZE:
        return text

    # Разбиваем на окна с перекрытием 20%
    overlap = int(CHUNK_SIZE * 0.2)
    chunks = []
    start = 0
    while start < n_total:
        end = min(start + CHUNK_SIZE, n_total)
        chunk_tokens = tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += CHUNK_SIZE - overlap

    print(f"[LLM] Transcript: {n_total} tokens -> {len(chunks)} windows")

    # Суммаризируем каждое окно
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = _summarize_chunk(chunk, i + 1, len(chunks))
        summaries.append(f"--- Фрагмент {i+1} ---\n{summary}")

    combined = "\n\n".join(summaries)
    print(f"[LLM] Combined summary: {_count_tokens(combined)} tokens")
    return combined


# ── Заполнение протокола ───────────────────────────────────────────────

def fill_protocol(transcript: str, template_content: str | None = None) -> str:
    """Заполнить протокол через LLM (vLLM) с оконной суммаризацией.

    Args:
        transcript: текст транскрипта в формате спикер | время | текст
        template_content: шаблон протокола в markdown (или None)

    Returns:
        заполненный протокол в markdown
    """
    cleaned = post_process(strip_timestamps(transcript))

    if not template_content or not template_content.strip():
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

    # Если влазит в одно окно — отправляем как есть
    n_transcript = _count_tokens(cleaned)
    n_template = _count_tokens(template_content)

    if n_transcript + n_template <= CONTEXT_WINDOW - FINAL_OVERHEAD:
        final_text = cleaned
        print(f"[LLM] Fits in one window: {n_transcript} tokens")
    else:
        # Оконная суммаризация
        final_text = _sliding_window_summarize(cleaned)

    messages = [
        {"role": "system", "content": SYSTEM_MAIN},
        {
            "role": "user",
            "content": (
                "Возьми ПУСТОЙ ШАБЛОН протокола ниже и заполни его на основе ТРАНСКРИПТА. "
                "Важно: сохрани СТРУКТУРУ шаблона — все заголовки, таблицы, колонки. "
                "Только замени пустые поля на данные из транскрипта.\n\n"
                f"=== ТРАНСКРИПТ ===\n{final_text}\n\n"
                f"=== ПУСТОЙ ШАБЛОН ===\n{template_content}\n\n"
                "Верни ТОЛЬКО заполненный шаблон, без комментариев."
            ),
        },
    ]

    print(f"[LLM] Final request: transcript={_count_tokens(final_text)} + template={n_template} tokens")
    result = _call_llm(messages)
    print(f"[LLM] Response: {len(result)} chars")
    return result


# ── Чат по транскрипту ────────────────────────────────────────────────

SYSTEM_CHAT = """Ты — ассистент, который отвечает на вопросы по содержанию совещания.
Используй только информацию из транскрипта. Если ответа нет в транскрипте — скажи об этом."""


def chat_about_transcript(message: str, transcript: str, history: list[dict] | None = None) -> str:
    """Ответить на вопрос пользователя о совещании."""
    if history is None:
        history = []

    cleaned = post_process(strip_timestamps(transcript))
    n_tokens = _count_tokens(cleaned)
    if n_tokens > 4000:
        tokens = TOKENIZER.encode(cleaned)
        tokens = tokens[:4000]
        cleaned = TOKENIZER.decode(tokens)

    messages = [
        {"role": "system", "content": SYSTEM_CHAT},
        {"role": "user", "content": f"Вот транскрипт совещания:\n\n{cleaned}"},
    ]

    for h in history[-6:]:
        messages.append(h)

    messages.append({"role": "user", "content": message})

    result = _call_llm(messages, max_tokens=2048)
    return result
