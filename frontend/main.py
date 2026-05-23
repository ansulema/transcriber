"""Точка входа для фронтенда — монтирует Gradio на порт 8080.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8080
"""

import gradio as gr
from gradio.routes import mount_gradio_app
from fastapi import FastAPI
from frontend import demo, theme

app = FastAPI()
app = mount_gradio_app(app, demo, path="/", theme=theme)
