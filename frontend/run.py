"""Launch Gradio frontend directly — simplest way."""
from app import demo, theme, css

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        theme=theme,
        css=css,
    )
