"""Whisper medium — ASR через faster-whisper. Lazy singleton в GPU."""

import os
import numpy as np
import av
from faster_whisper import WhisperModel

WHISPER_MODEL = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
DOWNLOAD_ROOT = os.path.expanduser("~/.cache/whisper")

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"[ASR] Loading faster-whisper {WHISPER_MODEL} on {DEVICE}...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=DOWNLOAD_ROOT,
        )
        print("[ASR] Model loaded.")
    return _model


def load_audio_mono_16k(path: str) -> np.ndarray:
    container = av.open(path)
    stream = container.streams.audio[0]
    orig_sr = stream.sample_rate

    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        frames.append(arr)
    container.close()

    audio = np.concatenate(frames, axis=1)
    if audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)
    else:
        audio = audio[0]

    if orig_sr != 16000:
        ratio = 16000 / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    return audio


def transcribe(path: str, language: str = "ru") -> tuple[list[dict], float]:
    audio = load_audio_mono_16k(path)
    duration = len(audio) / 16000

    model = get_model()
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            threshold=0.5,
            speech_pad_ms=400,
        ),
    )

    result = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        result.append({"start": seg.start, "end": seg.end, "text": text})

    print(f"[ASR] {len(result)} segments, lang={info.language}")
    return result, duration
