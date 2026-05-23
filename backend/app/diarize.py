"""Pyannote speaker diarization 3.1 — lazy singleton."""

import os
import torch
import numpy as np
from pathlib import Path
from pyannote.audio import Pipeline

DEVICE = "cuda"
_LOOKUP_PATHS = [
    Path(__file__).parent.parent / ".env",           # backend/.env
    Path(__file__).parent.parent.parent / ".env",    # корень transcriber-v2/.env
]

_pipeline: Pipeline | None = None


def _load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        for p in _LOOKUP_PATHS:
            if p.exists():
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "HF_TOKEN":
                                token = v.strip()
                                break
                if token:
                    break
    return token


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        hf_token = _load_hf_token()
        if not hf_token:
            raise RuntimeError("HF_TOKEN не найден. Укажи в .env или export HF_TOKEN=...")
        print("[DIAR] Loading pyannote/speaker-diarization-3.1...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        ).to(torch.device(DEVICE))
        print("[DIAR] Pipeline loaded.")
    return _pipeline


def load_audio_stereo(path: str):
    import av
    container = av.open(path)
    stream = container.streams.audio[0]
    sr = stream.sample_rate
    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        frames.append(arr)
    container.close()
    audio = np.concatenate(frames, axis=1)
    return torch.from_numpy(audio).float(), sr


def diarize(path: str) -> list[tuple[float, float, int]]:
    waveform, sr = load_audio_stereo(path)
    pipeline = get_pipeline()
    result = pipeline({"waveform": waveform, "sample_rate": sr})
    annotation = result.speaker_diarization

    raw_segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append((segment.start, segment.end, speaker))

    seen = {}
    counter = 0
    for start, end, spk in sorted(raw_segments, key=lambda x: x[0]):
        if spk not in seen:
            counter += 1
            seen[spk] = counter

    result_segments = [(s, e, seen[spk]) for s, e, spk in raw_segments]
    print(f"[DIAR] {len(result_segments)} segments, {len(seen)} speakers")
    return result_segments
