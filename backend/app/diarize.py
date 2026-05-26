"""Pyannote speaker diarization 3.1 — lazy singleton."""

import os
import torch
import numpy as np
from pathlib import Path
from pyannote.audio import Pipeline

DEVICE = "cuda"

DIAR_PARAMS = {
    "clustering": {
        "method": "centroid",
        "threshold": 0.72,
        "min_cluster_size": 15,
    },
    "segmentation": {
        "min_duration_off": 0.1,
    },
}

_LOOKUP_PATHS = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
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

        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

        print("[DIAR] Loading pyannote/speaker-diarization-3.1...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )

        _pipeline.instantiate(DIAR_PARAMS)

        if hasattr(_pipeline, "_segmentation") and hasattr(_pipeline._segmentation, "batch_size"):
            _pipeline._segmentation.batch_size = 64

        if hasattr(_pipeline, "_embedding") and hasattr(_pipeline._embedding, "batch_size"):
            _pipeline._embedding.batch_size = 64

        _pipeline.to(device)

        print(f"[DIAR] Pipeline loaded on {device}.")
        print(f"[DIAR] Params: {DIAR_PARAMS}")

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

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(1, np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)

    return torch.from_numpy(audio).float(), sr


def diarize(path: str, num_speakers: int | None = None) -> list[tuple[float, float, int]]:
    waveform, sr = load_audio_stereo(path)
    pipeline = get_pipeline()

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    result = pipeline(
        {"waveform": waveform, "sample_rate": sr},
        **kwargs,
    )

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
