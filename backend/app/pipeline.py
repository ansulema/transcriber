"""Пайплайн: ASR → Diarization → Merge → Egor-формат."""

from app.asr import transcribe
from app.diarize import diarize


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def find_speaker(mid_time: float, speaker_segments: list) -> int:
    best_spk = 1
    best_dist = float("inf")
    for spk_start, spk_end, spk in speaker_segments:
        if spk_start <= mid_time <= spk_end:
            return spk
        dist = min(abs(mid_time - spk_start), abs(mid_time - spk_end))
        if dist < best_dist:
            best_dist = dist
            best_spk = spk
    return best_spk


def run_pipeline(audio_path: str, language: str = "ru") -> dict:
    import os
    asr_segments, duration = transcribe(audio_path, language=language)
    speaker_segments = diarize(audio_path)

    utterances = []
    for seg in asr_segments:
        mid = (seg["start"] + seg["end"]) / 2.0
        speaker = find_speaker(mid, speaker_segments)
        utterances.append({
            "speaker": speaker,
            "text": seg["text"],
            "start_sec": seg["start"],
            "end_sec": seg["end"],
            "start_ts": format_ts(seg["start"]),
            "end_ts": format_ts(seg["end"]),
        })

    transcript_lines = []
    for u in utterances:
        transcript_lines.append(f"{u['speaker']}   {u['text']}")
        transcript_lines.append(f"    {u['start_ts']} - {u['end_ts']}")
    transcript_text = "\n".join(transcript_lines)

    return {
        "filename": os.path.basename(audio_path),
        "duration_sec": duration,
        "num_segments": len(utterances),
        "segments": utterances,
        "transcript_text": transcript_text,
    }
