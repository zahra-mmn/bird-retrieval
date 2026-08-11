"""Trim field recordings to the highest-energy window instead of naive truncation, so a long
silent lead-in (common in Xeno-canto field recordings) doesn't dominate what ImageBind embeds.
"""

import numpy as np
import librosa
import soundfile as sf


def isolate_call(in_path, out_path, target_sr: int = 16000, target_duration: float = 2.0,
                  frame_ms: float = 25) -> dict:
    wav, sr = librosa.load(str(in_path), sr=target_sr, mono=True)
    target_len = int(target_sr * target_duration)

    if len(wav) <= target_len:
        best = np.zeros(target_len, dtype=wav.dtype)
        best[: len(wav)] = wav
        start_sec = 0.0
    else:
        frame_len = max(1, int(target_sr * frame_ms / 1000))
        n_frames = max(1, (len(wav) - frame_len) // frame_len)
        energy = np.array([
            np.sum(wav[i * frame_len:(i + 1) * frame_len] ** 2) for i in range(n_frames)
        ])
        window_frames = max(1, target_len // frame_len)
        if len(energy) >= window_frames:
            window_energy = np.convolve(energy, np.ones(window_frames), mode="valid")
            best_frame = int(np.argmax(window_energy))
        else:
            best_frame = 0
        start = best_frame * frame_len
        best = wav[start:start + target_len]
        if len(best) < target_len:
            best = np.pad(best, (0, target_len - len(best)))
        start_sec = start / target_sr

    snr_estimate = _estimate_snr(wav, best)
    sf.write(str(out_path), best, target_sr)
    return {"start_sec": start_sec, "snr_estimate": snr_estimate}


def _estimate_snr(full_wav: np.ndarray, call_window: np.ndarray) -> float:
    call_power = float(np.mean(call_window ** 2)) + 1e-10
    noise_floor = float(np.percentile(full_wav ** 2, 10)) + 1e-10
    return float(10 * np.log10(call_power / noise_floor))
