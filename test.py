import queue
import sys
import tempfile
import threading

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import mlx_whisper
from deep_translator import GoogleTranslator

DEVICE = "BlackHole 2ch"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
MODEL_REPO = "mlx-community/whisper-small-mlx"

audio_q: queue.Queue[np.ndarray] = queue.Queue()
translator = GoogleTranslator(source="en", target="pt")


def recorder():
    samples_per_chunk = CHUNK_SECONDS * SAMPLE_RATE
    buf = np.zeros((0, 1), dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal buf
        if status:
            print(status, file=sys.stderr)
        buf = np.concatenate([buf, indata.copy()])
        while len(buf) >= samples_per_chunk:
            audio_q.put(buf[:samples_per_chunk].copy())
            buf = buf[samples_per_chunk:]

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=DEVICE,
        callback=callback,
    ):
        threading.Event().wait()


def worker():
    idx = 0
    while True:
        chunk = audio_q.get()
        if chunk is None:
            break
        idx += 1

        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(chunk)))
        print(f"[chunk {idx}] rms={rms:.5f} peak={peak:.5f}", flush=True)

        if peak < 1e-4:
            print("  -> silêncio, pulando", flush=True)
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            wav.write(f.name, SAMPLE_RATE, chunk)
            result = mlx_whisper.transcribe(
                f.name,
                path_or_hf_repo=MODEL_REPO,
                language="en",
                task="transcribe",
            )

        en = str(result["text"]).strip()
        if not en:
            print("  -> sem transcrição", flush=True)
            continue

        try:
            pt = translator.translate(en)
        except Exception as e:
            pt = f"[erro tradução: {e}]"

        print(f"[EN] {en}\n[PT] {pt}", flush=True)


def main():
    print(f"Ouvindo {DEVICE} em chunks de {CHUNK_SECONDS}s. Ctrl+C para sair.")
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        recorder()
    except KeyboardInterrupt:
        print("\nEncerrando...")


if __name__ == "__main__":
    main()
