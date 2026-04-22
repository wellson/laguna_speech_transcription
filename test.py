import json
import queue
import re
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import mlx_whisper
from deep_translator import GoogleTranslator
from translate import Translator as MyMemoryTranslator

DEVICE = "BlackHole 2ch"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
MODEL_REPO = "mlx-community/whisper-small-mlx"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8765
TRANSLATOR_BACKEND = "mymemory"  # "google" | "mymemory"

audio_q: queue.Queue[np.ndarray] = queue.Queue()
google_translator = GoogleTranslator(source="en", target="pt")
mymemory_translator = MyMemoryTranslator(from_lang="en", to_lang="pt-BR")


def translate_text(text: str) -> str:
    if TRANSLATOR_BACKEND == "mymemory":
        return mymemory_translator.translate(text)
    return google_translator.translate(text)


def is_hallucination(text: str) -> bool:
    """Detecta loops típicos de Whisper ('Sum of Sum of...', 'make sure that we make sure...').

    Critérios: baixa diversidade de palavras OU n-gramas curtos muito repetidos
    OU repetição imediata consecutiva do mesmo trecho.
    """
    words = text.lower().split()
    if len(words) < 4:
        return False

    # repetição imediata: o mesmo bloco de 1–4 palavras aparece ≥3x em sequência
    # pega casos curtos tipo "sum of sum of sum of"
    for n in (1, 2, 3, 4):
        if len(words) < n * 3:
            continue
        for i in range(len(words) - n * 3 + 1):
            block = words[i:i + n]
            if words[i + n:i + 2 * n] == block and words[i + 2 * n:i + 3 * n] == block:
                return True

    if len(words) < 8:
        return False

    # diversidade: únicas / total. loops costumam ficar em 0.25 ou abaixo
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio <= 0.3:
        return True

    # repetição de n-gramas de 2–4 palavras
    for n in (2, 3, 4):
        if len(words) < n * 3:
            continue
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts: dict[tuple, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        top = max(counts.values())
        if top >= 3 and top / len(grams) > 0.3:
            return True

    return False

subscribers: list[queue.Queue] = []
subscribers_lock = threading.Lock()


def publish(event: dict) -> None:
    data = json.dumps(event, ensure_ascii=False)
    with subscribers_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            subscribers.remove(q)


INDEX_HTML = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<title>Laguna — transcrição ao vivo</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 14px 20px;
    background: #1e293b;
    border-bottom: 1px solid #334155;
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
  }
  header .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 10px #ef4444;
    animation: pulse 1.4s infinite;
  }
  header.connected .dot {
    background: #22c55e;
    box-shadow: 0 0 10px #22c55e;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column-reverse;
    gap: 12px;
  }
  .msg {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 14px;
    background: #1e293b;
    border: 1px solid #334155;
    animation: slideIn .2s ease-out;
  }
  .msg .en { color: #94a3b8; font-size: 13px; line-height: 1.4; }
  .msg .pt { color: #f1f5f9; font-size: 15px; line-height: 1.45; margin-top: 4px; }
  .msg .time { color: #64748b; font-size: 11px; margin-top: 6px; }
  .tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .5px;
    padding: 2px 6px;
    border-radius: 4px;
    margin-right: 6px;
    vertical-align: middle;
  }
  .tag.en { background: #1e3a8a; color: #bfdbfe; }
  .tag.pt { background: #14532d; color: #bbf7d0; }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .empty {
    margin: auto;
    color: #475569;
    font-size: 14px;
  }
</style>
</head>
<body>
<header id="hdr"><span class="dot"></span><span>Laguna — transcrição ao vivo</span></header>
<div id="messages"><div class="empty" id="empty">aguardando áudio…</div></div>
<script>
  const msgs = document.getElementById('messages');
  const hdr = document.getElementById('hdr');
  const empty = document.getElementById('empty');
  const fmt = (t) => new Date(t).toLocaleTimeString();
  const es = new EventSource('/stream');
  es.onopen = () => hdr.classList.add('connected');
  es.onerror = () => hdr.classList.remove('connected');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'message') {
      if (empty) empty.remove();
      const div = document.createElement('div');
      div.className = 'msg';
      const en = document.createElement('div'); en.className = 'en';
      en.innerHTML = '<span class="tag en">EN</span>';
      en.appendChild(document.createTextNode(ev.en));
      const pt = document.createElement('div'); pt.className = 'pt';
      pt.innerHTML = '<span class="tag pt">PT-BR</span>';
      pt.appendChild(document.createTextNode(ev.pt));
      const tm = document.createElement('div'); tm.className = 'time'; tm.textContent = fmt(ev.ts * 1000);
      div.append(en, pt, tm);
      msgs.prepend(div);
      msgs.scrollTop = 0;
    }
  };
</script>
</body>
</html>
"""


class QuietHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # silencia desconexões normais do navegador (SSE fecha a conexão ao recarregar)
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:
        pass

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, OSError):
            self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q: queue.Queue = queue.Queue()
            with subscribers_lock:
                subscribers.append(q)
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except Exception:
                pass
            finally:
                with subscribers_lock:
                    if q in subscribers:
                        subscribers.remove(q)
            return

        self.send_response(404)
        self.end_headers()


def http_server() -> None:
    srv = QuietHTTPServer((HTTP_HOST, HTTP_PORT), Handler)
    srv.serve_forever()


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
    while True:
        chunk = audio_q.get()
        if chunk is None:
            break

        peak = float(np.max(np.abs(chunk)))
        if peak < 1e-4:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            wav.write(f.name, SAMPLE_RATE, chunk)
            result = mlx_whisper.transcribe(
                f.name,
                path_or_hf_repo=MODEL_REPO,
                language="en",
                task="transcribe",
                # não suprime nenhum token — mantém palavrões e gírias como falados
                suppress_tokens=[],
                # evita que o modelo condicione no texto anterior e entre em loops tipo "Sum of Sum of..."
                condition_on_previous_text=False,
                # descarta segmentos com compressão suspeita (sintoma clássico de loop) e baixa confiança
                compression_ratio_threshold=2.2,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
            )

        en = str(result["text"])
        # remove tokens especiais tipo [BLANK_AUDIO], [MUSIC], (inaudible), etc.
        en = re.sub(r"[\[(][^)\]]*[\])]", "", en).strip()
        if not en:
            continue

        if is_hallucination(en):
            continue

        try:
            pt = translate_text(en)
        except Exception as e:
            pt = f"[erro tradução: {e}]"

        print(f"[EN] {en}\n[PT] {pt}", flush=True)
        publish({
            "type": "message",
            "en": en,
            "pt": pt,
            "ts": time.time(),
        })


def main():
    threading.Thread(target=http_server, daemon=True).start()
    url = f"http://{HTTP_HOST}:{HTTP_PORT}"
    print(f"Interface: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print(f"Ouvindo {DEVICE} em chunks de {CHUNK_SECONDS}s. Ctrl+C para sair.")
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        recorder()
    except KeyboardInterrupt:
        print("\nEncerrando...")


if __name__ == "__main__":
    main()
