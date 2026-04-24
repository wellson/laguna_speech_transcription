import json
import queue
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sounddevice as sd
import mlx_whisper
from deep_translator import GoogleTranslator
from translate import Translator as MyMemoryTranslator

DEVICE = "BlackHole 2ch"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
MODEL_REPO = "mlx-community/whisper-base.en-mlx"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8765
TRANSLATOR_BACKEND = "google"  # "google" | "mymemory"

audio_q: queue.Queue[np.ndarray] = queue.Queue()
translate_q: queue.Queue = queue.Queue()
google_translator = GoogleTranslator(source="en", target="pt")
mymemory_translator = MyMemoryTranslator(from_lang="en", to_lang="pt-BR")


def _is_bad_translation(original: str, translated: str) -> bool:
    if not translated:
        return True
    t = translated.strip()
    if not t:
        return True
    # MyMemory devolve mensagens crus tipo "MYMEMORY WARNING: ...", "PLEASE SELECT TWO DISTINCT LANGUAGES",
    # ou ecos com marcador de idioma ("EN>> ...", "PT-BR>> ..."), principalmente após estourar cota.
    upper = t.upper()
    bad_markers = ("MYMEMORY WARNING", "PLEASE SELECT TWO DISTINCT LANGUAGES", "INVALID", "QUERY LENGTH LIMIT")
    if any(m in upper for m in bad_markers):
        return True
    if re.match(r"^[A-Z-]{2,6}>>\s*", t):
        return True
    # mesma string que entrou = não traduziu
    if t.strip().lower() == original.strip().lower():
        return True
    return False


def translate_text(text: str) -> str:
    if TRANSLATOR_BACKEND == "mymemory":
        try:
            pt = mymemory_translator.translate(text)
        except Exception:
            pt = ""
        if _is_bad_translation(text, pt):
            # fallback automático para o Google quando o MyMemory devolve lixo/cota
            return google_translator.translate(text)
        return pt
    return google_translator.translate(text)


# frases curtas que o Whisper "alucina" em silêncio/ruído baixo — vêm dos dados de treino (YouTube)
HALLUCINATION_PHRASES = {
    "thank you.", "thank you", "thanks.", "thanks", "thanks for watching.",
    "thanks for watching", "thank you for watching.", "thank you for watching",
    "please subscribe.", "please subscribe", "subscribe.", "you", "you.",
    "bye.", "bye", "bye-bye.", "bye-bye", "goodbye.", "goodbye",
    ".", "okay.", "okay", "ok.", "ok", "yeah.", "yeah",
    "...", "music", "applause", "silence",
}


def is_hallucination(text: str) -> bool:
    """Detecta alucinações típicas do Whisper: frases curtas de silêncio ('Thank you.'),
    loops ('Sum of Sum of...'), n-gramas repetidos e baixa diversidade.
    """
    normalized = text.lower().strip()
    if normalized in HALLUCINATION_PHRASES:
        return True

    words = normalized.split()
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

        # gate de energia mais agressivo: silêncio/ruído baixo é a principal fonte
        # de alucinação do Whisper ("Thank you.", "you", etc.)
        peak = float(np.max(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if peak < 0.01 or rms < 0.003:
            continue

        # passa o numpy array direto pro whisper — evita o round-trip de escrever/ler .wav
        audio_f32 = chunk.astype(np.float32).flatten()
        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=MODEL_REPO,
            language="en",
            task="transcribe",
            suppress_tokens=[],
            condition_on_previous_text=False,
            compression_ratio_threshold=2.2,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            # fallback curto: só retenta 1x se o segmento estourar threshold
            temperature=(0.0, 0.4),
            fp16=True,
        )

        en = str(result["text"])
        en = re.sub(r"[\[(][^)\]]*[\])]", "", en).strip()
        if not en:
            continue

        if is_hallucination(en):
            continue

        # publica o EN imediatamente e delega a tradução pro translator_worker —
        # o próximo chunk pode começar a transcrever enquanto o Google traduz este
        translate_q.put((en, time.time()))


def translator_worker():
    while True:
        item = translate_q.get()
        if item is None:
            break
        en, ts = item
        try:
            pt = translate_text(en)
        except Exception as e:
            pt = f"[erro tradução: {e}]"

        print(f"[EN] {en}\n[PT] {pt}", flush=True)
        publish({
            "type": "message",
            "en": en,
            "pt": pt,
            "ts": ts,
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
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=translator_worker, daemon=True).start()
    try:
        recorder()
    except KeyboardInterrupt:
        print("\nEncerrando...")


if __name__ == "__main__":
    main()
