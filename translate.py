#!/usr/bin/env python3
"""Live Spanish -> English caption overlay. Fully local, no network.

Microphone -> voice detection -> Whisper (translate task) -> floating window.

Run:   ./venv/bin/python translate.py
Test:  ./venv/bin/python translate.py --file some_spanish_audio.wav [--no-ui]
"""
import argparse
import json
import platform
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"
SR = 16000  # Whisper wants 16 kHz audio

DEFAULTS = {
    "model": "small",      # "base" is faster but much rougher
    "font_size": 22,
    "lines": 3,
    "width": 900,
    "opacity": 0.85,
    "x": None,             # None = centered
    "y": None,             # None = near the bottom
    # Timing (seconds). Lower = faster text, more CPU.
    "update_every": 1.0,   # how often to refresh the in-progress phrase
    "pause_to_finish": 0.7,  # silence that ends a phrase
    "max_phrase": 12.0,    # force a cut if nobody pauses
}

# Whisper sometimes invents these on noise. Drop them.
JUNK = ("thanks for watching", "thank you for watching", "subtitles by",
        "subscribe", "amara.org")


def load_settings():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def save_settings(cfg):
    try:
        SETTINGS_FILE.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass


# ---------------------------------------------------------------- engine

def engine(audio_q, out_q, stop, cfg):
    """Background thread. Reads audio chunks, emits ("partial"|"final"|"status", text)."""
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    out_q.put(("status", "Loading model…"))
    try:
        # Use a model in the project's models/ folder if it is there.
        local = HERE / "models" / cfg["model"]
        model = WhisperModel(str(local) if local.is_dir() else cfg["model"],
                             device="cpu", compute_type="int8",
                             cpu_threads=4, local_files_only=True)
    except Exception as e:  # model not on disk, etc.
        out_q.put(("status", f"Could not load model '{cfg['model']}': {e}"))
        return
    out_q.put(("status", "Listening…"))

    vad = VadOptions(threshold=0.5, min_speech_duration_ms=200,
                     min_silence_duration_ms=300, speech_pad_ms=150)
    step = int(cfg["update_every"] * SR)
    pause = int(cfg["pause_to_finish"] * SR)
    max_len = int(cfg["max_phrase"] * SR)
    keep = int(0.5 * SR)

    def translate(audio):
        segs, _ = model.transcribe(
            audio, language="es", task="translate", beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
            without_timestamps=True, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.8).strip()
        # Only drop short junk-only text. A long phrase with real content stays.
        if len(text) < 45 and any(j in text.lower() for j in JUNK):
            return ""
        return text

    buf = np.zeros(0, np.float32)
    checked = 0          # buffer length at the last check
    showing_partial = False
    done = False
    heard = [0, 0]       # samples received, warned yet
    while not stop.is_set() and not done:
        try:
            chunks = [audio_q.get(timeout=0.2)]
        except queue.Empty:
            continue
        while True:  # drain, so we never fall behind
            try:
                chunks.append(audio_q.get_nowait())
            except queue.Empty:
                break
        if chunks[-1] is None:  # end of a test file
            done = True
        chunks = [c for c in chunks if c is not None]
        if chunks:
            buf = np.concatenate([buf] + chunks)
            if heard is not None:  # is the microphone giving sound?
                heard[0] += sum(len(c) for c in chunks)
                peak = max(float(np.abs(c).max()) for c in chunks)
                if peak > 0:
                    print(f"microphone ok after {heard[0] / SR:.1f}s", file=sys.stderr, flush=True)
                    if heard[1]:
                        out_q.put(("status", "Listening…"))
                    heard = None
                elif heard[0] > 3 * SR and not heard[1]:
                    heard[1] = 1  # warned once; keep watching
                    print("microphone silent (permission?)", file=sys.stderr, flush=True)
                    out_q.put(("status", "No sound from the microphone yet. If macOS asks for "
                                         "microphone access, click Allow."))
        if len(buf) - checked < step and not done:
            continue

        speech = get_speech_timestamps(buf, vad)
        if not speech:  # silence: no Whisper call at all
            if showing_partial:
                out_q.put(("partial", ""))
                showing_partial = False
            buf = buf[-keep:]
            checked = len(buf)
            continue

        start = max(0, speech[0]["start"] - int(0.2 * SR))
        last_end = speech[-1]["end"]
        if done or len(buf) - last_end >= pause:
            # Speaker paused: finish the phrase.
            text = translate(buf[start:last_end])
            buf = buf[max(last_end, len(buf) - keep):]
            kind = "final"
        elif len(buf) >= max_len:
            # Nobody paused. Cut at the last small gap if there is one.
            cut = speech[-1]["start"] if len(speech) > 1 else len(buf)
            text = translate(buf[start:cut])
            buf = buf[cut:]
            kind = "final"
        else:
            text = translate(buf[start:])
            kind = "partial"
        checked = len(buf)
        if text or showing_partial:
            out_q.put((kind, text))
        showing_partial = kind == "partial" and bool(text)
    out_q.put(("status", "Stopped."))


# ---------------------------------------------------------------- audio in

def start_mic(audio_q):
    import sounddevice as sd
    stream = sd.InputStream(
        samplerate=SR, channels=1, dtype="float32", blocksize=SR // 10,
        callback=lambda data, frames, t, status: audio_q.put(data[:, 0].copy()))
    stream.start()
    return stream


def feed_file(path, audio_q, stop):
    """Test mode: play a file into the engine at real-time speed."""
    from faster_whisper.audio import decode_audio
    audio = decode_audio(path, sampling_rate=SR)
    n = SR // 10
    for i in range(0, len(audio), n):
        if stop.is_set():
            return
        audio_q.put(audio[i:i + n])
        time.sleep(0.1)
    audio_q.put(np.zeros(SR, np.float32))  # one second of silence to close the phrase
    audio_q.put(None)


# ---------------------------------------------------------------- window

HELP = "+/- size   [ ] lines   ↑↓ opacity   ←→ width   T top/bottom   C clear   Esc quit"


class Overlay:
    def __init__(self, cfg, out_q, stop):
        import tkinter as tk
        from tkinter import font as tkfont
        self.tk, self.cfg, self.out_q, self.stop = tk, cfg, out_q, stop
        self.finals, self.partial, self.status = "", "", "Starting…"

        self.root = root = tk.Tk()
        root.title("Live Translate    " + HELP)
        root.configure(bg="#111111")
        root.attributes("-topmost", True)
        root.attributes("-alpha", cfg["opacity"])
        self.font = tkfont.Font(family="Helvetica Neue", size=cfg["font_size"])
        self.text = tk.Text(root, wrap="word", bg="#111111", fg="#ffffff",
                            font=self.font, bd=0, highlightthickness=0,
                            padx=16, pady=8, cursor="arrow", takefocus=0,
                            spacing1=2, spacing3=2)
        self.text.tag_configure("partial", foreground="#9fb4c8")
        self.text.tag_configure("status", foreground="#777777")
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self.place()
        root.bind("<Key>", self.on_key)
        root.bind("<Configure>", self.on_move)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.render()
        root.after(80, self.poll)

    def height(self):
        return self.cfg["lines"] * (self.font.metrics("linespace") + 4) + 18

    def place(self, reset_pos=False):
        c, r = self.cfg, self.root
        w, h = c["width"], self.height()
        if reset_pos or c["x"] is None or c["y"] is None:
            c["x"] = (r.winfo_screenwidth() - w) // 2
            c["y"] = r.winfo_screenheight() - h - 110
        r.geometry(f"{w}x{h}+{c['x']}+{c['y']}")

    def on_move(self, e):
        if e.widget is self.root:  # remember drags and edge-resizes
            self.cfg["x"], self.cfg["y"] = self.root.winfo_x(), self.root.winfo_y()
            self.cfg["width"] = self.root.winfo_width()

    def on_key(self, e):
        c, k = self.cfg, e.keysym
        if k == "Escape":
            return self.quit()
        if k in ("plus", "equal"):
            c["font_size"] = min(72, c["font_size"] + 2)
        elif k in ("minus", "underscore"):
            c["font_size"] = max(10, c["font_size"] - 2)
        elif k == "bracketright":
            c["lines"] = min(12, c["lines"] + 1)
        elif k == "bracketleft":
            c["lines"] = max(1, c["lines"] - 1)
        elif k == "Up":
            c["opacity"] = min(1.0, round(c["opacity"] + 0.05, 2))
        elif k == "Down":
            c["opacity"] = max(0.2, round(c["opacity"] - 0.05, 2))
        elif k == "Right":
            c["width"] = min(self.root.winfo_screenwidth(), c["width"] + 60)
        elif k == "Left":
            c["width"] = max(300, c["width"] - 60)
        elif k in ("t", "T"):
            top = c["y"] is not None and c["y"] < self.root.winfo_screenheight() // 2
            c["y"] = (self.root.winfo_screenheight() - self.height() - 110) if top else 40
        elif k in ("c", "C"):
            self.finals = self.partial = ""
            return self.render()
        else:
            return
        self.font.configure(size=c["font_size"])
        self.root.attributes("-alpha", c["opacity"])
        self.place()
        self.render()
        save_settings(c)

    def poll(self):
        changed = False
        try:
            while True:
                kind, text = self.out_q.get_nowait()
                changed = True
                if kind == "status":
                    self.status = text
                elif kind == "partial":
                    self.partial = text
                else:
                    self.partial = ""
                    if text:
                        self.finals = (self.finals + " " + text).strip()[-1500:]
        except queue.Empty:
            pass
        if changed:
            self.render()
        self.root.after(80, self.poll)

    def render(self):
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        if self.finals or self.partial:
            t.insert("end", self.finals)
            if self.partial:
                t.insert("end", (" " if self.finals else "") + self.partial, "partial")
        else:
            t.insert("end", self.status, "status")
        t.configure(state="disabled")
        t.see("end")

    def quit(self):
        self.stop.set()
        save_settings(self.cfg)
        self.root.destroy()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="test with an audio file instead of the microphone")
    ap.add_argument("--no-ui", action="store_true", help="print to the terminal, no window")
    args = ap.parse_args()

    cfg = load_settings()
    print(f"start: {platform.machine()} model={cfg['model']}", file=sys.stderr, flush=True)
    audio_q, out_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    worker = threading.Thread(target=engine, args=(audio_q, out_q, stop, cfg), daemon=True)
    worker.start()

    stream = None
    if args.file:
        threading.Thread(target=feed_file, args=(args.file, audio_q, stop), daemon=True).start()
    else:
        try:
            stream = start_mic(audio_q)
        except Exception as e:
            out_q.put(("status", f"Microphone problem: {e}"))

    try:
        if args.no_ui:
            t0 = time.time()
            while worker.is_alive() or not out_q.empty():
                try:
                    kind, text = out_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                print(f"{time.time() - t0:6.1f}s  {kind:8s} {text}", flush=True)
        else:
            Overlay(cfg, out_q, stop).root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if stream:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    main()
