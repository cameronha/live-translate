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
    "font": "Atkinson Hyperlegible",  # falls back to Helvetica Neue if missing
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
    "speaker_colors": True,  # guess the voice of each phrase and color it (S key)
    "two_columns": False,    # Spanish on the left, English on the right (2 key)
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
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import get_suppressed_tokens
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
    guesser = None
    try:
        from speaker import SpeakerGuesser
        guesser = SpeakerGuesser()
    except Exception as e:  # colors are optional: go on without them
        print(f"speaker colors off: {e}", file=sys.stderr, flush=True)
    out_q.put(("status", "Listening…"))

    vad = VadOptions(threshold=0.5, min_speech_duration_ms=200,
                     min_silence_duration_ms=300, speech_pad_ms=150)
    step = int(cfg["update_every"] * SR)
    pause = int(cfg["pause_to_finish"] * SR)
    keep = int(0.5 * SR)

    def clean(text):
        text = text.strip()
        # Only drop short junk-only text. A long phrase with real content stays.
        if len(text) < 45 and any(j in text.lower() for j in JUNK):
            return ""
        return text

    def translate(audio):
        """Return (english, spanish). Spanish is "" unless two-column mode is on."""
        if not cfg["two_columns"]:
            segs, _ = model.transcribe(
                audio, language="es", task="translate", beam_size=1,
                temperature=0.0, condition_on_previous_text=False,
                without_timestamps=True, vad_filter=False)
            return clean(" ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.8)), ""
        # Two columns: listen to the audio once (the slow part), then write it
        # twice, in English and in Spanish.
        enc = model.encode(pad_or_trim(model.feature_extractor(audio[:30 * SR])))
        out = []
        for task in ("translate", "transcribe"):
            tok, prompt, suppress = writers[task]
            r = model.model.generate(
                enc, [prompt], beam_size=1, max_length=model.max_length,
                return_no_speech_prob=True, suppress_blank=True,
                suppress_tokens=suppress)[0]
            if r.no_speech_prob >= 0.8:
                return "", ""
            out.append(clean(tok.decode(r.sequences_ids[0])))
            if not out[0]:
                return "", ""
        return out[0], out[1]

    writers = {}
    for task in ("translate", "transcribe"):
        tok = Tokenizer(model.hf_tokenizer, True, task=task, language="es")
        writers[task] = (tok, model.get_prompt(tok, [], without_timestamps=True),
                         get_suppressed_tokens(tok, [-1]))

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
                out_q.put(("partial", ("", "")))
                showing_partial = False
            buf = buf[-keep:]
            checked = len(buf)
            continue

        start = max(0, speech[0]["start"] - int(0.2 * SR))
        last_end = speech[-1]["end"]
        # A pause in the middle of the buffer (it happens when the processor is
        # busy and audio piles up): finish the phrase before that pause.
        gap = next((i for i in range(len(speech) - 1)
                    if speech[i + 1]["start"] - speech[i]["end"] >= pause - int(0.15 * SR)), None)
        if gap is not None:
            seg = buf[start:speech[gap]["end"]]
            buf = buf[speech[gap]["end"]:]
            kind = "final"
        elif done or len(buf) - last_end >= pause:
            # Speaker paused: finish the phrase.
            seg = buf[start:last_end]
            buf = buf[max(last_end, len(buf) - keep):]
            kind = "final"
        elif len(buf) >= int(cfg["max_phrase"] * SR):  # read live: , and . keys change it
            # Nobody paused. Cut at the last small gap if there is one.
            cut = speech[-1]["start"] if len(speech) > 1 else len(buf)
            seg = buf[start:cut]
            buf = buf[cut:]
            kind = "final"
        else:
            seg = buf[start:]
            kind = "partial"
        text, spanish = translate(seg)
        # After a cut in the middle, look at the rest of the buffer without delay.
        checked = max(0, len(buf) - step) if gap is not None else len(buf)
        if text or showing_partial:
            out_q.put((kind, (text, spanish)))
        if kind == "final" and text and guesser and cfg["speaker_colors"]:
            # The text is already on screen. The color follows a moment later.
            out_q.put(("speaker", guesser.identify(seg)))
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

def register_fonts():
    """Make the fonts in the project's fonts/ folder usable by this app only.
    Nothing is installed on the Mac."""
    import ctypes
    import ctypes.util
    try:
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        ct = ctypes.CDLL(ctypes.util.find_library("CoreText"))
        cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
        cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
        ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
        ct.CTFontManagerRegisterFontsForURL.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        for f in sorted((HERE / "fonts").glob("*.ttf")):
            path = str(f).encode()
            url = cf.CFURLCreateFromFileSystemRepresentation(None, path, len(path), False)
            ct.CTFontManagerRegisterFontsForURL(url, 1, None)  # 1 = this process only
    except Exception as e:
        print(f"font registration failed: {e}", file=sys.stderr, flush=True)


HELP = "+/- size   [ ] lines   ↑↓ opacity   ←→ width   T top/bottom   S speaker colors   2 two columns   C clear   Esc quit"
SPEAKER_COLORS = ["#ffffff", "#ffd166", "#7fdbff", "#b8f2a0", "#ffa8c5", "#c9b6ff"]


class Overlay:
    def __init__(self, cfg, out_q, stop):
        import tkinter as tk
        from tkinter import font as tkfont
        self.tk, self.cfg, self.out_q, self.stop = tk, cfg, out_q, stop
        self.finals, self.partial, self.status = [], ("", ""), "Starting…"
        # finals: [english, speaker, spanish]   partial: (english, spanish)

        register_fonts()
        self.root = root = tk.Tk()
        self.set_title()
        try:  # the sombrero in the Dock and the app switcher while it runs
            self.icon = tk.PhotoImage(file=str(HERE / "icon.png"))
            root.iconphoto(True, self.icon)
        except Exception as e:
            print(f"icon not set: {e}", file=sys.stderr, flush=True)
        root.configure(bg="#111111")
        root.attributes("-topmost", True)
        root.attributes("-alpha", cfg["opacity"])
        family = cfg["font"] if cfg["font"] in tkfont.families() else "Helvetica Neue"
        self.font = tkfont.Font(family=family, size=cfg["font_size"])
        print(f"font: {self.font.actual('family')}", file=sys.stderr, flush=True)
        def column():
            t = tk.Text(root, wrap="word", bg="#111111", fg="#ffffff",
                        font=self.font, bd=0, highlightthickness=0,
                        padx=16, pady=8, cursor="arrow", takefocus=0,
                        spacing1=2, spacing2=4, spacing3=2)
            t.tag_configure("partial", foreground="#8a8f98")
            for i, color in enumerate(SPEAKER_COLORS):
                t.tag_configure(f"spk{i}", foreground=color)
            t.tag_configure("status", foreground="#777777")
            t.configure(state="disabled")
            return t
        self.spanish = column()   # left column, only in two-column mode
        self.text = column()      # English
        root.configure(bg="#333333")  # shows as a thin line between the columns
        root.grid_rowconfigure(0, weight=1)
        self.layout()

        self.place()
        root.bind("<Key>", self.on_key)
        root.bind("<Configure>", self.on_move)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.render()
        root.after(80, self.poll)

    def set_title(self):
        self.root.title(f"Live Translate    {HELP}   , . phrase limit ({self.cfg['max_phrase']:.0f} s)")

    def layout(self):
        two = self.cfg["two_columns"]
        self.root.grid_columnconfigure(0, weight=1 if two else 0, uniform="col" if two else "")
        self.root.grid_columnconfigure(1, weight=1, uniform="col" if two else "")
        if two:
            self.spanish.grid(row=0, column=0, sticky="nsew", padx=(0, 1))
        else:
            self.spanish.grid_remove()
        self.text.grid(row=0, column=1, sticky="nsew")

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
            self.finals, self.partial = [], ("", "")
            return self.render()
        elif k in ("comma", "less"):
            c["max_phrase"] = max(4.0, c["max_phrase"] - 2.0)
        elif k in ("period", "greater"):
            c["max_phrase"] = min(16.0, c["max_phrase"] + 2.0)
        elif k == "2":
            c["two_columns"] = not c["two_columns"]
        elif k in ("s", "S"):
            c["speaker_colors"] = not c["speaker_colors"]
        else:
            return
        self.font.configure(size=c["font_size"])
        self.root.attributes("-alpha", c["opacity"])
        self.set_title()
        self.layout()
        self.place()
        self.render()
        save_settings(c)

    def poll(self):
        changed = False
        try:
            while True:
                kind, text = self.out_q.get_nowait()  # text: a string, or (english, spanish)
                changed = True
                if kind == "status":
                    self.status = text
                elif kind == "partial":
                    self.partial = text
                elif kind == "speaker":
                    if self.finals:
                        self.finals[-1][1] = text
                else:
                    self.partial = ("", "")
                    if text[0]:
                        self.finals = (self.finals + [[text[0], None, text[1]]])[-40:]
        except queue.Empty:
            pass
        if changed:
            self.render()
        self.root.after(80, self.poll)

    def render(self):
        cols = [self.spanish, self.text] if self.cfg["two_columns"] else [self.text]
        for t in cols:
            t.configure(state="normal")
            t.delete("1.0", "end")
        if not (self.finals or self.partial[0]):
            self.text.insert("end", self.status, "status")
        elif len(cols) == 1:
            self.render_flow()
        else:
            self.render_rows()
        for t in cols:
            t.configure(state="disabled")
            t.see("end")

    def tag(self, spk):
        if self.cfg["speaker_colors"] and spk is not None:
            return f"spk{spk % len(SPEAKER_COLORS)}"
        return ()

    def render_flow(self):
        """One column: English phrases flow as one paragraph."""
        t = self.text
        for i, (phrase, spk, _) in enumerate(self.finals):
            t.insert("end", (" " if i else "") + phrase, self.tag(spk))
        if self.partial[0]:
            t.insert("end", (" " if self.finals else "") + self.partial[0], "partial")

    def render_rows(self):
        """Two columns: each phrase is a row. Spanish left, English right, tops aligned."""
        self.root.update_idletasks()  # the wrap width must be current
        rows = [(es, en, self.tag(spk)) for en, spk, es in self.finals[-12:]]
        if self.partial[0]:
            rows.append((self.partial[1], self.partial[0], "partial"))
        for n, (es, en, tag) in enumerate(rows):
            lines = []
            for t, phrase in ((self.spanish, es), (self.text, en)):
                if n:
                    t.insert("end", "\n")
                start = t.index("end-1c")
                t.insert("end", phrase, tag)
                count = t.count(start, "end-1c", "displaylines")
                lines.append((count[0] if isinstance(count, tuple) else count or 0) + 1)
            # Blank lines under the shorter side keep the next row aligned.
            for t, have in zip((self.spanish, self.text), lines):
                t.insert("end", "\n" * (max(lines) - have))

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
