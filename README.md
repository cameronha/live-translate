# Live Translate

A floating caption strip for the Mac. It listens to the laptop microphone and shows Spanish speech as English text. All work is local. No cloud, no API keys, no saved audio or text.

## Limits

- Spanish to English only. Apple silicon Mac only.
- The delay is about 2 seconds. The translation is rough: good enough to follow a meeting, not to quote a person.
- The speaker colors are a guess for each phrase.
- First start: macOS asks for microphone permission. Click Allow.

## Use

- Start: click `Live Translate.app` (or double-click `Start Translator.command`).
- Stop: `Esc`.
- Keys (click the strip first): `+`/`-` font size, `[`/`]` lines, up/down opacity, left/right width, `T` top/bottom, `S` speaker colors, `2` two columns (Spanish | English), `,`/`.` phrase limit shorter/longer, `C` clear.
- Settings are saved in `settings.json`.

## How it works

Microphone -> voice detection (Silero VAD) -> Whisper Small with the built-in translate task -> Tkinter window. A small voiceprint model (WeSpeaker ResNet34) guesses the speaker of each phrase for the text color.

- `translate.py`: the engine and the window.
- `speaker.py`: speaker colors.
- `launcher.c`: the small native program inside the app. It runs Python through a link in the app, so macOS shows the overlay as "Live Translate".
- `make_icon.py`: draws the sombrero icon.

## Set up on a new Mac (Apple silicon, python.org Python 3.12)

The models and the virtual environment are not in this repository. Run these in the project folder:

```bash
python3.12 -m venv venv
./venv/bin/pip install faster-whisper sounddevice
./venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Systran/faster-whisper-small', local_dir='models/small'); d('Wespeaker/wespeaker-voxceleb-resnet34-LM', local_dir='models/speaker', allow_patterns='*.onnx')"
ln -sf /Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python "Live Translate.app/Contents/MacOS/python"
clang -arch arm64 -O1 -o "Live Translate.app/Contents/MacOS/launcher" launcher.c
codesign --force -s - "Live Translate.app"
```

Test without a microphone:

```bash
./venv/bin/python translate.py --file test/spanish.wav --no-ui
```

## Credits

- Font: Atkinson Hyperlegible (SIL Open Font License, see `fonts/OFL.txt`).
- Speech: faster-whisper. Voiceprints: WeSpeaker (CC BY 4.0).

## License

MIT. See `LICENSE`. The font and the models have their own licenses (see Credits).
