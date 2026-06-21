# Data Science & ML Notebooks

A curated collection of **Google Colab notebooks** for audio processing, transcription, and machine learning workflows.

This repository supports **two ways of running Colab notebooks**:
1. **Via the Google Colab web interface** (browser-based)
2. **Via the Google Colab extension in Visual Studio Code** (editor-first workflow)

The notebooks are organized accordingly to reduce confusion and make setup explicit.

---

## Repository Structure

```
.
├── colab-browser/                       # Standalone notebooks for the Colab web interface
│   ├── Transcribe_Audio_Whisper.ipynb
│   ├── speech_to_text_diarization.ipynb
│   ├── VibeVoice_ASR_Colab.ipynb
│   ├── Qwen3_ASR_Colab.ipynb
│   ├── Qwen3_TTS.ipynb
│   ├── H2O_Flow_Google_Colab.ipynb
│   └── README.md
├── colab-vscode/                        # Notebooks for the VS Code + Colab extension
│   ├── Colab_in_VSCode.ipynb
│   └── README.md
├── LICENSE
└── README.md
```

### How to choose the right folder
- Use **`colab-browser/`** (via the **Open in Colab** links below) for a zero-setup, browser-only experience  
- Use **`colab-vscode/`** if you prefer working in VS Code with Colab-backed compute

---

## Colab Notebooks (browser)

Standalone notebooks for the Colab web interface, with **Open in Colab** badges and per-notebook descriptions, live in [`colab-browser/`](colab-browser/#readme).

---

## Colab + VS Code Notebooks (`colab-vscode/`)

These notebooks are intended to be run using the **Google Colab extension for Visual Studio Code**, allowing you to:

- Use VS Code for editing, navigation, and tooling
- Execute code on Colab-backed runtimes (CPU/GPU)
- Follow a Git-first, reproducible workflow

### Colab in VS Code — Demo Notebook

This demo notebook validates and bootstraps a Colab runtime inside VS Code by:
- Checking GPU and CUDA availability
- Inspecting the runtime filesystem
- Cloning project repositories directly into the ephemeral runtime

See the README inside `colab-vscode/` for:
- VS Code setup instructions
- Runtime selection guidance
- Workflow notes and constraints

---

## Notes on Colab runtimes

- Colab runtimes are **ephemeral**
- Local files may be lost on restart or disconnect
- Commit changes to Git or export outputs if persistence is required

This repository intentionally favors **Git-based workflows** over Google Drive mounts for reproducibility and portability.

---

## License

MIT — built with AI.  
See the [`LICENSE`](LICENSE) file for full details.
