# Data Science & ML Notebooks

A curated collection of **Google Colab notebooks** for audio processing, transcription, and machine learning workflows.

This repository supports **two ways of running Colab notebooks**:
1. **Via the Google Colab web interface** (browser-based)
2. **Via the Google Colab extension in Visual Studio Code** (editor-first workflow)

The notebooks are organized accordingly to reduce confusion and make setup explicit.

---

## Repository Structure

```
- `notebooks/`
  - `colab-web/` — Notebooks intended to be run in the Colab browser UI  
  - `colab-vscode/` — Notebooks intended to be run via VS Code + Colab extension  
- `LICENSE`
- `README.md`
```


### How to choose the right folder
- Use **`colab-web/`** if you want a zero-setup, browser-only experience  
- Use **`colab-vscode/`** if you prefer working in VS Code with Colab-backed compute

---

## Colab Web Notebooks (`notebooks/colab-web/`)

These notebooks are designed to be opened and run directly in the **Google Colab web interface**.

### Transcribe Audio with OpenAI Whisper
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/notebooks/colab-web/Transcribe_Audio_Whisper.ipynb)

Demonstrates how to transcribe audio files using OpenAI’s Whisper model via the `whisper` Python package.  
Supports common audio formats (MP3, WAV) and provides a simple pipeline for uploading, transcribing, and saving transcripts.

---

### Audio Diarization with `gpt-4o-transcribe-diarize`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/notebooks/colab-web/speech_to_text_diarization.ipynb)

Demonstrates how to perform audio diarization using OpenAI’s `gpt-4o-transcribe-diarize` model.  
The notebook processes audio in chunks and generates a diarized transcript that identifies different speakers.

> An OpenAI API key is required. You can obtain one from the [OpenAI platform](https://platform.openai.com/).

---

### H2O Flow in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/notebooks/colab-web/H2O_Flow_Google_Colab.ipynb)

Provides a step-by-step guide to setting up and accessing **H2O Flow** inside a Google Colab environment.  
The notebook installs required dependencies, initializes the H2O server, and uses `localtunnel` to expose a public URL for the H2O Flow web interface.

---

## Colab + VS Code Notebooks (`notebooks/colab-vscode/`)

These notebooks are intended to be run using the **Google Colab extension for Visual Studio Code**, allowing you to:

- Use VS Code for editing, navigation, and tooling
- Execute code on Colab-backed runtimes (CPU/GPU)
- Follow a Git-first, reproducible workflow

### Colab in VS Code — Demo Notebook

This demo notebook validates and bootstraps a Colab runtime inside VS Code by:
- Checking GPU and CUDA availability
- Inspecting the runtime filesystem
- Cloning project repositories directly into the ephemeral runtime

See the README inside `notebooks/colab-vscode/` for:
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

This project is licensed under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for full details.
