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
  - `colab-vscode/` — Notebooks intended to be run via VS Code + Colab extension  
- `LICENSE`
- `README.md`
```

### How to choose the right folder
- Use the **Open in Colab** links below if you want a zero-setup, browser-only experience  
- Use **`notebooks/colab-vscode/`** if you prefer working in VS Code with Colab-backed compute

---

## Colab Notebooks (run in the browser)

These notebooks are designed to be opened and run directly in the **Google Colab web interface**.

# Transcribe Audio with OpenAI Whisper
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/Transcribe_Audio_Whisper.ipynb)

This notebook demonstrates how to transcribe audio files using OpenAI's Whisper model via the whisper Python package. It supports common audio formats (e.g., MP3, WAV) and provides a simple pipeline for uploading, transcribing, and saving transcripts.

# Audio Diarization with `gpt-4o-transcribe-diarize`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/speech_to_text_diarization.ipynb)

This notebook demonstrates how to perform audio diarization using OpenAI's `gpt-4o-transcribe-diarize` model. It takes an audio file as input, processes it in chunks, and generates a diarized transcript, identifying different speakers in the audio. Obtain an OpenAI API key from the [OpenAI platform](https://platform.openai.com/).

# H2O Flow in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/H2O_Flow_Google_Colab.ipynb)

This notebook provides a step-by-step guide to setting up and accessing H2O Flow within a Google Colab environment. It installs necessary dependencies, initializes the H2O server, and uses localtunnel to create a public URL for the H2O Flow web interface, allowing users to interact with H2O's machine learning capabilities directly from Colab.

# VibeVoice-ASR: Unified Speech-to-Text with Speaker Diarization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/VibeVoice_ASR_Colab.ipynb)

This notebook demonstrates [Microsoft's VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR), a 9B parameter model providing unified speech-to-text with speaker diarization and timestamps. Features include 60-minute single-pass processing, consistent speaker tracking, and customizable hotwords for domain-specific accuracy. Requires A100 GPU (Colab Pro) for best results; T4 works with 4-bit quantization.

---

# Qwen3-TTS Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/Qwen3_TTS.ipynb)

This notebook runs the [Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS) demo. It supports Voice Design (Instruct to Speech), Voice Cloning, and Custom Voices using the Qwen2.5-based TTS models. It is adapted to run on a free Colab GPU (T4).

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