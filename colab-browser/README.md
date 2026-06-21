# Colab Notebooks (browser)

Standalone **Google Colab notebooks** intended to be run from the **Colab web interface**. Each notebook is self-contained — open it in Colab with the badge below and run it top to bottom; no local setup or repository clone is required.

For the VS Code + Colab extension workflow, see [`../colab-vscode/`](../colab-vscode/) instead.

---

## Notebooks

### Transcribe Audio with OpenAI Whisper
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/colab-browser/Transcribe_Audio_Whisper.ipynb)

Transcribe audio files using OpenAI's Whisper model via the `whisper` Python package. Supports common audio formats (e.g., MP3, WAV) and provides a simple pipeline for uploading, transcribing, and saving transcripts.

### Audio Diarization with `gpt-4o-transcribe-diarize`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/colab-browser/speech_to_text_diarization.ipynb)

Perform audio diarization using OpenAI's `gpt-4o-transcribe-diarize` model. Takes an audio file, processes it in chunks, and generates a diarized transcript that identifies different speakers. Requires an OpenAI API key from the [OpenAI platform](https://platform.openai.com/).

### VibeVoice-ASR: Unified Speech-to-Text with Speaker Diarization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/colab-browser/VibeVoice_ASR_Colab.ipynb)

Demonstrates [Microsoft's VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR), a 9B parameter model providing unified speech-to-text with speaker diarization and timestamps. Features 60-minute single-pass processing, consistent speaker tracking, and customizable hotwords. Requires an A100 GPU (Colab Pro) for best results; T4 works with 4-bit quantization.

### Qwen3-ASR: High-Performance Automatic Speech Recognition
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/colab-browser/Qwen3_ASR_Colab.ipynb)

Implements the [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) model for fast, accurate Automatic Speech Recognition. Features automatic language detection, high-speed inference on T4 GPUs, and transcription of audio files directly from Google Drive.

### Qwen3-TTS Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/colab-browser/Qwen3_TTS.ipynb)

Runs the [Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS) demo. Supports Voice Design (Instruct to Speech), Voice Cloning, and Custom Voices using the Qwen2.5-based TTS models. Adapted to run on a free Colab GPU (T4).

### H2O Flow in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/notebooks/blob/main/colab-browser/H2O_Flow_Google_Colab.ipynb)

A step-by-step guide to setting up and accessing H2O Flow within Google Colab. Installs dependencies, initializes the H2O server, and uses localtunnel to create a public URL for the H2O Flow web interface.

---

## Notes on Colab runtimes

- Colab runtimes are **ephemeral** — local files may be lost on restart or disconnect
- Commit changes to Git or export outputs if persistence is required

---

## License

MIT — see the [`LICENSE`](../LICENSE) file for full details.
