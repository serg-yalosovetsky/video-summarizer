# Video / Audio Summarizer

A local web app that transcribes audio/video files and produces a cleaned transcript and structured summary — entirely on your own hardware, no cloud APIs.

**Pipeline (sequential):**
1. **FFmpeg** — converts any audio/video to 16 kHz mono WAV
2. **pyannote** — speaker diarization
3. **NVIDIA Canary 1B v2** (NeMo) — speech-to-text transcription
4. **Gemma** (Ollama) — cleans the raw transcript
5. **Gemma** (Ollama) — detects whether the recording is a meeting
6. **Gemma** (Ollama) — generates a structured summary
7. **Gemma** (Ollama) — generates a personal ToDo (meetings) or TL;DR (other)

Results stream to the browser in real time via SSE.

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python 3.10–3.12 | Runtime (3.13+ not supported by editdistance/numba) |
| FFmpeg | Audio/video conversion |
| Ollama | LLM inference (Gemma) |
| CUDA | GPU acceleration for Canary (default runtime mode) |

---

## Installation

### 1. Clone and create a virtual environment

Use [uv](https://github.com/astral-sh/uv) with the pinned Python version from `.python-version` (currently 3.12.11):

```bash
git clone <repo-url>
cd video-summarizer
uv python install 3.12.11
uv venv --python 3.12.11 .venv
```

### 2. Install Python dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

If you need Canary on CUDA, do not stop at the generic command above. That installs the default `torch` wheel, which is often CPU-only on Windows.
Use the project bootstrap instead, which verifies `torch.cuda.is_available()` and repairs PyTorch from the CUDA wheel index when `CANARY_DEVICE=cuda`:

```bash
./install.sh
```

On Windows:

```bat
install.bat
```

NeMo is a large package. For manual CUDA-specific installs see the [NeMo installation guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html).

### 3. Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH`.

### 4. Install and start Ollama

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the models
ollama pull gemma4:e4b
ollama pull gemma4:e2b

# Ollama starts automatically; or run manually:
ollama serve
```

### 5. Configure environment

Copy the template and edit what you need:

```bash
cp .env.example .env
```

The main settings most users change are:

```env
USER_PRIMARY_NAME=Сергей
USER_ALIASES=Сергей,Сергій,Serhii
HF_TOKEN=hf_your_token_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEVICE=gpu
OLLAMA_MODEL=gemma4:e4b
OLLAMA_CLEAN_MODEL=gemma4:e2b
FRAME_MODEL=gemma4:e4b
CANARY_DEVICE=cuda
MAX_UPLOAD_MB=0
```

`USER_PRIMARY_NAME` and `USER_ALIASES` are used in personalized summary sections like "What concerns ..." and personal meeting todos.
If these variables are set in `.env`, they take priority over `.omx/project-memory.json`.

`OLLAMA_MODEL` is used for summary and todo/tldr generation. `OLLAMA_CLEAN_MODEL` is used for the transcript cleaning step (defaults to a lighter model). `FRAME_MODEL` is used for visual frame analysis and defaults to `OLLAMA_MODEL` if not set.

The app defaults to `CANARY_DEVICE=cuda` and will fail fast if CUDA is unavailable instead of silently using CPU.
If you intentionally want CPU fallback for debugging, set:

```env
CANARY_DEVICE=auto
```

The app also defaults to `OLLAMA_DEVICE=gpu` and refuses to continue if Ollama loads models without GPU usage.
If you intentionally want to skip that verification for debugging, set:

```env
OLLAMA_DEVICE=auto
```

`MAX_UPLOAD_MB` is optional. Set it to `0` (or leave it empty) to disable the app-level upload cap, or set a positive number of megabytes if you want the server to reject larger files early.

### Advanced settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_TIMEOUT_SECONDS` | `900` | Timeout for summary/todo Ollama calls |
| `OLLAMA_CLEAN_TIMEOUT_SECONDS` | `1800` | Timeout for transcript cleaning calls |
| `OLLAMA_SUMMARY_MAX_TOKENS` | `4096` | Max tokens for summary/todo output |
| `OLLAMA_CLEAN_MAX_TOKENS` | `4096` | Max tokens for transcript cleaning output |
| `OLLAMA_NUM_CTX` | `12288` | Context window size sent to Ollama |
| `STAGE_DELAY_SECONDS` | `10` | Pause between Ollama stages (lets VRAM settle) |
| `CANARY_SEGMENT_BATCH_SIZE` | `8` | Batch size for Canary transcription segments |
| `PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | Speaker diarization model |
| `PYANNOTE_DEVICE` | `auto` | Device for pyannote (`auto`, `cuda`, `cpu`) |
| `FRAME_TIMESTAMPS` | `1,2,5,10` | Seconds at which to sample frames for visual analysis |
| `MAX_FRAMES` | `20` | Maximum number of frames to analyze |
| `MAX_VISUAL_CONTEXT_CHARS` | `2000` | Max chars of visual context passed to the summary step |
| `NTFY_TOPIC` | *(internal)* | ntfy.sh topic for push notifications on completion |
| `NTFY_URL` | derived from topic | Full ntfy URL (overrides `NTFY_TOPIC`) |
| `LANGFUSE_PUBLIC_KEY` | — | Enable Langfuse tracing (optional) |
| `LANGFUSE_SECRET_KEY` | — | Enable Langfuse tracing (optional) |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse endpoint |

---

## Running the web app

```bash
uvicorn main:app --host 0.0.0.0 --port 8888
```

Or use the bootstrap scripts, which install `uv`, install the pinned Python version, create `.venv`, install dependencies, and then start the app:

```bash
./run.sh
```

On Windows:

```bat
run.bat
```

Open [http://localhost:8888](http://localhost:8888), upload a file, and click **Суммаризировать**.

On first run the Canary model is downloaded from HuggingFace (~2 GB) and cached locally.

---

## Standalone transcription

To transcribe a single file from the command line:

```bash
python transcribe.py path/to/file.mp4
python transcribe.py path/to/file.wav --source-lang uk --target-lang en
```

The transcript is printed to stdout and saved as `<file>.transcript.txt`.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--source-lang` | `en` | Source language code |
| `--target-lang` | same as source | Target language (for translation) |
| `--model` | `nvidia/canary-1b-v2` | HF model name or local `.nemo` path |

---

## Supported languages (Canary 1B v2)

`en` `de` `es` `fr` and more — see the [model card](https://huggingface.co/nvidia/canary-1b-v2).

---

## Using a local model file

Download the model once and point to it to avoid re-downloading:

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/canary-1b-v2", local_dir="./models/canary-1b-v2")
EOF
```

Then set in `.env` or pass via `--model`:

```env
CANARY_MODEL=./models/canary-1b-v2/canary-1b-v2.nemo
```

---

## Project structure

```
video-summarizer/
├── main.py                      # FastAPI app entry point
├── processing_pipeline.py       # Full sequential processing pipeline
├── summary.py                   # Ollama summary/todo/tldr generation
├── prompts.py                   # Prompt templates
├── config.py                    # Settings loaded from .env
├── transcribe.py                # Standalone CLI transcription script
├── transcribe_diarization.py    # Speaker diarization integration
├── transcribe_ffmpeg.py         # FFmpeg audio conversion
├── frames_analyze.py            # Frame extraction and visual analysis
├── helpers.py                   # Shared utilities
├── tracing.py                   # Langfuse tracing helpers
├── requirements.txt
├── static/
│   └── index.html               # Single-page frontend
├── .env.example                 # Config template for new users
└── .env                         # Local overrides (not committed)
```

---

## Docker

The container includes the Python app and `ffmpeg`. Ollama is not bundled — run it separately and pass the address via `OLLAMA_BASE_URL`.

Build the image:

```bash
docker build -t video-summarizer .
```

Run on Windows/macOS Docker Desktop with Ollama on the host:

```bash
docker run --rm -p 8888:8888 ^
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ^
  -e OLLAMA_MODEL=gemma4:e4b ^
  -e HF_TOKEN=your_hf_token ^
  video-summarizer
```

Run on Linux with Ollama on the host:

```bash
docker run --rm -p 8888:8888 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=gemma4:e4b \
  -e HF_TOKEN=your_hf_token \
  video-summarizer
```

If Ollama runs in a separate container on the same Docker network, use its service name:

```text
OLLAMA_BASE_URL=http://ollama:11434
```
