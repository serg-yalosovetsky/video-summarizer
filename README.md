# Video / Audio Summarizer

A local web app that transcribes audio/video files and produces a cleaned transcript and structured summary — entirely on your own hardware, no cloud APIs.

**Pipeline:**
1. **FFmpeg** — converts any audio/video to 16 kHz mono WAV
2. **NVIDIA Canary 1B v2** (NeMo) — speech-to-text transcription
3. **Gemma 4 27B** (Ollama) — cleans the raw transcript
4. **Gemma 4 27B** (Ollama) — generates a structured summary

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

# Pull the model
ollama pull gemma4:27b

# Ollama starts automatically; or run manually:
ollama serve
```

### 5. Configure environment (optional)

Create a `.env` file in the project root if `nvidia/canary-1b-v2` is gated on HuggingFace:

```env
HF_TOKEN=hf_your_token_here
```

The app defaults to `CANARY_DEVICE=cuda` and will fail fast if CUDA is unavailable instead of silently using CPU.
If you intentionally want CPU fallback for debugging, set:

```env
CANARY_DEVICE=auto
```

---

## Running the web app

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the bootstrap scripts, which install `uv`, install the pinned Python version, create `.venv`, install dependencies, and then start the app:

```bash
./run.sh
```

On Windows:

```bat
run.bat
```

Open [http://localhost:8000](http://localhost:8000), upload a file, and click **Суммаризировать**.

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
# Download via huggingface_hub
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
├── main.py           # FastAPI app + pipeline
├── transcribe.py     # Standalone CLI transcription script
├── requirements.txt
├── static/
│   └── index.html    # Single-page frontend
└── .env              # HF token (not committed)
```

---

## Docker

The container includes the Python app and `ffmpeg`. Ollama is not bundled — run it separately and pass the address via `OLLAMA_URL`.

Build the image:

```bash
docker build -t video-summarizer .
```

Run on Windows/macOS Docker Desktop with Ollama on the host:

```bash
docker run --rm -p 8000:8000 ^
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate ^
  -e OLLAMA_MODEL=gemma4:e4b ^
  -e HF_TOKEN=your_hf_token ^
  video-summarizer
```

Run on Linux with Ollama on the host:

```bash
docker run --rm -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  -e OLLAMA_MODEL=gemma4:e4b \
  -e HF_TOKEN=your_hf_token \
  video-summarizer
```

If Ollama runs in a separate container on the same Docker network, use its service name:

```text
http://ollama:11434/api/generate
```
