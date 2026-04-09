"""
Download faster-whisper model weights from HuggingFace with progress logging.

Usage:
    python download_model.py
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")

MODEL = "Systran/faster-whisper-large-v3"
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    log.info("HF token found — authenticated download.")
else:
    log.warning("No HF_TOKEN in .env — unauthenticated (rate-limited).")

from huggingface_hub import snapshot_download, list_repo_files
import huggingface_hub

log.info("Model: %s", MODEL)

# List files to know total count
log.info("Fetching file list...")
files = list(list_repo_files(MODEL, token=HF_TOKEN))
log.info("Files to download: %d", len(files))
for f in files:
    log.info("  %s", f)

log.info("Starting download...")

from huggingface_hub import hf_hub_download

total = len(files)
for i, filename in enumerate(files, 1):
    log.info("[%d/%d] %s ...", i, total, filename)
    path = hf_hub_download(
        repo_id=MODEL,
        filename=filename,
        token=HF_TOKEN,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    log.info("[%d/%d] done — %.1f MB  →  %s", i, total, size_mb, path)

log.info("All files downloaded. Model is ready.")
