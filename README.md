# Video Summarizer

FastAPI-приложение для загрузки видео или аудио, автоматической расшифровки через `faster-whisper`, очистки транскрипта и генерации саммари через Ollama.

## Что делает

- принимает аудио и видеофайлы через веб-интерфейс;
- конвертирует медиа в `16 kHz mono WAV` через `ffmpeg`;
- строит сырую транскрипцию через Whisper;
- очищает текст отдельным LLM-промптом;
- генерирует структурированное саммари;
- отдает прогресс по этапам через SSE.

## Стек

- Python 3.11+
- FastAPI + Uvicorn
- `faster-whisper`
- `ffmpeg` / `ffprobe`
- Ollama

## Переменные окружения

- `HF_TOKEN` — опциональный токен Hugging Face для скачивания моделей быстрее и без жестких лимитов.
- `OLLAMA_URL` — URL Ollama API. По умолчанию: `http://localhost:11434/api/generate`
- `OLLAMA_MODEL` — модель для очистки и суммаризации. По умолчанию: `gemma4:e4b`

Пример `.env`:

```env
HF_TOKEN=your_hf_token
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=gemma4:e4b
```

## Локальный запуск

Установить системные зависимости:

- Python 3.11+
- FFmpeg с доступными в `PATH` командами `ffmpeg` и `ffprobe`
- запущенный Ollama

Установить Python-зависимости:

```bash
pip install -r requirements.txt
```

Запустить сервер:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Открыть:

```text
http://localhost:8000
```

## Docker

Контейнер включает Python-приложение и `ffmpeg`. Ollama в образ не входит: его нужно запускать отдельно и передавать адрес через `OLLAMA_URL`.

Сборка образа:

```bash
docker build -t video-summarizer .
```

Запуск на Windows/macOS Docker Desktop, если Ollama работает на хосте:

```bash
docker run --rm -p 8000:8000 ^
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate ^
  -e OLLAMA_MODEL=gemma4:e4b ^
  -e HF_TOKEN=your_hf_token ^
  video-summarizer
```

Запуск на Linux, если Ollama работает на хосте:

```bash
docker run --rm -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  -e OLLAMA_MODEL=gemma4:e4b \
  -e HF_TOKEN=your_hf_token \
  video-summarizer
```

Если Ollama запущен в отдельном контейнере в одной Docker-сети, укажите его сервисное имя:

```text
http://ollama:11434/api/generate
```

## Поведение при старте

- При первом запуске приложение загружает модель Whisper.
- Если CUDA недоступна, приложение автоматически переключается на CPU.
- Для CPU-режима используется более легкая модель `base`, для CUDA — `large-v3`.

## Полезные команды

Предзагрузка Whisper-модели:

```bash
python download_model.py
```

Локальная транскрипция файла без веб-интерфейса:

```bash
python transcribe.py path/to/file.mp4
```

## HTTP API

- `GET /` — веб-интерфейс
- `POST /process` — загрузка файла и поток событий SSE со статусом, транскриптом и саммари

## Ограничения

- размер загружаемого файла ограничен `500 MB`;
- для суммаризации нужен доступный Ollama API;
- качество и скорость зависят от доступности GPU и размера модели.
