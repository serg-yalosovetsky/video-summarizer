import asyncio
import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_test_stubs() -> None:
    return None


install_test_stubs()
main = importlib.import_module("main")
transcribe = importlib.import_module("transcribe")
transcribe_diarization = importlib.import_module("transcribe_diarization")
helpers = importlib.import_module("helpers")
summary = importlib.import_module("summary")
frames_analyze = importlib.import_module("frames_analyze")
models = importlib.import_module("models")


def decode_events(messages: list[str]) -> list[dict]:
    result = []
    for message in messages:
        payload = message.removeprefix("data: ").strip()
        result.append(json.loads(payload))
    return result


class LargeChunkFile:
    def __init__(self, total_bytes: int, *, fill_byte: bytes = b"x"):
        self.remaining = total_bytes
        self.fill_byte = fill_byte
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self.remaining
        chunk_size = min(size, self.remaining)
        self.remaining -= chunk_size
        return self.fill_byte * chunk_size

    def close(self) -> None:
        self.closed = True


class ProcessGeneratorTests(unittest.IsolatedAsyncioTestCase):
    def test_combine_sources_merges_non_empty_parts(self):
        combined = main.combine_sources("[00:00:01] hi", "hello chat")
        self.assertIn("=== Video/Audio Transcript ===", combined)
        self.assertIn("=== Chat Messages ===", combined)

    def test_is_context_sufficient_requires_real_signal(self):
        self.assertFalse(main.is_context_sufficient(""))
        self.assertFalse(main.is_context_sufficient("no people\nno names"))
        self.assertTrue(
            main.is_context_sufficient(
                "\n".join(
                    [
                        "meeting room with slide deck on screen",
                        "Alice Johnson speaking near the projector",
                        "lower third shows Bob Smith",
                    ]
                )
            )
        )

    async def test_process_generator_emits_expected_stage_sequence(self):
        upload = main.UploadFile(
            filename="demo.mp4",
            file=io.BytesIO(b"video-bytes"),
        )

        def fake_convert_to_wav(input_path: str, output_path: str) -> dict:
            Path(output_path).write_bytes(b"wav")
            return {
                "file_info": "demo",
                "format": "mp4",
                "duration": "00:00:12",
                "duration_sec": 12.0,
                "codec": "aac",
                "has_video": True,
            }

        def fake_extract_frames(input_path: str, tmp_dir: str, duration_sec: float) -> list[str]:
            frame_path = Path(tmp_dir) / "frame_1s.jpg"
            frame_path.write_bytes(b"frame")
            return [str(frame_path)]

        def fake_analyze_frames_with_progress(
            image_paths,
            async_q,
            loop,
            start_index=0,
            total_hint=None,
        ):
            loop.call_soon_threadsafe(
                async_q.put_nowait,
                {"current": start_index + 1, "total": total_hint or len(image_paths)},
            )
            loop.call_soon_threadsafe(async_q.put_nowait, None)
            return "[1s] Alice speaking in a meeting room"

        def fake_transcribe_with_canary(
            wav_path,
            async_q,
            loop,
            source_lang="ru",
        ):
            loop.call_soon_threadsafe(async_q.put_nowait, 50)
            loop.call_soon_threadsafe(async_q.put_nowait, None)
            return "hello world"

        patched_settings = replace(main.settings, stage_delay_seconds=0)
        with (
            mock.patch.object(main, "settings", patched_settings),
            mock.patch.object(main, "notify_done", new=mock.AsyncMock()),
            mock.patch.object(main, "convert_to_wav", side_effect=fake_convert_to_wav),
            mock.patch.object(main, "run_diarization", return_value=[]),
            mock.patch.object(main, "extract_frames", side_effect=fake_extract_frames),
            mock.patch.object(
                main,
                "analyze_frames_with_progress",
                side_effect=fake_analyze_frames_with_progress,
            ),
            mock.patch.object(
                main,
                "transcribe_with_canary",
                side_effect=fake_transcribe_with_canary,
            ),
            mock.patch.object(main, "clean_content", return_value="cleaned transcript"),
            mock.patch.object(main, "classify_is_meeting", return_value=False),
            mock.patch.object(main, "generate_summary", return_value="full summary"),
            mock.patch.object(main, "classify_text_language", return_value="other"),
            mock.patch.object(main, "translate_summary_to_russian", return_value="russian translation"),
            mock.patch.object(main, "generate_short_summary", return_value="short summary"),
            mock.patch.object(main, "generate_personal_todo", return_value="todo summary"),
        ):
            chunks = []
            async for item in main.process_generator(upload, "chat line", "ru"):
                chunks.append(item)

        events = decode_events(chunks)
        event_names = [event["event"] for event in events]

        self.assertIn("ffmpeg_done", event_names)
        self.assertIn("frames_done", event_names)
        self.assertIn("transcript_done", event_names)
        self.assertIn("cleaned_done", event_names)
        self.assertIn("summary_done", event_names)
        self.assertIn("tldr_done", event_names)
        self.assertLess(event_names.index("ffmpeg_done"), event_names.index("transcript_done"))
        self.assertLess(event_names.index("cleaned_done"), event_names.index("summary_done"))
        cleaned_event = next(event for event in events if event["event"] == "cleaned_done")
        self.assertEqual(cleaned_event["payload"]["text"], "cleaned transcript")
        self.assertTrue(cleaned_event["payload"]["download_url"].startswith("/artifacts/"))
        self.assertTrue(cleaned_event["payload"]["filename"].endswith(".cleaned.txt"))
        tldr_event = next(event for event in events if event["event"] == "tldr_done")
        self.assertEqual(tldr_event["payload"]["title"], "Краткое саммари")
        self.assertFalse(tldr_event["payload"]["is_meeting"])

    async def test_process_generator_streams_large_upload_when_limit_disabled(self):
        upload = main.UploadFile(
            filename="large-demo.mp4",
            file=LargeChunkFile(12 * 1024 * 1024),
        )

        def fake_convert_to_wav(input_path: str, output_path: str) -> dict:
            Path(output_path).write_bytes(b"wav")
            return {
                "file_info": "demo",
                "format": "mp4",
                "duration": "00:00:12",
                "duration_sec": 12.0,
                "codec": "aac",
                "has_video": True,
            }

        def fake_extract_frames(input_path: str, tmp_dir: str, duration_sec: float) -> list[str]:
            frame_path = Path(tmp_dir) / "frame_1s.jpg"
            frame_path.write_bytes(b"frame")
            return [str(frame_path)]

        def fake_analyze_frames_with_progress(
            image_paths,
            async_q,
            loop,
            start_index=0,
            total_hint=None,
        ):
            loop.call_soon_threadsafe(
                async_q.put_nowait,
                {"current": start_index + 1, "total": total_hint or len(image_paths)},
            )
            loop.call_soon_threadsafe(async_q.put_nowait, None)
            return "[1s] Alice speaking in a meeting room"

        def fake_transcribe_with_canary(
            wav_path,
            async_q,
            loop,
            source_lang="ru",
        ):
            loop.call_soon_threadsafe(async_q.put_nowait, 100)
            loop.call_soon_threadsafe(async_q.put_nowait, None)
            return "hello world"

        patched_settings = replace(main.settings, max_upload_bytes=None, stage_delay_seconds=0)
        with (
            mock.patch.object(main, "settings", patched_settings),
            mock.patch.object(main, "notify_done", new=mock.AsyncMock()),
            mock.patch.object(main, "convert_to_wav", side_effect=fake_convert_to_wav),
            mock.patch.object(main, "run_diarization", return_value=[]),
            mock.patch.object(main, "extract_frames", side_effect=fake_extract_frames),
            mock.patch.object(
                main,
                "analyze_frames_with_progress",
                side_effect=fake_analyze_frames_with_progress,
            ),
            mock.patch.object(
                main,
                "transcribe_with_canary",
                side_effect=fake_transcribe_with_canary,
            ),
            mock.patch.object(main, "clean_content", return_value="cleaned transcript"),
            mock.patch.object(main, "classify_is_meeting", return_value=False),
            mock.patch.object(main, "generate_summary", return_value="full summary"),
            mock.patch.object(main, "classify_text_language", return_value="other"),
            mock.patch.object(main, "translate_summary_to_russian", return_value="russian translation"),
            mock.patch.object(main, "generate_short_summary", return_value="short summary"),
            mock.patch.object(main, "generate_personal_todo", return_value="todo summary"),
        ):
            chunks = []
            async for item in main.process_generator(upload, "chat line", "ru"):
                chunks.append(item)

        events = decode_events(chunks)
        event_names = [event["event"] for event in events]
        self.assertIn("ffmpeg_done", event_names)
        self.assertNotIn("error", event_names)

    async def test_process_generator_rejects_large_upload_when_limit_is_configured(self):
        upload = main.UploadFile(
            filename="too-large.mp4",
            file=LargeChunkFile(6 * 1024 * 1024),
        )

        patched_settings = replace(main.settings, max_upload_bytes=5 * 1024 * 1024)
        with (
            mock.patch.object(main, "settings", patched_settings),
            mock.patch.object(main, "notify_done", new=mock.AsyncMock()),
        ):
            chunks = []
            async for item in main.process_generator(upload, "", "ru"):
                chunks.append(item)

        events = decode_events(chunks)
        self.assertEqual([event["event"] for event in events], ["error"])
        self.assertEqual(events[0]["payload"]["stage"], "upload")
        self.assertEqual(
            events[0]["payload"]["message"],
            "File exceeds configured upload limit of 5 MB.",
        )

    async def test_transcribe_with_canary_signals_queue_on_error(self):
        class BrokenModel:
            def transcribe(self, **kwargs):
                raise RuntimeError("boom")

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        with mock.patch.object(transcribe, "get_canary_model", return_value=BrokenModel()):
            with self.assertRaises(RuntimeError):
                transcribe.transcribe_with_canary("audio.wav", queue, loop, "ru")

        progress = await queue.get()
        done = await queue.get()
        self.assertEqual(progress, 1)
        self.assertIsNone(done)

    async def test_transcribe_by_segments_batches_canary_calls(self):
        class FakeOutput:
            def __init__(self, text):
                self.text = text

        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, **kwargs):
                self.calls.append(kwargs["audio"])
                return [FakeOutput(f"text-{i}") for i, _ in enumerate(kwargs["audio"], 1)]

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        fake_model = FakeModel()
        segments = [
            (0.0, 1.0, "SPEAKER_00"),
            (2.0, 3.0, "SPEAKER_01"),
            (4.0, 5.0, "SPEAKER_02"),
        ]

        def fake_extract_audio_chunk(wav_path: str, start: float, end: float, out_path: str) -> bool:
            Path(out_path).write_bytes(b"wav")
            return True

        with (
            mock.patch.object(transcribe, "get_canary_model", return_value=fake_model),
            mock.patch.object(transcribe, "extract_audio_chunk", side_effect=fake_extract_audio_chunk),
            mock.patch.object(transcribe, "CANARY_SEGMENT_BATCH_SIZE", 2),
        ):
            text = transcribe.transcribe_by_segments("audio.wav", segments, queue, loop, "ru")

        self.assertEqual(len(fake_model.calls), 2)
        self.assertEqual(len(fake_model.calls[0]), 2)
        self.assertEqual(len(fake_model.calls[1]), 1)
        self.assertIn("[00:00:00] [SPEAKER_00]: text-1", text)
        self.assertIn("[00:00:02] [SPEAKER_01]: text-2", text)
        self.assertIn("[00:00:04] [SPEAKER_02]: text-1", text)
        self.assertEqual(await queue.get(), 33)
        self.assertEqual(await queue.get(), 66)
        self.assertEqual(await queue.get(), 100)
        self.assertIsNone(await queue.get())

    async def test_process_generator_emits_personal_todo_for_meetings(self):
        with (
            mock.patch.object(main, "settings", replace(main.settings, stage_delay_seconds=0)),
            mock.patch.object(main, "notify_done", new=mock.AsyncMock()),
            mock.patch.object(main, "clean_content", return_value="meeting transcript"),
            mock.patch.object(main, "classify_is_meeting", return_value=True),
            mock.patch.object(main, "generate_summary", return_value="meeting summary"),
            mock.patch.object(main, "classify_text_language", return_value="ru"),
            mock.patch.object(main, "generate_short_summary", return_value="short summary"),
            mock.patch.object(main, "generate_personal_todo", return_value="todo for sergey"),
        ):
            chunks = []
            async for item in main.process_generator(None, "meeting chat", "ru"):
                chunks.append(item)

        events = decode_events(chunks)
        tldr_event = next(event for event in events if event["event"] == "tldr_done")
        self.assertEqual(tldr_event["payload"]["title"], "ToDo для меня")
        self.assertTrue(tldr_event["payload"]["is_meeting"])
        self.assertEqual(tldr_event["payload"]["text"], "todo for sergey")

    async def test_process_generator_retries_truncated_tldr(self):
        with (
            mock.patch.object(main, "settings", replace(main.settings, stage_delay_seconds=0)),
            mock.patch.object(main, "notify_done", new=mock.AsyncMock()),
            mock.patch.object(main, "clean_content", return_value="discussion transcript"),
            mock.patch.object(main, "classify_is_meeting", return_value=False),
            mock.patch.object(main, "generate_summary", return_value="Full summary."),
            mock.patch.object(main, "classify_text_language", return_value="ru"),
            mock.patch.object(
                main,
                "generate_short_summary",
                side_effect=[
                    "**Problem**: The discussion addresses two main technical issues and",
                    "**Problem**: The discussion addresses two main technical issues and proposes next steps.",
                ],
            ),
        ):
            chunks = []
            async for item in main.process_generator(None, "meeting chat", "ru"):
                chunks.append(item)

        events = decode_events(chunks)
        tldr_event = next(event for event in events if event["event"] == "tldr_done")
        self.assertEqual(
            tldr_event["payload"]["text"],
            "**Problem**: The discussion addresses two main technical issues and proposes next steps.",
        )

    async def test_process_generator_retries_invalid_structured_tldr(self):
        with (
            mock.patch.object(main, "settings", replace(main.settings, stage_delay_seconds=0)),
            mock.patch.object(main, "notify_done", new=mock.AsyncMock()),
            mock.patch.object(main, "clean_content", return_value="discussion transcript"),
            mock.patch.object(main, "classify_is_meeting", return_value=False),
            mock.patch.object(main, "generate_summary", return_value="Full summary."),
            mock.patch.object(main, "classify_text_language", return_value="ru"),
            mock.patch.object(
                main,
                "generate_short_summary",
                side_effect=[
                    ValueError("Model returned invalid structured content"),
                    "Short recap.\n- Investigate the deployment issue.\n- Share the fix plan.",
                ],
            ),
        ):
            chunks = []
            async for item in main.process_generator(None, "meeting chat", "ru"):
                chunks.append(item)

        events = decode_events(chunks)
        tldr_event = next(event for event in events if event["event"] == "tldr_done")
        self.assertEqual(
            tldr_event["payload"]["text"],
            "Short recap.\n- Investigate the deployment issue.\n- Share the fix plan.",
        )

    async def test_lifespan_checks_ollama_before_loading_canary(self):
        calls = []

        def fake_ensure_ollama_ready(*models, timeout=10.0):
            calls.append(("ollama", models, timeout))

        def fake_get_canary_model():
            calls.append(("canary",))
            return object()

        with (
            mock.patch.object(main, "ensure_ollama_ready", side_effect=fake_ensure_ollama_ready),
            mock.patch.object(main, "get_canary_model", side_effect=fake_get_canary_model),
        ):
            async with main.lifespan(main.app):
                pass

        self.assertEqual(calls[0][0], "ollama")
        self.assertEqual(calls[1][0], "canary")
        self.assertIn(main.OLLAMA_MODEL, calls[0][1])
        self.assertIn(main.FRAME_MODEL, calls[0][1])


class OllamaReadyTests(unittest.TestCase):
    @staticmethod
    def _response(payload: dict):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_ensure_ollama_ready_requires_gpu_when_configured(self):
        tags_response = self._response({"models": [{"name": "gemma4:e4b"}]})
        ps_response = self._response(
            {
                "models": [
                    {
                        "name": "gemma4:e4b",
                        "processor": "100% GPU",
                        "size_vram": 123456,
                    }
                ]
            }
        )
        warm_response = self._response({"response": "ok"})

        with (
            mock.patch.object(helpers, "settings", replace(helpers.settings, ollama_device="gpu")),
            mock.patch.object(helpers.httpx, "get", side_effect=[tags_response, ps_response]),
            mock.patch.object(helpers.httpx, "post", return_value=warm_response) as post_mock,
        ):
            helpers.ensure_ollama_ready("gemma4:e4b")

        post_mock.assert_called_once()
        self.assertEqual(
            post_mock.call_args.kwargs["json"],
            models.OllamaWarmModelRequest(
                model="gemma4:e4b",
                prompt="Reply with exactly one token: ok",
                stream=False,
                keep_alive=0,
                options={"num_predict": 1, "temperature": 0},
            ).model_dump(),
        )

    def test_ensure_ollama_ready_rejects_cpu_only_ollama(self):
        tags_response = self._response({"models": [{"name": "gemma4:e4b"}]})
        ps_response = self._response(
            {
                "models": [
                    {
                        "name": "gemma4:e4b",
                        "processor": "100% CPU",
                        "size_vram": 0,
                    }
                ]
            }
        )
        warm_response = self._response({"response": "ok"})

        with (
            mock.patch.object(helpers, "settings", replace(helpers.settings, ollama_device="gpu")),
            mock.patch.object(helpers.httpx, "get", side_effect=[tags_response, ps_response]),
            mock.patch.object(helpers.httpx, "post", return_value=warm_response),
        ):
            with self.assertRaisesRegex(RuntimeError, "not using the GPU"):
                helpers.ensure_ollama_ready("gemma4:e4b")

    def test_ensure_ollama_ready_skips_gpu_verification_in_auto_mode(self):
        tags_response = self._response({"models": [{"name": "gemma4:e4b"}]})

        with (
            mock.patch.object(helpers, "settings", replace(helpers.settings, ollama_device="auto")),
            mock.patch.object(helpers.httpx, "get", return_value=tags_response) as get_mock,
            mock.patch.object(helpers.httpx, "post") as post_mock,
        ):
            helpers.ensure_ollama_ready("gemma4:e4b")

        self.assertEqual(get_mock.call_count, 1)
        post_mock.assert_not_called()


class OllamaRequestPayloadTests(unittest.TestCase):
    def test_call_ollama_serializes_text_request_model(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": "done"}
        request_format = {"type": "object", "properties": {"summary": {"type": "string"}}}
        options = {"temperature": 0, "num_predict": 128}

        with (
            mock.patch.object(helpers, "save_text_request"),
            mock.patch.object(
                helpers,
                "start_observation",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
            ),
            mock.patch.object(helpers.httpx, "post", return_value=response) as post_mock,
        ):
            result = helpers.call_ollama(
                "Summarize this",
                "Be concise",
                timeout=12.0,
                model="gemma4:e4b",
                options=options,
                format=request_format,
            )

        self.assertEqual(result, "done")
        self.assertEqual(
            post_mock.call_args.kwargs["json"],
            models.OllamaTextGenerateRequest(
                model="gemma4:e4b",
                prompt="Summarize this",
                system="Be concise",
                stream=False,
                options=options,
                format=request_format,
            ).model_dump(exclude_none=True),
        )

    def test_unload_ollama_models_serializes_unload_request_model(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        ps_response = mock.Mock()
        ps_response.raise_for_status.return_value = None
        ps_response.json.return_value = {"models": []}

        with (
            mock.patch.object(helpers.httpx, "post", return_value=response) as post_mock,
            mock.patch.object(helpers.httpx, "get", return_value=ps_response),
        ):
            helpers.unload_ollama_models("gemma4:e4b")

        self.assertEqual(
            post_mock.call_args.kwargs["json"],
            models.OllamaUnloadModelRequest(model="gemma4:e4b", keep_alive=0).model_dump(),
        )

    def test_frames_vision_payload_uses_request_model(self):
        payload = frames_analyze._ollama_request_payload(
            "Describe this frame",
            "Return JSON",
            "image-as-b64",
            {"type": "object"},
        )

        self.assertIsInstance(payload, models.OllamaVisionGenerateRequest)
        self.assertEqual(
            payload.model_dump(),
            models.OllamaVisionGenerateRequest(
                model=frames_analyze.FRAME_MODEL,
                prompt="Describe this frame",
                system="Return JSON",
                images=["image-as-b64"],
                stream=False,
                format={"type": "object"},
                keep_alive=frames_analyze._OLLAMA_KEEP_ALIVE,
            ).model_dump(),
        )


class RunDiarizationTests(unittest.TestCase):
    def test_run_diarization_handles_falsey_annotation_from_diarize_output(self):
        class FalseyAnnotation:
            def __bool__(self):
                return False

            def itertracks(self, yield_label=False):
                yield (SimpleNamespace(start=1.5, end=3.0), None, "SPEAKER_00")

        class FakeOutput:
            speaker_diarization = FalseyAnnotation()

        fake_pipeline = mock.Mock(return_value=FakeOutput())
        with (
            mock.patch.object(transcribe_diarization, "get_diarizer", return_value=fake_pipeline),
            mock.patch.object(
                transcribe_diarization,
                "start_observation",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
            ),
        ):
            segments = transcribe_diarization.run_diarization("audio.wav")

        self.assertEqual(segments, [(1.5, 3.0, "SPEAKER_00")])

    def test_run_diarization_handles_serialized_diarize_output(self):
        class FakeOutput:
            def serialize(self):
                return {
                    "exclusive_diarization": [
                        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_01"},
                    ]
                }

        fake_pipeline = mock.Mock(return_value=FakeOutput())
        with (
            mock.patch.object(transcribe_diarization, "get_diarizer", return_value=fake_pipeline),
            mock.patch.object(
                transcribe_diarization,
                "start_observation",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
            ),
        ):
            segments = transcribe_diarization.run_diarization("audio.wav")

        self.assertEqual(
            segments,
            [
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 2.5, "SPEAKER_01"),
            ],
        )

    def test_run_diarization_logs_clear_warning_for_silent_audio(self):
        fake_pipeline = mock.Mock(return_value={"segments": []})
        with (
            mock.patch.object(transcribe_diarization, "get_diarizer", return_value=fake_pipeline),
            mock.patch.object(
                transcribe_diarization,
                "_inspect_wav_activity",
                return_value={
                    "sampled_windows": 3,
                    "max_peak_dbfs": -91.0,
                    "max_rms_dbfs": -91.0,
                    "likely_silent": True,
                },
            ),
            mock.patch.object(
                transcribe_diarization,
                "start_observation",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
            ),
            mock.patch.object(transcribe_diarization.log, "warning") as warning_mock,
        ):
            segments = transcribe_diarization.run_diarization("audio.wav")

        self.assertEqual(segments, [])
        warning_mock.assert_called_once()
        self.assertIn("audio looks silent or near-silent", warning_mock.call_args.args[0])


class ChooseDeviceTests(unittest.TestCase):
    def test_choose_device_uses_cuda_when_available(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        with (
            mock.patch.dict(transcribe.__dict__, {"CANARY_DEVICE": "cuda"}),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            self.assertEqual(transcribe.choose_device(), "cuda")

    def test_choose_device_fails_when_cuda_required_but_unavailable(self):
        fake_torch = mock.Mock()
        fake_torch.__version__ = "2.8.0"
        fake_torch.version.cuda = None
        fake_torch.cuda.is_available.return_value = False
        with (
            mock.patch.dict(transcribe.__dict__, {"CANARY_DEVICE": "cuda"}),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"torch\.version\.cuda=None.*install\.bat/install\.sh",
            ):
                transcribe.choose_device()

    def test_choose_device_allows_auto_fallback_to_cpu(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        with (
            mock.patch.dict(transcribe.__dict__, {"CANARY_DEVICE": "auto"}),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            self.assertEqual(transcribe.choose_device(), "cpu")

    def test_choose_pyannote_device_uses_cuda_when_available(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.device.side_effect = lambda name: f"device:{name}"
        with (
            mock.patch.dict(transcribe.__dict__, {"PYANNOTE_DEVICE": "auto"}),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            self.assertEqual(transcribe.choose_pyannote_device(), "device:cuda")

    def test_choose_pyannote_device_allows_auto_fallback_to_cpu(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.device.side_effect = lambda name: f"device:{name}"
        with (
            mock.patch.dict(transcribe.__dict__, {"PYANNOTE_DEVICE": "auto"}),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            self.assertEqual(transcribe.choose_pyannote_device(), "device:cpu")

    def test_choose_pyannote_device_fails_when_cuda_required_but_unavailable(self):
        fake_torch = mock.Mock()
        fake_torch.__version__ = "2.8.0"
        fake_torch.version.cuda = None
        fake_torch.cuda.is_available.return_value = False
        with (
            mock.patch.dict(transcribe.__dict__, {"PYANNOTE_DEVICE": "cuda"}),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"PYANNOTE_DEVICE=auto/cpu",
            ):
                transcribe.choose_pyannote_device()

    def test_get_diarizer_moves_pipeline_to_selected_device(self):
        fake_pipeline = mock.Mock()
        fake_pipeline_cls = mock.Mock()
        fake_pipeline_cls.from_pretrained.return_value = fake_pipeline
        fake_pyannote = mock.Mock(Pipeline=fake_pipeline_cls)

        with (
            mock.patch.dict(sys.modules, {"pyannote.audio": fake_pyannote}),
            mock.patch.object(transcribe, "choose_pyannote_device", return_value="device:cuda"),
        ):
            transcribe.get_diarizer.cache_clear()
            diarizer = transcribe.get_diarizer()

        self.assertIs(diarizer, fake_pipeline)
        fake_pipeline.to.assert_called_once_with("device:cuda")
        transcribe.get_diarizer.cache_clear()


class AnalyzeSpeakerFramesTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _active(has_active_speaker: bool, speaker_position: str | None = None) -> str:
        return models.ActiveSpeakerDetection(
            has_active_speaker=has_active_speaker,
            speaker_position=speaker_position,
        ).model_dump_json()

    @staticmethod
    def _caption(has_caption: bool, last_speaker_name: str | None = None) -> str:
        return models.CaptionExtraction(
            has_caption=has_caption,
            last_speaker_name=last_speaker_name,
        ).model_dump_json()

    @staticmethod
    def _appearance(appearance: str) -> str:
        return models.SpeakerAppearance(appearance=appearance).model_dump_json()

    @staticmethod
    def _panel_name(name: str | None) -> str:
        return models.SpeakerPanelName(name=name).model_dump_json()

    async def test_analyze_speaker_frames_prefers_named_candidate(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        diarization_segments = [
            (0.0, 20.0, "SPEAKER_00"),
            (30.0, 36.0, "SPEAKER_00"),
            (40.0, 42.0, "SPEAKER_00"),
        ]
        responses = [
            self._active(True, "top-left"),
            self._caption(False, None),
            self._appearance("person in blue shirt"),
            self._panel_name(None),
            self._active(False, None),
            self._caption(True, "Alice"),
            self._active(False, None),
            self._caption(False, None),
        ]
        extracted_timestamps: list[int] = []

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            extracted_timestamps.append(ts)
            Path(out_path).write_bytes(f"frame-{ts}".encode())
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(frames_analyze, "_ollama_vision_post", side_effect=responses),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                result = frames_analyze.analyze_speaker_frames(
                    "video.mp4",
                    tmp_dir,
                    diarization_segments,
                    queue,
                    loop,
                )

        await asyncio.sleep(0)

        self.assertIn("[SPEAKER_00 @ 33s]", result)
        self.assertIn("name: Alice", result)
        self.assertNotIn("[SPEAKER_00 @ 3s]", result)
        self.assertEqual(extracted_timestamps, [3, 33, 40])
        self.assertEqual(await queue.get(), {"current": 1, "total": 1})
        self.assertIsNone(await queue.get())

    async def test_analyze_speaker_frames_keeps_zero_second_fallback(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        diarization_segments = [(0.0, 1.0, "SPEAKER_00")]
        responses = [
            self._active(True, "middle-center"),
            self._caption(False, None),
            self._appearance("person in red sweater"),
            self._panel_name(None),
        ]

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            Path(out_path).write_bytes(b"frame")
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(frames_analyze, "_ollama_vision_post", side_effect=responses),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                result = frames_analyze.analyze_speaker_frames(
                    "video.mp4",
                    tmp_dir,
                    diarization_segments,
                    queue,
                    loop,
                )

        await asyncio.sleep(0)

        self.assertIn("[SPEAKER_00 @ 0s]", result)
        self.assertIn("position: middle-center", result)

    async def test_analyze_speaker_frames_checks_all_candidate_frames_before_selecting_best(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        diarization_segments = [
            (0.0, 20.0, "SPEAKER_00"),
            (30.0, 36.0, "SPEAKER_00"),
            (40.0, 42.0, "SPEAKER_00"),
        ]
        responses = [
            self._active(True, "top-left"),
            self._caption(False, None),
            self._appearance("person in blue shirt"),
            self._panel_name(None),
            self._active(True, "bottom-left"),
            self._caption(True, "MOHAMMAD"),
            self._appearance("person in dark shirt"),
            self._panel_name("LEV"),
            self._active(False, None),
            self._caption(False, None),
        ]
        extracted_timestamps: list[int] = []

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            extracted_timestamps.append(ts)
            Path(out_path).write_bytes(f"frame-{ts}".encode())
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                with mock.patch.object(
                    frames_analyze,
                    "_ollama_vision_post",
                    side_effect=responses,
                ) as vision_post:
                    result = frames_analyze.analyze_speaker_frames(
                        "video.mp4",
                        tmp_dir,
                        diarization_segments,
                        queue,
                        loop,
                    )

        await asyncio.sleep(0)

        self.assertEqual(extracted_timestamps, [3, 33, 40])
        self.assertEqual(vision_post.call_count, 10)
        self.assertIn("[SPEAKER_00 @ 33s]", result)
        self.assertIn("name: LEV", result)

    async def test_analyze_speaker_frames_prefers_glowing_panel_name_on_conflict(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        diarization_segments = [(0.0, 10.0, "SPEAKER_00")]
        responses = [
            self._active(True, "middle-left"),
            self._caption(True, "MOHAMMAD"),
            self._appearance("person in dark shirt"),
            self._panel_name("LEV"),
        ]

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            Path(out_path).write_bytes(b"frame")
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(frames_analyze, "_ollama_vision_post", side_effect=responses),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                result = frames_analyze.analyze_speaker_frames(
                    "video.mp4",
                    tmp_dir,
                    diarization_segments,
                    queue,
                    loop,
                )

        await asyncio.sleep(0)

        self.assertIn("name: LEV", result)
        self.assertIn("name_source: active_border", result)
        self.assertNotIn("name: MOHAMMAD", result)

    async def test_analyze_speaker_frames_uses_last_name_from_multi_entry_caption(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        diarization_segments = [(0.0, 12.0, "SPEAKER_00")]
        responses = [
            self._active(False, None),
            self._caption(True, "MOHAMMAD M: Yes.\nLEV H: So you tested it"),
        ]

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            Path(out_path).write_bytes(b"frame")
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(frames_analyze, "_ollama_vision_post", side_effect=responses),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                result = frames_analyze.analyze_speaker_frames(
                    "video.mp4",
                    tmp_dir,
                    diarization_segments,
                    queue,
                    loop,
                )

        await asyncio.sleep(0)

        self.assertIn("name: LEV H", result)
        self.assertNotIn("position:", result)

    async def test_analyze_speaker_frames_keeps_caption_only_fallback_without_position(self):
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        diarization_segments = [(0.0, 12.0, "SPEAKER_00")]
        responses = [
            self._active(False, None),
            self._caption(True, "Alice"),
        ]

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            Path(out_path).write_bytes(b"frame")
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                with mock.patch.object(
                    frames_analyze,
                    "_ollama_vision_post",
                    side_effect=responses,
                ) as vision_post:
                    result = frames_analyze.analyze_speaker_frames(
                        "video.mp4",
                        tmp_dir,
                        diarization_segments,
                        queue,
                        loop,
                    )

        await asyncio.sleep(0)

        self.assertEqual(vision_post.call_count, 2)
        self.assertIn("name: Alice", result)
        self.assertIn("name_source: caption", result)
        self.assertNotIn("position:", result)
        self.assertNotIn("appearance:", result)


    async def test_analyze_speaker_frames_self_speaker_beats_active_border(self):
        """Teams self-speaker pattern (no border + caption) should rank above active-border."""
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Two segments for same speaker: first has active border (LEV HIRNYI),
        # second has no border but caption shows SERHII Y (self-speaker pattern).
        diarization_segments = [
            (0.0, 10.0, "SPEAKER_00"),
            (20.0, 30.0, "SPEAKER_00"),
        ]
        responses = [
            # Segment 1 frame: active border visible → active_panel_name="LEV HIRNYI" (rank 2)
            self._active(True, "middle-left"),
            self._caption(True, "LEV HIRNYI"),
            self._appearance("person in dark jacket"),
            self._panel_name("LEV HIRNYI"),
            # Segment 2 frame: no active border, caption shows SERHII Y (rank 3)
            self._active(False, None),
            self._caption(True, "SERHII Y"),
        ]

        def fake_extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
            Path(out_path).write_bytes(b"frame")
            return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(
                    frames_analyze,
                    "extract_single_frame",
                    side_effect=fake_extract_single_frame,
                ),
                mock.patch.object(
                    frames_analyze,
                    "start_observation",
                    side_effect=lambda *args, **kwargs: contextlib.nullcontext(None),
                ),
            ):
                with mock.patch.object(
                    frames_analyze,
                    "_ollama_vision_post",
                    side_effect=responses,
                ):
                    result = frames_analyze.analyze_speaker_frames(
                        "video.mp4",
                        tmp_dir,
                        diarization_segments,
                        queue,
                        loop,
                    )

        await asyncio.sleep(0)

        self.assertIn("name: SERHII Y", result)
        self.assertIn("name_source: caption_self", result)
        self.assertNotIn("name: LEV HIRNYI", result)


class SpeakerFrameResultTests(unittest.TestCase):
    def test_to_context_str_prefers_active_panel_name_over_caption_name(self):
        result = models.SpeakerFrameResult(
            person_visible=True,
            caption_name="MOHAMMAD",
            active_panel_name="LEV",
            appearance="person in blue shirt",
            position="top-left",
        )

        context = result.to_context_str("SPEAKER_00", 12)

        self.assertIn("name: LEV", context)
        self.assertIn("name_source: active_border", context)
        self.assertNotIn("name: MOHAMMAD", context)


class SummaryHelperTests(unittest.TestCase):
    def test_detect_language_heuristically_prefers_ukrainian_chars(self):
        self.assertEqual(
            summary.detect_language_heuristically("Привіт, як твої справи? Це український текст."),
            "uk",
        )

    def test_detect_language_heuristically_detects_russian(self):
        self.assertEqual(
            summary.detect_language_heuristically("Это русский текст, который содержит обычные слова."),
            "ru",
        )

    def test_trim_visual_context_caps_large_input(self):
        text = "a" * (summary.MAX_VISUAL_CONTEXT_CHARS + 50)
        trimmed = summary.trim_visual_context(text)
        self.assertLess(len(trimmed), len(text))
        self.assertIn("[truncated]", trimmed)

    def test_local_preclean_content_normalizes_spacing(self):
        cleaned = summary.local_preclean_content("Hello,\tworld  !\n\n\nNext   line")
        self.assertEqual(cleaned, "Hello, world!\n\nNext line")

    def test_clean_content_uses_dedicated_clean_model(self):
        with mock.patch.object(summary, "call_ollama", return_value="ok") as call_ollama:
            result = summary.clean_content("Hello,\tworld  !", "ctx")

        self.assertEqual(result, "ok")
        self.assertEqual(call_ollama.call_args.kwargs["model"], summary.OLLAMA_CLEAN_MODEL)

    def test_looks_like_missing_content_response_detects_placeholder(self):
        self.assertTrue(
            summary.looks_like_missing_content_response(
                "Please provide the content you would like me to summarize."
            )
        )

    def test_looks_truncated_response_detects_cut_off_sentence(self):
        self.assertTrue(
            summary.looks_truncated_response(
                "**Problem**: The discussion addresses two main technical issues and"
            )
        )

    def test_looks_truncated_response_ignores_completed_sentence(self):
        self.assertFalse(
            summary.looks_truncated_response(
                "**Problem**: The discussion addresses two main technical issues."
            )
        )

    def test_prefer_meaningful_content_falls_back_from_placeholder(self):
        fallback = "[00:00:01] Hello team, let's start the meeting."
        self.assertEqual(
            summary.prefer_meaningful_content(
                "Please provide the content you would like me to summarize.",
                fallback,
            ),
            fallback,
        )

    def test_generate_short_summary_renders_structured_json(self):
        raw = json.dumps(
            {
                "summary": "Коротко: команда обсудила проблему деплоя.",
                "problem": "Продовый деплой падает на миграции базы данных.",
                "ways_to_solve": [
                    "Проверить порядок запуска миграций.",
                    "Добавить предварительную валидацию схемы.",
                ],
                "blockers": ["Нет доступа к логам production у части команды."],
                "estimated_resolution": "Собрать логи сегодня и выкатить фикс завтра утром.",
                "key_points": [],
            },
            ensure_ascii=False,
        )

        with mock.patch.object(summary, "call_ollama", return_value=raw) as call_ollama:
            result = summary.generate_short_summary("Some transcript")

        self.assertEqual(
            result,
            "\n".join(
                [
                    "**Problem**: Продовый деплой падает на миграции базы данных.",
                    "**Ways to solve**:",
                    "- Проверить порядок запуска миграций.",
                    "- Добавить предварительную валидацию схемы.",
                    "**Blockers**:",
                    "- Нет доступа к логам production у части команды.",
                    "**Estimated resolution**: Собрать логи сегодня и выкатить фикс завтра утром.",
                    "**Summary**: Коротко: команда обсудила проблему деплоя.",
                ]
            ),
        )
        self.assertIn('"summary": "short 1-2 sentence TLDR"', call_ollama.call_args.args[0])
        self.assertIn('"key_points": ["important point 1", "important point 2"]', call_ollama.call_args.args[0])
        self.assertEqual(call_ollama.call_args.kwargs["format"], summary.SHORT_SUMMARY_SCHEMA)

    def test_generate_short_summary_accepts_fenced_json(self):
        raw = """```json
        {"summary":"Brief recap.","problem":"","ways_to_solve":[],"blockers":[],"estimated_resolution":"","key_points":["First point.","Second point."]}
        ```"""

        with mock.patch.object(summary, "call_ollama", return_value=raw):
            result = summary.generate_short_summary("Some transcript")

        self.assertEqual(result, "Brief recap.\n- First point.\n- Second point.")

    def test_generate_personal_todo_renders_structured_json(self):
        raw = json.dumps(
            {
                "items": [
                    {
                        "timestamp": "00:12:34",
                        "assigner": "Alice",
                        "action": "Проверить логи и прислать статус в общий чат.",
                    }
                ]
            },
            ensure_ascii=False,
        )

        with (
            mock.patch.object(summary, "load_user_profile", return_value={"primary_name": "Сергей", "aliases": ["Сергей"]}),
            mock.patch.object(summary, "call_ollama", return_value=raw) as call_ollama,
        ):
            result = summary.generate_personal_todo("Meeting transcript")

        self.assertEqual(result, "- [00:12:34] [Alice] → Проверить логи и прислать статус в общий чат.")
        self.assertIn('"items": [{"timestamp": "HH:MM:SS", "assigner": "Name", "action": "Concrete action sentence."}]', call_ollama.call_args.args[0])
        self.assertIn('return exactly: {"items": []}', call_ollama.call_args.args[0])
        self.assertEqual(call_ollama.call_args.kwargs["format"], summary.PERSONAL_TODO_SCHEMA)

    def test_generate_personal_todo_returns_default_message_for_empty_items(self):
        with (
            mock.patch.object(summary, "load_user_profile", return_value={"primary_name": "Сергей", "aliases": ["Сергей"]}),
            mock.patch.object(summary, "call_ollama", return_value='{"items": []}'),
        ):
            result = summary.generate_personal_todo("Meeting transcript")

        self.assertEqual(result, "Задач для Сергей не найдено.")


class DiarizationPreparationTests(unittest.TestCase):
    def test_prepare_diarized_turns_normalizes_overlaps_and_merges_same_speaker(self):
        segments = [
            (4.0, 5.0, "SPEAKER_01"),
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.2, "SPEAKER_00"),
            (2.1, 3.0, "SPEAKER_01"),
        ]

        prepared = transcribe.prepare_diarized_turns(segments)

        self.assertEqual(
            prepared,
            [
                (0.0, 2.2, "SPEAKER_00"),
                (2.2, 3.0, "SPEAKER_01"),
                (4.0, 5.0, "SPEAKER_01"),
            ],
        )

    def test_prepare_diarized_turns_splits_long_monologue(self):
        segments = [(0.0, 45.0, "SPEAKER_00")]

        prepared = transcribe.prepare_diarized_turns(segments, max_duration=20.0)

        self.assertEqual(len(prepared), 3)
        self.assertEqual(prepared[0][2], "SPEAKER_00")
        self.assertLessEqual(prepared[0][1] - prepared[0][0], 20.0)
        self.assertEqual(prepared[0][0], 0.0)
        self.assertEqual(prepared[-1][1], 45.0)


class NotificationTests(unittest.IsolatedAsyncioTestCase):
    def test_ntfy_payload_supports_unicode_title(self):
        payload = helpers._ntfy_payload("Готово", "Видео обработано")

        self.assertEqual(payload["topic"], helpers.NTFY_TOPIC)
        self.assertEqual(payload["title"], "Готово")
        self.assertEqual(payload["message"], "Видео обработано")
        self.assertEqual(payload["priority"], 3)
        self.assertEqual(payload["tags"], ["white_check_mark"])

    def test_desktop_notifications_require_gui_and_dbus(self):
        with mock.patch.dict(helpers.os.environ, {}, clear=True):
            self.assertFalse(helpers._desktop_notifications_available())

        with mock.patch.dict(
            helpers.os.environ,
            {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/tmp/dbus", "DISPLAY": ":0"},
            clear=True,
        ):
            self.assertTrue(helpers._desktop_notifications_available())

    async def test_notify_done_skips_desktop_notification_without_session(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        post = mock.AsyncMock(return_value=response)
        client = mock.AsyncMock()
        client.__aenter__.return_value = client
        client.post = post

        with (
            mock.patch.object(helpers.httpx, "AsyncClient", return_value=client),
            mock.patch.object(helpers, "_desktop_notifications_available", return_value=False),
            mock.patch.object(helpers.log, "warning") as warning_log,
        ):
            await helpers.notify_done("Готово", "Видео обработано")

        post.assert_awaited_once_with(
            helpers.NTFY_URL,
            json=helpers._ntfy_payload("Готово", "Видео обработано"),
        )
        warning_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
