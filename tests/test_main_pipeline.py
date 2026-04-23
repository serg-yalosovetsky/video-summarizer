import asyncio
import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_test_stubs() -> None:
    return None


install_test_stubs()
main = importlib.import_module("main")
transcribe = importlib.import_module("transcribe")
helpers = importlib.import_module("helpers")


def decode_events(messages: list[str]) -> list[dict]:
    result = []
    for message in messages:
        payload = message.removeprefix("data: ").strip()
        result.append(json.loads(payload))
    return result


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

        with (
            mock.patch.object(main, "convert_to_wav", side_effect=fake_convert_to_wav),
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
        tldr_event = next(event for event in events if event["event"] == "tldr_done")
        self.assertEqual(tldr_event["payload"]["title"], "Краткое саммари")
        self.assertFalse(tldr_event["payload"]["is_meeting"])

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
                r"install\.bat/install\.sh.*torch\.version\.cuda=None",
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
