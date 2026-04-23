from __future__ import annotations

import shutil
from dataclasses import dataclass


LegacySegment = tuple[float, float, str]


@dataclass(frozen=True)
class DiarizationSegment:
    """Speaker-attributed time span in seconds."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class AudioChunk:
    """Extracted WAV chunk for a diarized speaker segment."""

    segment: DiarizationSegment
    path: str


@dataclass(frozen=True)
class WavConversionResult:
    """Metadata returned after converting media into a mono 16 kHz WAV file."""

    file_info: str
    format_name: str
    duration_display: str
    duration_sec: float
    codec: str
    has_video: bool
    output_path: str

    def to_payload(self) -> dict[str, object]:
        return {
            "file_info": self.file_info,
            "format": self.format_name,
            "duration": self.duration_display,
            "duration_sec": self.duration_sec,
            "codec": self.codec,
            "has_video": self.has_video,
            "output_path": self.output_path,
        }

    def __getitem__(self, key: str):
        return self.to_payload()[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


@dataclass(frozen=True)
class PreparedAudio:
    """Temporary WAV file prepared for transcription."""

    path: str
    temp_dir: str

    def cleanup(self) -> None:
        """Remove the temporary directory that owns this prepared file."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
