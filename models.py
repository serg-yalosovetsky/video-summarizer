from __future__ import annotations

from pydantic import BaseModel, Field


class SpeakerFrameResult(BaseModel):
    person_visible: bool = Field(description="Whether a person is clearly visible in the frame")
    name: str | None = Field(default=None, description="Full name from name tag, lower-third, or on-screen label; null if not visible")
    appearance: str = Field(default="", description="Brief appearance description: gender, clothing, hair colour")
    position: str = Field(default="", description="Position in frame: left, center, or right")

    def to_context_str(self, speaker_id: str, ts: int) -> str:
        parts = []
        if self.name:
            parts.append(f"name: {self.name}")
        if self.appearance:
            parts.append(self.appearance)
        if self.position:
            parts.append(f"position: {self.position}")
        desc = ", ".join(parts) if parts else "person visible, name unknown"
        return f"[{speaker_id} @ {ts}s] {desc}"


class FrameAnalysisResult(BaseModel):
    setting: str = Field(default="", description="Scene description: meeting room, lecture hall, interview, etc.")
    people: list[str] = Field(default_factory=list, description="List of visible people — use name if shown on screen, otherwise brief description")
    on_screen_text: list[str] = Field(default_factory=list, description="Any visible text: titles, logos, slides, lower-thirds")

    def to_context_str(self) -> str:
        parts = []
        if self.setting:
            parts.append(self.setting)
        if self.people:
            parts.append("People: " + ", ".join(self.people))
        if self.on_screen_text:
            parts.append("Text: " + "; ".join(self.on_screen_text))
        return ". ".join(parts) if parts else "no relevant information"
