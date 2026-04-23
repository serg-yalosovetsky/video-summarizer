from __future__ import annotations

from pydantic import BaseModel, Field


class SpeakerFrameResult(BaseModel):
    person_visible: bool = Field(description="Whether a person is clearly visible in the frame")
    caption_name: str | None = Field(
        default=None,
        description=(
            "Speaker name extracted from the caption/subtitle overlay at the bottom of the screen "
            "(text formatted as 'NAME: spoken words'). Copy only the part before the colon. "
            "Null if no such caption is visible."
        ),
    )
    active_panel_name: str | None = Field(
        default=None,
        description=(
            "Name label on the participant panel that has a glowing, highlighted, or bright-colored border, "
            "or a glowing ring/halo around a circular avatar. Null if no such panel is visible."
        ),
    )
    appearance: str = Field(default="", description="Brief appearance description: gender, clothing, hair colour")
    position: str = Field(default="", description="Grid position of the speaker's panel in the frame. One of: top-left, top-center, top-right, middle-left, middle-center, middle-right, bottom-left, bottom-center, bottom-right")

    def to_context_str(self, speaker_id: str, ts: int) -> str:
        name = self.caption_name or self.active_panel_name
        source = "caption" if self.caption_name else ("active_border" if self.active_panel_name else "")
        parts = []
        if name:
            parts.append(f"name: {name}")
        if source:
            parts.append(f"name_source: {source}")
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
