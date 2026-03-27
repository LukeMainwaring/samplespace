"""Song context tool for the sample assistant agent."""

import logging

from pydantic_ai import RunContext

from samplespace.agents.deps import AgentDeps
from samplespace.models.thread import Thread
from samplespace.schemas.agent_type import AgentType
from samplespace.schemas.thread import SongContext

logger = logging.getLogger(__name__)


async def set_song_context(
    ctx: RunContext[AgentDeps],
    key: str | None = None,
    bpm: int | None = None,
    genre: str | None = None,
    vibe: str | None = None,
) -> str:
    """Set or update the song context for this conversation.

    Call this when the user mentions they're working on a track in a specific
    key, BPM, genre, or vibe. Only provide the fields that are being set or
    changed — existing fields are preserved.

    Args:
        key: Musical key, e.g., "C major", "A minor".
        bpm: Tempo in beats per minute.
        genre: Genre or style, e.g., "lo-fi hip hop", "techno".
        vibe: Mood/character description, e.g., "dark and atmospheric", "uplifting".
    """
    # Only include fields the agent explicitly provided so exclude_unset works correctly.
    # This allows clearing a field by passing None (e.g., vibe=None removes the vibe).
    provided = {k: v for k, v in {"key": key, "bpm": bpm, "genre": genre, "vibe": vibe}.items() if v is not None}
    if not provided:
        return "No song context fields provided. Specify at least one of: key, bpm, genre, vibe."
    updates = SongContext.model_validate(provided)

    if not ctx.deps.thread_id:
        return "Cannot set song context — no active thread."

    try:
        merged = await Thread.update_song_context(ctx.deps.db, ctx.deps.thread_id, AgentType.CHAT.value, updates)
        # Update in-memory deps so subsequent tool calls in this turn see the change
        ctx.deps.song_context = merged

        parts = []
        if merged.key:
            parts.append(f"Key: {merged.key}")
        if merged.bpm:
            parts.append(f"BPM: {merged.bpm}")
        if merged.genre:
            parts.append(f"Genre: {merged.genre}")
        if merged.vibe:
            parts.append(f"Vibe: {merged.vibe}")

        return f"Song context updated — {', '.join(parts)}"
    except Exception:
        logger.exception("Error updating song context")
        return "An error occurred while updating the song context."
