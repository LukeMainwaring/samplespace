import logging

from pydantic_ai import RunContext, ToolReturn

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.formatting import format_sample_results
from samplespace.models.sample import Sample
from samplespace.models.thread import Thread
from samplespace.schemas.agent_type import AgentType
from samplespace.schemas.sample import SampleSchema
from samplespace.schemas.thread import SongContext

logger = logging.getLogger(__name__)


async def find_similar_to_upload(ctx: RunContext[AgentDeps], sample_id: str) -> str | ToolReturn:
    """Find library samples similar to an uploaded reference track using CLAP embeddings.

    Use this tool when the user has uploaded a WAV file (a song, snippet, or reference
    track) and wants to find similar samples from the sample library. The search compares
    the uploaded sample's audio embedding against the library.

    Args:
        sample_id: The ID of the uploaded sample to find similar matches for.
    """
    try:
        sample = await Sample.get(ctx.deps.db, sample_id)
        if sample is None:
            return f"Sample {sample_id} not found."

        if sample.source != "upload":
            return f"Sample {sample_id} is not an uploaded sample. Use find_similar_samples for library samples."

        if sample.clap_embedding is None:
            return f"Sample {sample_id} does not have a CLAP embedding. It may not have been processed correctly."

        # Search the library (exclude uploads) using the uploaded sample's CLAP embedding
        results = await Sample.search_by_clap(
            ctx.deps.db,
            sample.clap_embedding,
            exclude_source="upload",
            limit=10,
        )

        if not results:
            return "No similar library samples found."

        formatted = [SampleSchema.model_validate(s) for s in results]
        return format_sample_results(
            formatted,
            f'Found {len(formatted)} library samples similar to uploaded file "{sample.filename}":',
        )
    except Exception:
        logger.exception("Error in upload similarity search")
        return "An error occurred while finding similar samples."


async def find_upload(ctx: RunContext[AgentDeps], query: str) -> str | ToolReturn:
    """Search uploaded samples by filename.

    Use this when the user wants to find a previously uploaded reference track
    by name (e.g., "find my southern twang upload").

    Args:
        query: Search term to match against upload filenames (case-insensitive).
    """
    try:
        uploads = await Sample.get_by_source(ctx.deps.db, "upload")
        if not uploads:
            return "No uploaded samples found."

        query_lower = query.lower()
        matches = [s for s in uploads if query_lower in s.filename.lower()]

        if not matches:
            all_names = ", ".join(s.filename for s in uploads)
            return f"No uploads matching '{query}'. Available uploads: {all_names}"

        formatted = [SampleSchema.model_validate(s) for s in matches]
        header = f"Found {len(formatted)} uploaded sample{'s' if len(formatted) != 1 else ''}:"
        result = format_sample_results(formatted, header)
        # Append a hint so the LLM knows which IDs to use in follow-up tool calls
        if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
            ids = ", ".join(s.id for s in matches)
            result.return_value += f"\n\nUse sample ID(s) for follow-up tools: {ids}"
        return result
    except Exception:
        logger.exception("Error searching uploads")
        return "An error occurred while searching uploads."


async def set_context_from_upload(
    ctx: RunContext[AgentDeps],
    sample_id: str,
    genre: str | None = None,
    vibe: str | None = None,
) -> str:
    """Set song context from an uploaded sample's detected key and BPM, plus optional genre/vibe.

    Use this after finding an uploaded reference track to align the conversation's
    song context with the upload's metadata, so subsequent searches are context-aware.
    Key and BPM are extracted from the upload's analysis; genre and vibe can be
    provided by the user to further refine searches.

    Args:
        sample_id: The ID of the uploaded sample to extract context from.
        genre: Genre or style, e.g., "lo-fi hip hop", "techno".
        vibe: Mood/character description, e.g., "dark and atmospheric", "uplifting".
    """
    try:
        sample = await Sample.get(ctx.deps.db, sample_id)
        if sample is None:
            return f"Sample {sample_id} not found."

        if sample.source != "upload":
            return f"Sample {sample_id} is not an uploaded sample."

        if not ctx.deps.thread_id:
            return "Cannot set song context — no active thread."

        provided: dict[str, object] = {}
        if sample.key:
            provided["key"] = sample.key
        if sample.bpm and sample.bpm > 0:
            provided["bpm"] = sample.bpm
        if genre:
            provided["genre"] = genre
        if vibe:
            provided["vibe"] = vibe

        if not provided:
            return (
                f"No key or BPM detected for '{sample.filename}'. "
                "You can set the song context manually with set_song_context."
            )

        updates = SongContext.model_validate(provided)
        merged = await Thread.update_song_context(ctx.deps.db, ctx.deps.thread_id, AgentType.CHAT.value, updates)
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

        return f"Song context set from '{sample.filename}' — {', '.join(parts)}"
    except Exception:
        logger.exception("Error setting context from upload")
        return "An error occurred while setting song context from the upload."
