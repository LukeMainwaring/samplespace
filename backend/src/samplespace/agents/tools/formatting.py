from pydantic_ai import ToolReturn
from pydantic_ai.ui.vercel_ai.response_types import DataChunk

from samplespace.schemas.sample import SampleSchema


def format_sample_results(
    results: list[SampleSchema],
    header: str,
    *,
    annotations: dict[str, str] | None = None,
) -> ToolReturn:
    """Build a ToolReturn with a DataChunk for the frontend to render as sample cards."""
    samples: list[dict[str, object]] = []
    summary_lines: list[str] = [header]
    for i, s in enumerate(results, start=1):
        payload = sample_to_payload(s, index=i)
        if annotations and s.id in annotations:
            payload["annotation"] = annotations[s.id]
        samples.append(payload)
        summary_lines.append(_sample_summary_line(s, index=i))

    return ToolReturn(
        return_value="\n".join(summary_lines),
        metadata=DataChunk(type="data-sample-results", data={"samples": samples}),
    )


def _sample_summary_line(sample: SampleSchema, *, index: int) -> str:
    """One-line summary with ID for LLM follow-up calls."""
    parts = [f"{index}. {sample.id} — {sample.filename}"]
    meta: list[str] = []
    if sample.sample_type:
        meta.append(sample.sample_type)
    if sample.is_loop:
        if sample.key:
            meta.append(sample.key)
        if sample.bpm and sample.bpm > 0:
            meta.append(f"{sample.bpm} BPM")
    if meta:
        parts.append(f"({', '.join(meta)})")
    return " ".join(parts)


def sample_to_payload(
    sample: SampleSchema,
    audio_url: str | None = None,
    index: int | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable payload dict for a sample."""
    payload: dict[str, object] = {
        "id": sample.id,
        "filename": sample.filename,
        "audio_url": audio_url or f"/api/samples/{sample.id}/audio",
        "is_loop": sample.is_loop,
    }
    if index is not None:
        payload["index"] = index
    if sample.sample_type:
        payload["type"] = sample.sample_type
    if sample.is_loop:
        if sample.key:
            payload["key"] = sample.key
        if sample.bpm and sample.bpm > 0:
            payload["bpm"] = sample.bpm
    return payload
