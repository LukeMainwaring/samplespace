from samplespace.schemas.sample import SampleSchema


def format_sample_results(
    results: list[SampleSchema],
    header: str,
    *,
    include_duration: bool = False,
    annotations: dict[str, str] | None = None,
) -> str:
    """Format a list of sample results as numbered markdown lines.

    Args:
        annotations: Optional map of sample_id -> annotation string
                     (e.g., " ✓ same key") appended after metadata.
    """
    lines = [f"{header}\n"]
    for i, s in enumerate(results, 1):
        parts = [f"{i}. **{s.filename}**"]
        if s.sample_type:
            parts.append(f"type={s.sample_type}")
        parts.append("loop" if s.is_loop else "one-shot")
        if s.is_loop:
            key_part = f"key={s.key}" if s.key else None
            if key_part:
                annotation = annotations.get(s.id, "") if annotations else ""
                parts.append(f"{key_part}{annotation}")
            if s.bpm and s.bpm > 0:
                parts.append(f"bpm={s.bpm}")
        if include_duration and s.duration:
            parts.append(f"duration={s.duration:.1f}s")
        parts.append(f"id={s.id}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def sample_to_payload(sample: SampleSchema, audio_url: str | None = None) -> dict[str, object]:
    """Build a JSON-serializable payload dict for a sample."""
    payload: dict[str, object] = {
        "id": sample.id,
        "filename": sample.filename,
        "audio_url": audio_url or f"/api/samples/{sample.id}/audio",
    }
    if sample.sample_type:
        payload["type"] = sample.sample_type
    if sample.is_loop:
        if sample.key:
            payload["key"] = sample.key
        if sample.bpm and sample.bpm > 0:
            payload["bpm"] = sample.bpm
    return payload
