from samplespace.schemas.sample import SampleSchema


def format_sample_results(
    results: list[SampleSchema],
    header: str,
    *,
    include_duration: bool = False,
) -> str:
    """Format a list of sample results as numbered markdown lines."""
    lines = [f"{header}\n"]
    for i, s in enumerate(results, 1):
        parts = [f"{i}. **{s.filename}**"]
        if s.sample_type:
            parts.append(f"type={s.sample_type}")
        parts.append("loop" if s.is_loop else "one-shot")
        if s.is_loop:
            if s.key:
                parts.append(f"key={s.key}")
            if s.bpm and s.bpm > 0:
                parts.append(f"bpm={s.bpm}")
        if include_duration and s.duration:
            parts.append(f"duration={s.duration:.1f}s")
        parts.append(f"id={s.id}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
