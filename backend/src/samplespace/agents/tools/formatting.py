import json

from samplespace.schemas.sample import SampleSchema


def format_sample_results(
    results: list[SampleSchema],
    header: str,
    *,
    annotations: dict[str, str] | None = None,
) -> str:
    """Format a list of sample results as a playable sample-results code fence."""
    samples: list[dict[str, object]] = []
    for i, s in enumerate(results, start=1):
        payload = sample_to_payload(s, index=i)
        if annotations and s.id in annotations:
            payload["annotation"] = annotations[s.id]
        samples.append(payload)

    json_str = json.dumps({"samples": samples}, indent=2)
    return f"{header}\n\n```sample-results\n{json_str}\n```"


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
