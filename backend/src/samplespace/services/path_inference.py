from pathlib import PurePosixPath

from samplespace.schemas.sample_type import KEYWORD_TO_SAMPLE_TYPE


def infer_sample_type_from_path(relative_path: str) -> str | None:
    """Infer sample_type by scanning path segments for known keywords.

    Scans from deepest to shallowest directory to prefer the most specific match.
    Splits directory names on underscores and checks segment membership to avoid
    false positives from substring matching (e.g., "pad" inside "padding").
    """
    parts = PurePosixPath(relative_path).parts[:-1]  # exclude filename
    for part in reversed(parts):
        normalized = part.lower().replace("-", "_").replace(" ", "_")
        # Exact directory name match
        if normalized in KEYWORD_TO_SAMPLE_TYPE:
            return KEYWORD_TO_SAMPLE_TYPE[normalized]
        # Split into segments and check membership
        segments = set(normalized.split("_"))
        for kw, sample_type in KEYWORD_TO_SAMPLE_TYPE.items():
            if kw in segments:
                return sample_type
    return None


def extract_pack_name(relative_path: str) -> str:
    """Extract the pack name from the first path component."""
    return PurePosixPath(relative_path).parts[0]
