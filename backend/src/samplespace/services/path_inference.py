import logging
import re
from pathlib import PurePosixPath

from samplespace.schemas.sample_type import KEYWORD_TO_SAMPLE_TYPE

logger = logging.getLogger(__name__)


def infer_sample_type_from_path(relative_path: str) -> str | None:
    """Infer sample_type by scanning path segments for known keywords.

    Scans filename stem first, then directories deepest-to-shallowest, to prefer
    the most specific match. Splits names on underscores and checks segment
    membership to avoid false positives from substring matching.
    """
    path = PurePosixPath(relative_path)
    stem = path.stem  # filename without extension
    parts = [*path.parts[:-1], stem]  # directories + filename stem
    for part in reversed(parts):
        normalized = part.lower().replace("-", "_").replace(" ", "_")
        # Exact directory name match
        if normalized in KEYWORD_TO_SAMPLE_TYPE:
            return KEYWORD_TO_SAMPLE_TYPE[normalized]
        # Split into segments, strip trailing digits (e.g. "synth1" -> "synth")
        raw_segments = normalized.split("_")
        segments = {re.sub(r"\d+$", "", s) for s in raw_segments} | set(raw_segments)
        for kw, sample_type in KEYWORD_TO_SAMPLE_TYPE.items():
            if kw in segments:
                return sample_type
    scanned = [p.lower().replace("-", "_").replace(" ", "_") for p in parts]
    logger.info(f"No sample_type matched for '{relative_path}' — scanned segments: {scanned}\n    ------------------")
    return None


def extract_pack_name(relative_path: str) -> str:
    """Extract the pack name from the first path component."""
    return PurePosixPath(relative_path).parts[0]
