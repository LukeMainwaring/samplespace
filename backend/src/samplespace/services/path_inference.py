"""Infer sample metadata from file paths (sample_type, pack_name)."""

from pathlib import PurePosixPath

# Keyword-to-type mapping, checked against lowercased path segments.
# Order matters: more specific keywords first to avoid partial matches.
_SAMPLE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "kick": ["kicks", "kick"],
    "snare": ["snares", "snare"],
    "clap": ["claps", "clap"],
    "hihat": ["hihats", "hi_hats", "hi-hats", "hihat", "hi_hat", "hi-hat"],
    "cymbal": ["cymbals", "rides", "crashes", "cymbal", "ride", "crash"],
    "percussion": ["percussion", "perc", "shakers", "shaker", "cowbells", "cowbell"],
    "drum": ["drums", "drum_loops", "drum_one_shots", "drum_fills", "drum"],
    "bass": ["bass", "basses", "808s", "808"],
    "vocal": ["vocals", "vox", "vocal", "choir", "choirs"],
    "synth": ["synths", "synth", "leads", "lead"],
    "pad": ["pads", "pad"],
    "keys": ["keys", "piano", "organ", "electric_piano"],
    "guitar": ["guitars", "guitar", "electric_guitar", "electric_guitars"],
    "strings": ["strings", "string"],
    "horn": [
        "horns",
        "horn",
        "brass",
        "woodwinds",
        "woodwind",
        "trumpet",
        "trumpets",
        "trombone",
        "trombones",
        "saxophone",
        "saxophones",
        "sax",
        "flute",
        "flutes",
        "clarinet",
        "clarinets",
        "tuba",
        "oboe",
    ],
    "fx": ["fx", "sfx", "effects", "risers", "impacts", "sweeps"],
}

# Flatten for lookup: keyword -> sample_type
_KEYWORD_TO_TYPE: dict[str, str] = {}
for sample_type, keywords in _SAMPLE_TYPE_KEYWORDS.items():
    for kw in keywords:
        _KEYWORD_TO_TYPE[kw] = sample_type


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
        if normalized in _KEYWORD_TO_TYPE:
            return _KEYWORD_TO_TYPE[normalized]
        # Split into segments and check membership
        segments = set(normalized.split("_"))
        for kw, sample_type in _KEYWORD_TO_TYPE.items():
            if kw in segments:
                return sample_type
    return None


def extract_pack_name(relative_path: str) -> str:
    """Extract the pack name from the first path component."""
    return PurePosixPath(relative_path).parts[0]
