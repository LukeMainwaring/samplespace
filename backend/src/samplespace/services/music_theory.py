"""Music theory utilities for key compatibility scoring."""

# Circle of fifths for key distance calculation
CIRCLE_OF_FIFTHS = [
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
    "F",
]

# Relative major/minor pairs (bidirectional)
RELATIVE_PAIRS: dict[str, str] = {
    "C major": "A minor",
    "G major": "E minor",
    "D major": "B minor",
    "A major": "F# minor",
    "E major": "C# minor",
    "B major": "G# minor",
    "F# major": "D# minor",
    "C# major": "A# minor",
    "G# major": "F minor",
    "D# major": "C minor",
    "A# major": "G minor",
    "F major": "D minor",
}
RELATIVE_PAIRS.update({v: k for k, v in RELATIVE_PAIRS.items()})

# Map circle-of-fifths distance (0-6) to a 0.0-1.0 compatibility score
_DISTANCE_SCORES: dict[int, float] = {
    0: 1.0,
    1: 0.85,
    2: 0.65,
    3: 0.4,
    4: 0.2,
    5: 0.1,
    6: 0.1,
}


def _parse_root(key: str) -> str | None:
    """Extract the root note from a key string like 'C major' or 'A minor'."""
    root = key.split()[0] if " " in key else key
    if root in CIRCLE_OF_FIFTHS:
        return root
    return None


def key_distance(key1: str, key2: str) -> int | None:
    """Circle-of-fifths distance between two keys (0-6).

    Returns None if either key's root note cannot be parsed.
    """
    root1 = _parse_root(key1)
    root2 = _parse_root(key2)
    if root1 is None or root2 is None:
        return None
    idx1 = CIRCLE_OF_FIFTHS.index(root1)
    idx2 = CIRCLE_OF_FIFTHS.index(root2)
    return min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))


def are_relative_pairs(key1: str, key2: str) -> bool:
    """Check if two keys are relative major/minor pairs."""
    return RELATIVE_PAIRS.get(key1) == key2


def key_compatibility_score(key1: str, key2: str) -> tuple[float, str]:
    """Score key compatibility from 0.0 (clashing) to 1.0 (perfect).

    Returns (score, explanation) tuple.
    """
    if key1 == key2:
        return 1.0, f"same key ({key1})"

    if are_relative_pairs(key1, key2):
        return 0.95, f"relative major/minor pair ({key1} / {key2})"

    distance = key_distance(key1, key2)
    if distance is None:
        return 0.5, f"could not determine distance between {key1} and {key2}"

    score = _DISTANCE_SCORES.get(distance, 0.1)
    if distance <= 1:
        label = "adjacent on circle of fifths"
    elif distance <= 2:
        label = "close on circle of fifths"
    else:
        label = "distant on circle of fifths"

    return score, f"{label} (distance {distance})"
