"""Music theory utilities for key compatibility and pitch transformation."""

# Chromatic note index (C=0, C#=1, ..., B=11) for semitone calculations
CHROMATIC_INDEX: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

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


def _parse_mode(key: str) -> str | None:
    """Extract the mode ('major' or 'minor') from a key string like 'C major'."""
    parts = key.split()
    if len(parts) == 2 and parts[1] in ("major", "minor"):
        return parts[1]
    return None


def semitone_delta(from_key: str, to_key: str) -> int | None:
    """Signed chromatic distance between two key root notes, preferring shorter direction.

    Computes distance based on root notes only — mode is ignored. For cross-mode
    transformations, use compute_target_key() first to resolve the actual target.

    Returns a value in the range [-6, +6], or None if either key is unparseable.
    Positive means shift up, negative means shift down. The tritone (6 semitones)
    is always returned as +6 by convention.
    """
    from_root = _parse_root(from_key)
    to_root = _parse_root(to_key)
    if from_root is None or to_root is None:
        return None
    from_idx = CHROMATIC_INDEX[from_root]
    to_idx = CHROMATIC_INDEX[to_root]
    delta = (to_idx - from_idx) % 12
    if delta > 6:
        delta -= 12
    return delta


def compute_target_key(sample_key: str, song_key: str) -> str | None:
    """Determine the best pitch-shift target for a sample given the song's key.

    Same mode (minor→minor or major→major): shift root to song's root.
        e.g. "D minor" + "G minor" song → "G minor"

    Different mode: target the relative key of the song context that matches
    the sample's mode. Relative keys share the same key signature, producing
    maximum harmonic compatibility.
        e.g. "D minor" + "G major" song → "E minor" (relative minor of G major)
        e.g. "F major" + "A minor" song → "C major" (relative major of A minor)

    Returns None if either key is unparseable.
    """
    sample_mode = _parse_mode(sample_key)
    song_mode = _parse_mode(song_key)
    song_root = _parse_root(song_key)
    if sample_mode is None or song_mode is None or song_root is None:
        return None

    if sample_mode == song_mode:
        return f"{song_root} {sample_mode}"

    # Cross-mode: find the relative key of the song that matches the sample's mode
    relative = RELATIVE_PAIRS.get(song_key)
    if relative is not None:
        return relative

    # Fallback: parallel key (same root, sample's mode)
    return f"{song_root} {sample_mode}"
