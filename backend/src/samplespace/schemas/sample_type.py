"""Canonical sample type definitions.

Single source of truth for sample type categories used across
path inference, ML training, and API contracts.
"""

from enum import StrEnum


class SampleType(StrEnum):
    BASS = "bass"
    CLAP = "clap"
    CYMBAL = "cymbal"
    DRUM = "drum"
    FX = "fx"
    GUITAR = "guitar"
    HIHAT = "hihat"
    HORN = "horn"
    KEYS = "keys"
    KICK = "kick"
    PAD = "pad"
    PERCUSSION = "percussion"
    SNARE = "snare"
    STRINGS = "strings"
    SYNTH = "synth"
    VOCAL = "vocal"


SAMPLE_TYPES: list[str] = sorted(t.value for t in SampleType)

# Keyword-to-type mapping for inferring sample type from file paths.
# Keys are SampleType enum members; values are directory/segment keywords
# that map to that type (checked against lowercased path segments).
SAMPLE_TYPE_KEYWORDS: dict[SampleType, list[str]] = {
    SampleType.KICK: ["kicks", "kick"],
    SampleType.SNARE: ["snares", "snare"],
    SampleType.CLAP: ["claps", "clap"],
    SampleType.HIHAT: ["hihats", "hi_hats", "hi-hats", "hihat", "hi_hat", "hi-hat", "hats", "hat"],
    SampleType.CYMBAL: ["cymbals", "rides", "crashes", "cymbal", "ride", "crash"],
    SampleType.PERCUSSION: [
        "percussion",
        "perc",
        "shakers",
        "shaker",
        "cowbells",
        "cowbell",
        "top",
        "kit",
        "sidestick",
        "gong",
    ],
    SampleType.DRUM: ["drums", "drum_loops", "drum_one_shots", "drum_fills", "drum", "break", "breaks"],
    SampleType.BASS: ["bass", "basses", "808s", "808", "bassline", "basslines"],
    SampleType.VOCAL: ["vocals", "vox", "vocal", "choir", "choirs", "voice", "talkbox"],
    SampleType.SYNTH: ["synths", "synth", "leads", "lead", "arp", "arps", "stab", "stabs"],
    SampleType.PAD: ["pads", "pad", "atmosphere", "atmospheres", "drone", "drones", "texture", "textures"],
    SampleType.KEYS: ["keys", "piano", "organ", "electric_piano", "clav"],
    SampleType.GUITAR: [
        "guitars",
        "guitar",
        "electric_guitar",
        "electric_guitars",
        "banjo",
        "pedal_steel",
        "guitarra",
    ],
    SampleType.STRINGS: ["strings", "string", "violin", "violins", "viola", "cello", "cellos", "zither"],
    SampleType.HORN: [
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
        "ney",
        "didgeridoo",
    ],
    SampleType.FX: ["fx", "sfx", "effects", "risers", "impacts", "sweeps", "foley", "field_recordings"],
}

# Flattened lookup: keyword -> sample type value
KEYWORD_TO_SAMPLE_TYPE: dict[str, str] = {
    kw: st.value for st, keywords in SAMPLE_TYPE_KEYWORDS.items() for kw in keywords
}
