# Feature Brainstorm (Archive)

All features below are implemented (phases 1-3 complete). See `docs/ROADMAP.md` for remaining work.

## Origin

These features were inspired by reviewing the predecessor project ([music-sample-assistant](https://github.com/LukeMainwaring/music-sample-assistant)), which had three ideas that SampleSpace now carries forward:

1. **"Match to my song"** — the user provides their song's key and BPM, and the system finds compatible samples, then pitch-shifts and time-stretches them to fit. Implemented via persistent song context + the `match_to_context` tool.
2. **Pair-quality prediction** — predicting whether two samples sound good together. Implemented via multi-dimensional pair scoring (key, BPM, type complementarity, CNN spectral distance) and the pairing feedback loop.
3. **Explicit song context** — persistent key/BPM/genre/vibe per conversation thread. Implemented as a JSONB column on the `threads` table, mutated only through the agent.

The opportunity was to keep the agent's flexibility for open-ended queries while adding structured workflows (song context, audio transformation, pair learning, kit building) that make SampleSpace feel like a production tool rather than a search interface.

---

## Dependency Graph

```
                    +--------------------------+
                    | 1. Song Context /        |
                    |    Session State         |
                    +-----------+--------------+
                                |
              +-----------------+------------------+
              |                 |                   |
              v                 v                   v
+-------------+--+   +---------+--------+   +------+----------+
| 2. Pair        |   | 3. Audio         |   | 4. Sample       |
|   Compatibility|   |   Transformation |   |   Upload        |
|   Scoring      |   |                  |   |                 |
+------+---------+   +--------+---------+   +------+----------+
       |                      |                     |
       +----------+-----------+                     |
                  |                                  |
                  v                                  |
       +----------+-----------+                     |
       | 5. Pairing &         |<--------------------+
       |    Feedback Loop     |
       +----------+-----------+
                  |
                  v
       +----------+-----------+
       | 6. Kit Builder        |
       +-----------------------+
```

## Phased Build Order

- **Phase 1 (Context Foundation)**: Song Context + Pair Compatibility Scoring — complete
- **Phase 2 (Audio Pipeline)**: Audio Transformation + Sample Upload — complete
- **Phase 3 (Learning & Assembly)**: Pairing & Feedback Loop (stages 1-3) + Kit Builder — complete

---

## Feature Specs

### 1. Song Context (Thread-Backed)

**What**: The user tells the agent about their song through natural conversation ("I'm working on a track in D minor at 120 BPM, kind of a dark techno vibe"). The agent persists this context to the thread record and automatically applies it when searching, scoring, and recommending. Context survives page refreshes and is unique per conversation thread.

**Why**: Without context, every search is stateless — the user has to repeat "in D minor at 120 BPM" with every request. Song context transforms the interaction from "search engine" to "production assistant that knows your song."

**Depends on**: Threads & messages.

---

### 2. Pair Compatibility Scoring

**What**: A multi-dimensional compatibility score between two samples, combining key compatibility (circle of fifths distance), BPM compatibility, type complementarity (kick+hihat > kick+kick), and CNN embedding distance.

**Why**: The original `check_key_compatibility` only considered key. Real pairing decisions involve multiple dimensions. A composite score lets the agent rank candidate pairs and explain *why* two samples work together (or don't).

**Depends on**: Feature 1 (Song Context).

---

### 3. Real-time Audio Transformation

**What**: An agent tool that pitch-shifts and/or time-stretches a sample to match a target key and BPM, returning a playable transformed audio file inline in the chat.

**Why**: Finding a great pad in E minor is useless if the song is in G minor — unless the system can transpose it. This is the difference between "here are samples that might work" and "here's a sample that *does* work in your song."

**Depends on**: Feature 1 (Song Context).

---

### 4. Sample Upload / "Find More Like This"

**What**: The user uploads a WAV file (often a full song or snippet used as a reference track). The system analyzes it (key, BPM, duration, type), generates a CLAP embedding, and stores it permanently. The user can then ask the agent to find similar samples from the splice library using CLAP audio-to-audio similarity.

**Why**: Closes the loop between "I have audio I like" and "find me more." Previously users could only search by text or reference samples already in the library.

**Depends on**: Feature 1 (Song Context) loosely.

---

### 5. Sample Pairing & Feedback Loop

**What**: A complete learning pipeline — the agent presents plausible sample pairs in conversation, the user gives yes/no verdicts, the system computes relational audio features on each pair, periodically analyzes verdict patterns against features, extracts heuristic rules, and feeds those rules back into recommendation logic.

**Why**: This is how SampleSpace gets smarter over time. Every other feature makes static recommendations. The flywheel makes recommendations that improve with use. Every verdict is a labeled training example that can't be acquired any other way.

**Depends on**: Features 2 (Pair Scoring), 3 (Audio Transformation), and optionally 4 (Sample Upload).

#### The Flywheel

```
Stage 1: Pair Generation  [IMPLEMENTED]
  Agent selects plausible pairs using pair scoring (Feature 2)
  Biased toward: compatible keys, complementary types, CNN neighborhoods
  NOT random — the signal is in users rejecting "should-work" pairs

         |

Stage 2: Verdict Collection  [IMPLEMENTED]
  Agent presents pair in chat with side-by-side playback
  User responds: yes / no (+ optional free-text reason)
  Stored in pair_verdicts table

         |

Stage 3: Feature Extraction (async, background)  [IMPLEMENTED]
  Compute relational audio features for the pair:

  | Feature             | Computation                                              | Signal                                      |
  |---------------------|----------------------------------------------------------|---------------------------------------------|
  | Spectral overlap    | Magnitude spectrograms -> normalize -> IoU across freq bins | Do they compete for the same frequencies?   |
  | Onset alignment     | Detect onsets -> cross-correlate onset vectors             | Do transients collide or interleave?        |
  | Timbral contrast    | MFCCs -> cosine distance of mean vectors                  | How different is their timbral character?   |
  | Harmonic consonance | Chroma features -> correlate mean chroma vectors           | Is their harmonic content consonant?        |
  | Spectral centroid gap | Spectral centroid difference                            | Do they occupy different frequency registers?|
  | RMS energy ratio    | RMS energy -> ratio                                       | Relative loudness balance                   |

  All computable with librosa. Key insight: these are *relational* features
  (computed on the pair), not individual features.

         |

Stage 4: Pattern Analysis  [DEFERRED — needs ~20+ verdicts per type pair]
  Aggregate verdicts + features
  Group by type pair, compute feature distributions for accepted vs rejected
  Statistical tests or simple threshold analysis:
    "For kick+bass pairs, accepted pairs have mean spectral overlap < 0.3,
     rejected pairs have mean spectral overlap > 0.6"

         |

Stage 5: Rule Extraction  [DEFERRED]
  Convert patterns into structured rules:
    PairRule(type_pair="kick+bass", feature="spectral_overlap",
             threshold=0.4, direction="below", confidence=0.75)
  Require minimum sample size (e.g., 20+ verdicts per type pair)
  Rules are versioned — new analysis creates new versions, old ones archived

         |

Stage 6: Rule Application  [DEFERRED — infrastructure ready]
  Two integration points:
  a) Agent system prompt: inject top rules as guidelines
     "When pairing kick+bass, prefer pairs with low spectral overlap"
  b) pair_scoring service: adjust weights/thresholds using learned rules
```

---

### 6. Kit Builder

**What**: The agent assembles a coherent multi-sample kit (kick + snare + hihat + bass + pad) given a vibe, genre, or reference description. Uses song context for key/BPM targeting, pair scoring for inter-sample compatibility, CNN diversity to avoid redundancy, CLAP relevance for vibe matching, and optionally learned rules from the flywheel.

**Why**: Instead of finding one sample at a time, the agent delivers a complete, production-ready set. This is the capstone — the moment SampleSpace goes from "tool" to "collaborator."

**Depends on**: Features 1, 2, and benefits from 3 and 5.
