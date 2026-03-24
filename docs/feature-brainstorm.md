# Feature Brainstorm

Ideas for evolving SampleSpace from a search-oriented sample assistant into a context-aware production tool that learns from its users.

## Origin

These features are inspired by reviewing the predecessor project ([music-sample-assistant](https://github.com/LukeMainwaring/music-sample-assistant)), which had three ideas that the current codebase doesn't fully carry forward:

1. **"Match to my song"** — the user provides their song's key and BPM, and the system finds compatible samples, then pitch-shifts and time-stretches them to fit. The current agent is stateless — every search starts from scratch.
2. **Pair-quality prediction** — a binary CNN classifier trained on hand-curated audio pairs ("do these two samples sound good together?"). The current CNN answers a different question ("do these samples sound alike?"). Both are useful, but the old framing is more directly actionable.
3. **Explicit song context** — the old UI had dedicated key/BPM inputs. The current agent-based approach is more flexible but loses persistent context about what the user is building.

The opportunity: keep the agent's flexibility for open-ended queries, but add structured workflows (song context, audio transformation, pair learning, kit building) that make SampleSpace feel like a production tool rather than a search interface.

## How to Read This Document

Each feature is scoped with: what it is, why it matters, dependencies, backend/frontend/data work, new agent tools, and open questions. Features are ordered by phase (build order). The cross-cutting section at the end summarizes all new tables, tools, endpoints, and services across features.

This is a brainstorm document — exploratory in tone but structured enough that the best ideas can be promoted into `docs/ROADMAP.md`.

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

- **Feature 1 (Song Context)** is the foundation. Depends on the threads infrastructure (now implemented). Every other feature is more useful when it can filter/rank against the user's active song context.
- **Feature 2 (Pair Scoring)** depends on 1 — scoring is more meaningful when weighted by song context (e.g., BPM compatibility is moot if both samples will be time-stretched to the song's BPM).
- **Feature 3 (Audio Transformation)** depends on 1 — `match_to_context` needs a target key and BPM.
- **Feature 4 (Sample Upload)** has a soft dependency on 1 — uploaded samples benefit from context-aware ranking, but could be built independently.
- **Feature 5 (Pairing & Feedback)** depends on 2 and 3 — needs pair scoring to generate plausible pairs and benefits from transformed audio (users judge pairs that are already key/BPM-matched). Can also ingest uploaded samples from 4.
- **Feature 6 (Kit Builder)** depends on 5 (and transitively on 1, 2, 3) — uses pair scoring, learned rules, song context, and optionally transformation.

---

## Phased Build Order

### Phase 1: Context Foundation

**Features**: Song Context (1), Pair Compatibility Scoring (2)

These transform SampleSpace from a stateless search engine into a context-aware assistant. Song context is the prerequisite for nearly everything else, and pair scoring gives the agent a new capability without heavy infrastructure. Both are backend-focused with minimal frontend changes.

### Phase 2: Audio Pipeline

**Features**: Audio Transformation (3), Sample Upload (4)

These add new audio I/O capabilities — transformation creates a processing pipeline (librosa pitch_shift + time_stretch) with cache management; upload adds an ingestion endpoint with on-the-fly embedding computation. Independent of each other, can be built in parallel.

### Phase 3: Learning & Assembly

**Features**: Pairing & Feedback Loop (5), Kit Builder (6)

The most complex features, building on Phase 1 and 2 infrastructure. The flywheel (5) is a data collection + offline analysis pipeline. Kit Builder (6) is the capstone that combines everything.

---

## Feature Specs

### 1. Song Context (Thread-Backed)

**What**: The user tells the agent about their song through natural conversation ("I'm working on a track in D minor at 120 BPM, kind of a dark techno vibe"). The agent persists this context to the thread record and automatically applies it when searching, scoring, and recommending. Context survives page refreshes and is unique per conversation thread.

**Why**: Today every search is stateless. The user has to repeat "in D minor at 120 BPM" with every request. Song context transforms the interaction from "search engine" to "production assistant that knows your song."

**Depends on**: Threads & messages (implemented). Song context is stored as a JSONB column on the existing `threads` table — just a migration to add the column.

#### Backend

**Data model — migration:**
- Add `song_context: JSONB | None` column to `threads` table via Alembic migration.

**Pydantic model:**
- Define `SongContext` in `schemas/song_context.py`: `key: str | None`, `bpm: int | None`, `genre: str | None`, `vibe: str | None`.
- Used for validation, serialization, and type safety throughout the stack.

**AgentDeps — add thread context:**
- Add `thread_id: str` and `song_context: SongContext | None` to `AgentDeps`.
- In `routers/agent.py`, after `build_run_input()` extracts `thread_id`, load the thread's `song_context` from DB and inject it into `AgentDeps`.
- This makes song context available to all agent tools via `ctx.deps.song_context` without extra DB queries per tool call.

**Agent tool — `set_song_context`:**
- New tool in `agents/tools/context_tools.py`.
- Signature: `set_song_context(ctx, key?, bpm?, genre?, vibe?) -> str`.
- Reads current context from DB (`Thread.get`), merges new values (partial update — only overwrites provided fields, preserves the rest), writes back to `thread.song_context`, flushes.
- Also updates `ctx.deps.song_context` in-place so subsequent tool calls in the same turn see the new context without a DB re-read.
- Returns confirmation string: "Song context updated: D minor, 120 BPM, dark techno."

**System prompt changes (`sample_agent.py`):**
- Add a dynamic system prompt (via `@sample_agent.system_prompt` decorator that reads from `ctx.deps`) that injects the active song context so the agent always knows the current state.
- Instruct the agent to: (a) detect when the user mentions song properties and call `set_song_context`, (b) consult song context when calling search/suggest tools to filter by key/BPM, (c) mention active context in responses.

**Search tool modifications:**
- `search_by_description` (`clap_tools.py`): if `ctx.deps.song_context` has a `vibe`, append it to the CLAP query for relevance. If it has `key`/`bpm`, post-filter or re-rank results for compatibility.
- `suggest_complement` (`analysis_tools.py`): use song context `key`/`bpm` as defaults when not explicitly specified by the user.

**Thread model changes (`models/thread.py`):**
- Add `song_context` JSONB column.
- Add `update_song_context(db, thread_id, agent_type, context_data)` classmethod.

#### Frontend

- Display a "Song Context" badge in the chat header showing active key/BPM/genre when set.
- Load song context alongside thread messages — add `song_context` to `ThreadMessagesResponse` schema, or expose via thread metadata.
- No need to pass context in the request body — the backend reads it from the thread record.
- When `set_song_context` tool call appears in the streamed response, refresh the badge by invalidating the thread query.

#### Data Model

- **Migration**: add `song_context JSONB` column to existing `threads` table (nullable, default NULL).
- No new tables.

#### Agent Tools

| Tool | Signature |
|------|-----------|
| `set_song_context` | `(key?, bpm?, genre?, vibe?) -> str` |

#### Open Questions

- How does "vibe" interact with CLAP search — append to query string, or use as a separate re-ranking signal?
- Should there be an explicit `clear_song_context` tool, or does the agent call `set_song_context` with all nulls?
- Should the frontend badge be directly editable (click to change key/BPM) or only through conversation?

---

### 2. Pair Compatibility Scoring

**What**: A multi-dimensional compatibility score between two samples, combining key compatibility (circle of fifths distance), BPM compatibility, type complementarity (kick+hihat > kick+kick), and CNN embedding distance.

**Why**: The current `check_key_compatibility` only considers key. Real pairing decisions involve multiple dimensions. A composite score lets the agent rank candidate pairs and explain *why* two samples work together (or don't).

**Depends on**: Feature 1 (Song Context).

#### Backend

- Create `services/pair_scoring.py` with `score_pair(sample_a, sample_b, song_context?) -> PairScore`.
- `PairScore` model:
  - `overall: float` (0-1 weighted composite)
  - `key_score: float` (circle of fifths distance, reuse logic from `analysis_tools.py`)
  - `bpm_score: float` (BPM difference ratio)
  - `type_score: float` (from a type complementarity matrix)
  - `spectral_score: float` (CNN cosine distance — inverted for complementary types, direct for similar)
  - `explanation: str` (human-readable summary)
- Type complementarity matrix: hardcoded dict. High scores for kick+hihat, bass+lead, pad+lead; low for kick+kick, bass+bass. Later overridable by learned rules from Feature 5.
- CNN spectral score interpretation: for *complementary* types (kick+pad), lower cosine similarity is better (should sound different); for *similar* types (kick+kick), higher similarity is better.
- Add agent tool `rate_pair(sample_a_id, sample_b_id)` calling `score_pair`, returning formatted multi-dimensional score.

#### Frontend

Minimal — score is returned as text in the agent's response. Later could render as a visual score card.

#### Data Model

None. Complementarity matrix is a hardcoded dict.

#### Agent Tools

| Tool | Signature |
|------|-----------|
| `rate_pair` | `(sample_a_id, sample_b_id) -> str` |

#### Open Questions

- Right weights for the composite score? Start equal and tune.
- Should `spectral_score` use CNN distance, CLAP distance, or both?
- Should BPM score account for integer multiples (120 and 60 BPM are compatible)?

---

### 3. Real-time Audio Transformation

**What**: An agent tool that pitch-shifts and/or time-stretches a sample to match a target key and BPM, returning a playable transformed audio file.

**Why**: This was the core feature of the predecessor project. Finding a great pad in E minor is useless if the song is in G minor — unless the system can transpose it. This is the difference between "here are samples that might work" and "here's a sample that *does* work in your song."

**Depends on**: Feature 1 (Song Context).

#### Backend

- Create `services/audio_transform.py`:
  - `pitch_shift(audio_path, from_key, to_key) -> Path` — `librosa.effects.pitch_shift` with semitone delta from key difference.
  - `time_stretch(audio_path, from_bpm, to_bpm) -> Path` — `librosa.effects.time_stretch` with rate = to_bpm / from_bpm.
  - `transform_sample(audio_path, from_key, to_key, from_bpm, to_bpm) -> Path` — applies both, writes to cache directory.
- Cache directory: `data/transformed/`. Naming convention: `{sample_id}_{target_key}_{target_bpm}.wav` for deduplication.
- Key-to-semitone conversion: parse key strings ("D minor" → D), compute chromatic index difference.
- Cache eviction: LRU by access time or max directory size.
- Add agent tool `match_to_context(sample_id, target_key?, target_bpm?)`:
  1. Read sample's current key/BPM from DB.
  2. Default target_key/target_bpm from song context if not provided.
  3. Call `transform_sample`.
  4. Return message with transformed file URL.
- Add endpoint `GET /samples/{sample_id}/audio/transformed?key=...&bpm=...` to serve transformed files.

#### Frontend

- Agent returns a URL for transformed audio. Chat panel renders an inline audio player for these URLs (similar to waveform-viz pattern).
- Alternatively, tool call result includes a transformed file identifier that the frontend plays through the existing audio endpoint pattern.

#### Data Model

- Add `TRANSFORM_CACHE_DIR` to `Settings`. No new DB tables — transformed files are ephemeral cache.

#### Agent Tools

| Tool | Signature |
|------|-----------|
| `match_to_context` | `(sample_id, target_key?, target_bpm?) -> str` |

#### Open Questions

- One-shots have no key/BPM. Pitch-shifting a kick by explicit request is valid, but auto-matching to song context doesn't apply.
- librosa pitch_shift introduces artifacts at large intervals (>4 semitones). Warn the user or cap the shift?
- Should the agent proactively offer transformation? ("This pad is in E minor but your song is in G minor — want me to transpose it?")
- Persist transformed audio to DB or keep as filesystem-only cache?

---

### 4. Sample Upload / "Find More Like This"

**What**: The user uploads a WAV/MP3 file. The system computes CLAP + CNN embeddings on the fly and uses them to search for similar or complementary samples in the library. Can be temporary (just for search) or permanent (added to library).

**Why**: Closes the loop between "I have audio I like" and "find me more." Currently users can only search by text or by referencing samples already in the library. The old project had a placeholder for this ("Upload current song section") but never shipped it.

**Depends on**: Feature 1 (Song Context) loosely.

#### Backend

- Add endpoint `POST /samples/upload`:
  1. Accept multipart file upload.
  2. Save to temp directory.
  3. Run `analyze_and_classify` for metadata.
  4. Run `embed_audio` for CLAP embedding (CLAP model from app state).
  5. Optionally run CNN inference for CNN embedding.
  6. Return metadata + temporary sample ID. Does not persist to DB by default.
- Add endpoint `POST /samples/upload/search` that takes the computed embeddings and runs `Sample.search_by_clap` / `Sample.find_similar_by_cnn`.
- `permanent` flag: if true, create a Sample record and save file to `data/samples/uploads/`.
- Agent integration: upload happens through REST (frontend), which returns a temp ID. User tells the agent "find more like the sample I just uploaded" and the agent uses the temp ID.
- Add agent tool `find_similar_to_upload(temp_sample_id)` that queries by the uploaded sample's precomputed embeddings.

#### Frontend

- Upload button/dropzone in the chat panel or sample browser.
- On upload, call `POST /samples/upload`, display analyzed metadata, store temp ID.
- User says "find more like this" in chat; frontend passes temp sample ID in message context.

#### Data Model

- Add `UPLOAD_DIR` and `UPLOAD_MAX_SIZE_MB` to Settings.
- Uploaded samples use existing `samples` table if permanent, otherwise ephemeral.

#### Agent Tools

| Tool | Signature |
|------|-----------|
| `find_similar_to_upload` | `(temp_sample_id) -> str` |

#### Open Questions

- Max file size? Duration limit?
- How long do temporary uploads live? Clean up on session end or TTL?
- MP3 support? librosa handles it, but CLAP expects 48kHz WAV internally.
- Should uploaded samples get CNN embeddings? Fast but requires model to be loaded.

---

### 5. Sample Pairing & Feedback Loop

**What**: A complete learning pipeline — the agent presents plausible sample pairs in conversation, the user gives yes/no verdicts, the system computes relational audio features on each pair, periodically analyzes verdict patterns against features, extracts heuristic rules, and feeds those rules back into recommendation logic.

**Why**: This is how SampleSpace gets smarter over time. Every other feature makes static recommendations. The flywheel makes recommendations that improve with use. Every verdict is a labeled training example that can't be acquired any other way.

**Depends on**: Features 2 (Pair Scoring), 3 (Audio Transformation), and optionally 4 (Sample Upload).

#### The Flywheel

```
Stage 1: Pair Generation
  Agent selects plausible pairs using pair scoring (Feature 2)
  Biased toward: compatible keys, complementary types, CNN neighborhoods
  NOT random — the signal is in users rejecting "should-work" pairs

         ↓

Stage 2: Verdict Collection
  Agent presents pair in chat with side-by-side playback
  User responds: yes / no (+ optional free-text reason)
  Stored in pair_verdicts table

         ↓

Stage 3: Feature Extraction (async, background)
  Compute relational audio features for the pair:

  | Feature             | Computation                                              | Signal                                      |
  |---------------------|----------------------------------------------------------|---------------------------------------------|
  | Spectral overlap    | Magnitude spectrograms → normalize → IoU across freq bins | Do they compete for the same frequencies?   |
  | Onset alignment     | Detect onsets → cross-correlate onset vectors             | Do transients collide or interleave?        |
  | Timbral contrast    | MFCCs → cosine distance of mean vectors                  | How different is their timbral character?   |
  | Harmonic consonance | Chroma features → correlate mean chroma vectors           | Is their harmonic content consonant?        |
  | Spectral centroid gap | Spectral centroid difference                            | Do they occupy different frequency registers?|
  | RMS energy ratio    | RMS energy → ratio                                       | Relative loudness balance                   |

  All computable with librosa. Key insight: these are *relational* features
  (computed on the pair), not individual features.

         ↓

Stage 4: Pattern Analysis (periodic job or management command)
  Aggregate verdicts + features
  Group by type pair, compute feature distributions for accepted vs rejected
  Statistical tests or simple threshold analysis:
    "For kick+bass pairs, accepted pairs have mean spectral overlap < 0.3,
     rejected pairs have mean spectral overlap > 0.6"

         ↓

Stage 5: Rule Extraction
  Convert patterns into structured rules:
    PairRule(type_pair="kick+bass", feature="spectral_overlap",
             threshold=0.4, direction="below", confidence=0.75)
  Require minimum sample size (e.g., 20+ verdicts per type pair)
  Rules are versioned — new analysis creates new versions, old ones archived

         ↓

Stage 6: Rule Application
  Two integration points:
  a) Agent system prompt: inject top rules as guidelines
     "When pairing kick+bass, prefer pairs with low spectral overlap"
  b) pair_scoring service: adjust weights/thresholds using learned rules
```

#### Backend

- Create `models/pair_verdict.py` with `PairVerdict` model.
- Create `models/pair_rule.py` with `PairRule` model.
- Create `services/pair_features.py` — relational audio feature extraction (librosa, runs as background task).
- Create `services/pair_analysis.py` — periodic analysis logic, rule extraction.
- Add agent tools:
  - `present_pair(sample_a_id, sample_b_id, reason?)` — formats pair for user evaluation with playback context.
  - `record_verdict(sample_a_id, sample_b_id, verdict, reason?)` — stores verdict, triggers background feature extraction.
- Add endpoint `POST /pairs/verdict` for direct API access.
- Add management command `uv run analyze-pairs` for Stages 4-5.
- Modify system prompt to include active rules (loaded from DB at request time).

#### Frontend

- Render pair presentations in chat with side-by-side audio players.
- Thumbs-up/thumbs-down buttons that send a message back through useChat (or call REST directly).
- Optionally: display active rules in a debug/settings panel for transparency.

#### Data Model

**New table `pair_verdicts`:**

| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| sample_a_id | str | FK → samples.id |
| sample_b_id | str | FK → samples.id |
| verdict | bool | true = works together |
| reason | str \| None | user's free-text explanation |
| song_context_key | str \| None | song context at time of verdict |
| song_context_bpm | int \| None | |
| pair_features | JSONB \| None | relational features, populated async |
| created_at | timestamp | |

**New table `pair_rules`:**

| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| type_pair | str | e.g., "kick+bass" |
| feature_name | str | e.g., "spectral_overlap" |
| threshold | float | |
| direction | str | "above" or "below" |
| confidence | float | 0-1, based on sample size + effect size |
| sample_count | int | how many verdicts informed this |
| version | int | |
| is_active | bool | |
| created_at | timestamp | |

#### Agent Tools

| Tool | Signature |
|------|-----------|
| `present_pair` | `(sample_a_id, sample_b_id, reason?) -> str` |
| `record_verdict` | `(sample_a_id, sample_b_id, verdict, reason?) -> str` |

#### Open Questions

- How many verdicts before rule extraction is meaningful? Probably 20-30 per type pair.
- Should the agent proactively present pairs ("Want to rate some pairs?") or only when asked?
- How to handle rule conflicts (one rule says accept, another says reject)? Weighted confidence?
- Should rules be per-user or global? Start global (small user base).
- Feature extraction should be async (background task) — it loads audio from disk and is CPU-bound.
- Pairing strategy matters: random pairs have a low "yes" base rate and feel tedious. Bias toward plausible pairs so the user is refining a boundary, not rejecting obvious mismatches.

---

### 6. Kit Builder

**What**: The agent assembles a coherent multi-sample kit (kick + snare + hihat + bass + pad) given a vibe, genre, or reference description. Uses song context for key/BPM targeting, pair scoring for inter-sample compatibility, CNN diversity to avoid redundancy, CLAP relevance for vibe matching, and optionally learned rules from the flywheel.

**Why**: Instead of finding one sample at a time, the agent delivers a complete, production-ready set. This is the capstone — the moment SampleSpace goes from "tool" to "collaborator."

**Depends on**: Features 1, 2, and benefits from 3 and 5.

#### Backend

- Create `services/kit_builder.py` with the assembly algorithm:
  1. **Define kit template**: list of required types (e.g., `["kick", "snare", "hihat", "bass", "pad"]`). User can customize.
  2. **Candidate retrieval**: for each slot, CLAP search filtered by type + song context key/BPM → top 10-20 candidates.
  3. **Greedy assembly**: pick best candidate for first slot, then for each subsequent slot, pick the candidate that maximizes pair compatibility with all already-selected samples while maintaining CNN diversity (penalize high CNN similarity to already-selected samples of the same type-category).
  4. **Score the kit**: sum of pairwise scores / number of pairs.
  5. **Optional**: generate a second "alternative" kit from different top candidates.
- Agent tools:
  - `build_kit(vibe?, genre?, types?)` — calls kit builder, returns assembled kit with per-sample and overall scores.
  - `swap_kit_sample(position, new_sample_id)` — iterative refinement ("replace the bass with something warmer").
- If Feature 3 is available, offer to transform all kit samples to the song's key/BPM.

#### Frontend

- Render kit results as a structured card with all samples, types, and pairwise compatibility indicators.
- Each sample has a play button (wavesurfer).
- "Play all" — sequential or layered playback.
- Swap buttons per slot to request alternatives.

#### Data Model

None initially. Kits are ephemeral (returned in chat). A `kits` table could store saved kits later.

#### Agent Tools

| Tool | Signature |
|------|-----------|
| `build_kit` | `(vibe?, genre?, types?) -> str` |
| `swap_kit_sample` | `(position, new_sample_id) -> str` |

#### Open Questions

- How to handle small libraries (<100 samples)? Relax constraints gracefully.
- Optimize for *diversity* (different-sounding samples) or *cohesion* (similar vibe across all)?
- Kit templates by genre? EDM = kick+snare+hihat+bass+lead, ambient = pad+texture+lead+fx.
- "Play all" UX — sequential playback or layered mixdown?

---

## Cross-Cutting Concerns

### New Database Tables

| Table | Phase | Feature |
|-------|-------|---------|
| `pair_verdicts` | 3 | Pairing & Feedback Loop |
| `pair_rules` | 3 | Pairing & Feedback Loop |

Features 1-4 and 6 require no new tables beyond the existing `threads` and `messages` tables (already implemented). Song context is a JSONB column on the `threads` table. Pair scoring is computed on the fly. Transformed audio is filesystem cache. Uploads use the existing `samples` table if permanent. Kits are ephemeral.

### New Agent Tools

| Tool | Phase | Feature |
|------|-------|---------|
| `set_song_context` | 1 | Song Context |
| `rate_pair` | 1 | Pair Scoring |
| `match_to_context` | 2 | Audio Transformation |
| `find_similar_to_upload` | 2 | Sample Upload |
| `present_pair` | 3 | Pairing & Feedback |
| `record_verdict` | 3 | Pairing & Feedback |
| `build_kit` | 3 | Kit Builder |
| `swap_kit_sample` | 3 | Kit Builder |

### New API Endpoints

| Endpoint | Method | Phase | Feature |
|----------|--------|-------|---------|
| `/samples/{id}/audio/transformed` | GET | 2 | Audio Transformation |
| `/samples/upload` | POST | 2 | Sample Upload |
| `/samples/upload/search` | POST | 2 | Sample Upload |
| `/pairs/verdict` | POST | 3 | Pairing & Feedback |
| `/pairs/score` | POST | 1 | Pair Scoring |

### New Service Modules

| Module | Phase | Purpose |
|--------|-------|---------|
| `services/pair_scoring.py` | 1 | Multi-dimensional pair compatibility scoring |
| `services/audio_transform.py` | 2 | Pitch-shift + time-stretch with caching |
| `services/pair_features.py` | 3 | Relational audio feature extraction (librosa) |
| `services/pair_analysis.py` | 3 | Verdict pattern analysis + rule extraction |
| `services/kit_builder.py` | 3 | Greedy kit assembly algorithm |

### New Agent Tool Modules

| Module | Phase | Tools |
|--------|-------|-------|
| `agents/tools/context_tools.py` | 1 | `set_song_context` |
| `agents/tools/pair_tools.py` | 1, 3 | `rate_pair`, `present_pair`, `record_verdict` |
| `agents/tools/transform_tools.py` | 2 | `match_to_context` |
| `agents/tools/upload_tools.py` | 2 | `find_similar_to_upload` |
| `agents/tools/kit_tools.py` | 3 | `build_kit`, `swap_kit_sample` |

### New Settings

| Setting | Phase | Purpose |
|---------|-------|---------|
| `TRANSFORM_CACHE_DIR` | 2 | Cache directory for transformed audio files |
| `UPLOAD_DIR` | 2 | Temporary storage for uploaded files |
| `UPLOAD_MAX_SIZE_MB` | 2 | Upload size limit |

---

## Relationship to Existing Roadmap

The current `docs/ROADMAP.md` focuses on ML improvements (CNN training loss, augmentation, dataset scaling, CLAP model, CNN architecture) and UI features (sample detail view, demo GIF). These features are complementary:

- **CNN training loss** (contrastive/triplet loss) directly improves Features 2 and 5 — better CNN embeddings produce better pair scoring and more meaningful spectral similarity in pair features.
- **Dataset scaling** improves the CNN embedding space, which improves pair scoring, similarity search, and kit building.
- **CLAP text embedding cache** benefits Features 1 and 6 — song context vibe queries and kit builder vibe matching hit CLAP repeatedly.
- **Sample detail view** could display pair compatibility scores (Feature 2) and transformation options (Feature 3) alongside existing metadata.

None of the features in this document conflict with the existing roadmap. The ML improvements make these features work better; these features give the ML improvements a reason to exist beyond benchmarks.
