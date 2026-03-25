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

- **Feature 1 (Song Context)** is the foundation (implemented). Every other feature is more useful when it can filter/rank against the user's active song context.
- **Feature 2 (Pair Scoring)** depends on 1 (implemented) — scoring is more meaningful when weighted by song context (e.g., BPM compatibility is moot if both samples will be time-stretched to the song's BPM).
- **Feature 3 (Audio Transformation)** depends on 1 — `match_to_context` needs a target key and BPM.
- **Feature 4 (Sample Upload)** has a soft dependency on 1 (implemented) — uploaded samples benefit from context-aware ranking.
- **Feature 5 (Pairing & Feedback)** depends on 2 and 3 — needs pair scoring to generate plausible pairs and benefits from transformed audio (users judge pairs that are already key/BPM-matched). Can also ingest uploaded samples from 4.
- **Feature 6 (Kit Builder)** depends on 5 (and transitively on 1, 2, 3) — uses pair scoring, learned rules, song context, and optionally transformation.

---

## Phased Build Order

### Phase 1: Context Foundation

**Features**: Song Context (1) ✅, Pair Compatibility Scoring (2) ✅

Both implemented. SampleSpace is now a context-aware assistant with multi-dimensional pair scoring. Phase 2 is next — Audio Transformation and Sample Upload can be built in parallel.

### Phase 2: Audio Pipeline

**Features**: Audio Transformation (3) ✅, Sample Upload (4) ✅

Both implemented. Phase 2 is complete. Phase 3 is next — Pairing & Feedback Loop and Kit Builder.

### Phase 3: Learning & Assembly

**Features**: Pairing & Feedback Loop (5), Kit Builder (6)

The most complex features, building on Phase 1 and 2 infrastructure. The flywheel (5) is a data collection + offline analysis pipeline. Kit Builder (6) is the capstone that combines everything. All prerequisite features (1-4) are now implemented.

---

## Feature Specs

### 1. Song Context (Thread-Backed) — Implemented

**What**: The user tells the agent about their song through natural conversation ("I'm working on a track in D minor at 120 BPM, kind of a dark techno vibe"). The agent persists this context to the thread record and automatically applies it when searching, scoring, and recommending. Context survives page refreshes and is unique per conversation thread.

**Why**: Today every search is stateless. The user has to repeat "in D minor at 120 BPM" with every request. Song context transforms the interaction from "search engine" to "production assistant that knows your song."

**Depends on**: Threads & messages (implemented). Song context is stored as a JSONB column on the existing `threads` table.

#### Implementation Summary

**Backend:**
- `SongContext` Pydantic schema in `schemas/thread.py` with four optional fields: `key`, `bpm`, `genre`, `vibe`. Used as the single type throughout the stack (AgentDeps, tools, routers) — no raw dicts.
- `song_context` JSONB column on `threads` table via Alembic migration. `Thread.update_song_context()` classmethod uses get-or-create pattern (works on first message) and `model_copy(update=exclude_unset)` for partial merges.
- `set_song_context` tool in `agents/tools/context_tools.py`. Persists to DB and updates `ctx.deps.song_context` in-place for same-turn visibility.
- Dynamic system prompt via `@sample_agent.system_prompt` decorator injects active context so the agent always knows the current state.
- `search_by_description` (`clap_tools.py`): appends song context vibe to CLAP query for semantic enrichment.
- `suggest_complement` (`analysis_tools.py`): uses song context vibe in CLAP query and falls back to song context key for compatibility filtering when the source sample has no key.
- Agent router loads song context from thread at request start and injects into `AgentDeps`.

**Frontend:**
- `SongContextBadge` component renders active context fields as pills in the chat header (read-only).
- `useThreadSongContext` hook shares the same TanStack Query cache as `useThreadMessages` (deduped fetch).
- `onFinish` invalidates the thread messages query to refresh the badge after `set_song_context` runs.
- Tool verb: "Updating song context" in tool call display.

#### Resolved Questions

- **Vibe + CLAP**: Vibe is appended to the CLAP query string (e.g., "warm pad" becomes "warm pad, dark and atmospheric"). Simple and effective.
- **Clearing context**: No dedicated `clear_song_context` tool. The agent can update individual fields; full clearing is not yet supported but could be added later.
- **Badge editability**: Display-only. Changes happen only through conversation, keeping the agent as the single mutation path.

---

### 2. Pair Compatibility Scoring — Implemented

**What**: A multi-dimensional compatibility score between two samples, combining key compatibility (circle of fifths distance), BPM compatibility, type complementarity (kick+hihat > kick+kick), and CNN embedding distance.

**Why**: The current `check_key_compatibility` only considers key. Real pairing decisions involve multiple dimensions. A composite score lets the agent rank candidate pairs and explain *why* two samples work together (or don't).

**Depends on**: Feature 1 (Song Context).

#### Implementation Summary

**Backend:**
- Extracted music theory logic (circle of fifths, relative pairs, key distance) from `analysis_tools.py` into `services/music_theory.py` for reuse across tools and services.
- `services/pair_scoring.py` with `score_pair(db, sample_a_id, sample_b_id) -> PairScore`. Computes four dimensions:
  - **Key score**: circle-of-fifths distance mapped to 0-1 via `music_theory_service.key_compatibility_score()`. Only for loops with known keys.
  - **BPM score**: ratio-based with integer-multiple normalization (halve/double into 60-180 range, so 60 and 120 BPM score as compatible). Only for loops with known BPMs.
  - **Type score**: hardcoded `TYPE_COMPLEMENTARITY` matrix using `frozenset` keys for symmetric lookup. High for kick+hihat (0.9), low for kick+kick (0.2), default 0.5 for unknown pairs.
  - **Spectral score**: CNN cosine distance with context-dependent interpretation — complementary types prefer spectral difference, same types prefer similarity. Uses explicit norm division for robustness.
- Dynamic weight rebalancing: when dimensions are unavailable (one-shots skip key/BPM, missing CNN embeddings skip spectral), weights redistribute proportionally. Default weights: key 0.30, type 0.25, spectral 0.25, BPM 0.20.
- `PairScore` and `DimensionScore` Pydantic schemas in `schemas/pair.py`. Each dimension includes value, effective weight, and explanation.
- `rate_pair` agent tool in `agents/tools/pair_tools.py` formats the score as readable markdown.
- Refactored `analysis_tools.py` to import from `music_theory_service` instead of inlining constants.

**Frontend:** None — score returned as text in agent response.

#### Resolved Questions

- **Weights**: Starting at key 0.30, type 0.25, spectral 0.25, BPM 0.20. Will tune based on usage.
- **Spectral score source**: CNN distance only. CLAP distance could be added later as a fifth dimension.
- **BPM integer multiples**: Yes — BPMs are normalized to 60-180 range before comparison.
- **REST endpoint**: Not added yet (agent tool only). Can add `POST /pairs/score` later when frontend needs direct access.

---

### 3. Real-time Audio Transformation — Implemented

**What**: An agent tool that pitch-shifts and/or time-stretches a sample to match a target key and BPM, returning a playable transformed audio file inline in the chat.

**Why**: This was the core feature of the predecessor project. Finding a great pad in E minor is useless if the song is in G minor — unless the system can transpose it. This is the difference between "here are samples that might work" and "here's a sample that *does* work in your song."

**Depends on**: Feature 1 (Song Context).

#### Implementation Summary

**Backend:**
- `CHROMATIC_INDEX` constant and `semitone_delta()`, `compute_target_key()` functions added to `services/music_theory.py`. `compute_target_key()` handles cross-mode shifts via relative keys — a minor sample against a major song targets the relative minor (same key signature = maximum harmonic compatibility).
- `services/audio_transform.py`: `transform_sample()` applies `librosa.effects.pitch_shift` and/or `librosa.effects.time_stretch`, writing results to a filesystem cache (`data/transforms/`). Cache keys are deterministic: `{sample_id}_key-{sanitized_key}_bpm-{bpm}.wav`. Called via `asyncio.to_thread()` to avoid blocking the event loop.
- `TRANSFORM_CACHE_DIR` setting in `core/config.py`. No new DB tables — cache is filesystem-only.
- `GET /api/samples/{sample_id}/audio/transformed?key=...&bpm=...` endpoint serves cached files. Returns 404 on cache miss (the agent tool pre-warms the cache).
- `match_to_context` agent tool in `agents/tools/transform_tools.py`. Validates sample is a loop, resolves targets from song context, handles cross-mode key resolution, runs transformation, returns markdown with an `audio` code fence containing the playback URL.
- System prompt updated with tool documentation and proactive behavior guideline.

**Frontend:**
- `AudioBlock` component (`components/elements/audio-block.tsx`) — a Streamdown `CustomRenderer` for the `audio` code fence language. Renders `WaveformViz` inline in chat messages.
- `Response` component updated with Streamdown `plugins` prop to register the audio renderer.
- Tool verb: "Transforming sample" in tool call display.

#### Resolved Questions

- **One-shots**: Tool rejects one-shots — they have no reference key/BPM for transformation.
- **Large pitch shifts**: Transformation proceeds but the tool warns for shifts >5 semitones.
- **Proactive behavior**: Yes — the agent suggests transformation when it finds a key/BPM mismatch with song context.
- **Cross-mode shifts**: Uses relative keys (e.g., minor sample + major song → targets relative minor). Grounded in music theory — relative keys share the same key signature.
- **Cache strategy**: Simple (keep all, no eviction). Can add LRU later.
- **Persistence**: Filesystem-only cache, not persisted to DB.

---

### 4. Sample Upload / "Find More Like This" — Implemented

**What**: The user uploads a WAV file (often a full song or snippet used as a reference track). The system analyzes it (key, BPM, duration, type), generates a CLAP embedding, and stores it permanently. The user can then ask the agent to find similar samples from the splice library using CLAP audio-to-audio similarity.

**Why**: Closes the loop between "I have audio I like" and "find me more." Previously users could only search by text or reference samples already in the library.

**Depends on**: Feature 1 (Song Context) loosely.

#### Implementation Summary

**Backend:**
- `POST /samples/upload` endpoint: accepts multipart WAV upload, validates (RIFF/WAVE header bytes, 50MB size limit, 60s duration limit), saves to `data/uploads/` with UUID filenames, runs `analyze_and_classify` for metadata, generates CLAP embedding via `asyncio.to_thread()`. Cleans up orphaned files on failure.
- Reuses existing `samples` table with `source="upload"` — no new tables or migrations. `find_audio_file()` extended with source-aware path resolution for uploads.
- `source` filter added to `Sample.get_all()` and `exclude_source` filter to `Sample.search_by_clap()` so the agent tool can search the library while excluding other uploads.
- `find_similar_to_upload` agent tool in `agents/tools/upload_tools.py`: fetches the uploaded sample's CLAP embedding and runs cosine distance search against the library.
- `UPLOAD_DIR` and `UPLOAD_MAX_SIZE_MB` settings in `core/config.py`.
- Shared `format_sample_results()` utility extracted to `agents/tools/formatting.py` (used by CLAP, CNN, and upload tools).

**Frontend:**
- Dedicated "Candidate Samples" page at `/candidates` with upload button, sample cards (metadata, playback via WaveformViz, copy-ID button), and empty state.
- Chat input file attachment: paperclip button triggers file picker, eager upload on file selection (Vercel chatbot template pattern), `PreviewAttachment` chip shows loading/complete state above textarea.
- Attachment state lifted to `chat.tsx` (parent owns `attachments/setAttachments`). On send, `[Uploaded sample: filename (ID: xxx)]` is prepended to the message text so the agent sees the reference.
- `useUploadSample` TanStack Query mutation hook wrapping the generated client.
- "Candidate Samples" nav link added to sidebar dropdown.

#### Resolved Questions

- **File size / duration**: 50MB / 60 seconds. Configurable via `UPLOAD_MAX_SIZE_MB`.
- **Temporary vs permanent**: All uploads are permanent. No ephemeral mode — simplifies the architecture.
- **Format support**: WAV only. RIFF/WAVE header bytes validated on upload.
- **CNN embeddings**: Out of scope — CLAP only. CNN embeddings can be added later when models are revisited.

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

Features 1-4 and 6 require no new tables beyond the existing `threads` and `messages` tables (already implemented). Song context is a JSONB column on the `threads` table. Pair scoring is computed on the fly. Transformed audio is filesystem cache. Uploads use the existing `samples` table with `source="upload"` and are stored in `data/uploads/`. Kits are ephemeral.

### New Agent Tools

| Tool | Phase | Feature | Status |
|------|-------|---------|--------|
| `set_song_context` | 1 | Song Context | ✅ Implemented |
| `rate_pair` | 1 | Pair Scoring | ✅ Implemented |
| `match_to_context` | 2 | Audio Transformation | ✅ Implemented |
| `find_similar_to_upload` | 2 | Sample Upload | ✅ Implemented |
| `present_pair` | 3 | Pairing & Feedback |
| `record_verdict` | 3 | Pairing & Feedback |
| `build_kit` | 3 | Kit Builder |
| `swap_kit_sample` | 3 | Kit Builder |

### New API Endpoints

| Endpoint | Method | Phase | Feature |
|----------|--------|-------|---------|
| `/samples/{id}/audio/transformed` | GET | 2 | Audio Transformation |
| `/samples/upload` | POST | 2 | Sample Upload | ✅ Implemented |
| `/pairs/verdict` | POST | 3 | Pairing & Feedback |
| `/pairs/score` | POST | 1 | Pair Scoring |

### New Service Modules

| Module | Phase | Purpose | Status |
|--------|-------|---------|--------|
| `services/pair_scoring.py` | 1 | Multi-dimensional pair compatibility scoring | ✅ Implemented |
| `services/music_theory.py` | 1 | Reusable key compatibility and circle-of-fifths logic | ✅ Implemented |
| `services/audio_transform.py` | 2 | Pitch-shift + time-stretch with caching | ✅ Implemented |
| `services/upload.py` | 2 | WAV upload pipeline with validation and CLAP embedding | ✅ Implemented |
| `services/pair_features.py` | 3 | Relational audio feature extraction (librosa) | |
| `services/pair_analysis.py` | 3 | Verdict pattern analysis + rule extraction | |
| `services/kit_builder.py` | 3 | Greedy kit assembly algorithm | |

### New Agent Tool Modules

| Module | Phase | Tools | Status |
|--------|-------|-------|--------|
| `agents/tools/context_tools.py` | 1 | `set_song_context` | ✅ Implemented |
| `agents/tools/pair_tools.py` | 1, 3 | `rate_pair`, `present_pair`, `record_verdict` | ✅ `rate_pair` implemented |
| `agents/tools/transform_tools.py` | 2 | `match_to_context` | ✅ Implemented |
| `agents/tools/upload_tools.py` | 2 | `find_similar_to_upload` | ✅ Implemented |
| `agents/tools/kit_tools.py` | 3 | `build_kit`, `swap_kit_sample` |

### New Settings

| Setting | Phase | Purpose |
|---------|-------|---------|
| `TRANSFORM_CACHE_DIR` | 2 | Cache directory for transformed audio files |
| `UPLOAD_DIR` | 2 | Permanent storage for uploaded files |
| `UPLOAD_MAX_SIZE_MB` | 2 | Upload size limit |

---

## Relationship to Existing Roadmap

The current `docs/ROADMAP.md` focuses on ML improvements (CNN training loss, augmentation, dataset scaling, CLAP model, CNN architecture) and UI features (sample detail view, demo GIF). These features are complementary:

- **CNN training loss** (contrastive/triplet loss) directly improves Features 2 and 5 — better CNN embeddings produce better pair scoring and more meaningful spectral similarity in pair features.
- **Dataset scaling** improves the CNN embedding space, which improves pair scoring, similarity search, and kit building.
- **CLAP text embedding cache** benefits Features 1 and 6 — song context vibe queries and kit builder vibe matching hit CLAP repeatedly.
- **Sample detail view** could display pair compatibility scores (Feature 2) and transformation options (Feature 3) alongside existing metadata.

None of the features in this document conflict with the existing roadmap. The ML improvements make these features work better; these features give the ML improvements a reason to exist beyond benchmarks.
