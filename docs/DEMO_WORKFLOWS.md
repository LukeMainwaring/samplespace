# Demo Workflows

Prompts and workflows that showcase what SampleSpace can do that general-purpose chatbots (ChatGPT, Claude) cannot. Organized by uniqueness and impressiveness.

---

## The Best Single Demo

> "I'm producing a funky tech house track in A minor at 126 BPM. Build me a kit with a drum loop, bass, and pad."

This one prompt triggers the full system. The agent calls `set_song_context` to persist key/BPM/genre/vibe, then `build_kit` to run CLAP retrieval for each sample type (enriched with "funky" vibe), greedy pairwise optimization with CNN diversity penalties, and multi-dimensional compatibility scoring — all rendered as an interactive kit card.

**What to watch for:**

- Tool call indicators appear with live spinner → green checkmark when complete
- Song context badge appears in the chat header: pills for "A minor", "126 BPM", "tech house", "funky"
- Kit block renders as a multi-slot grid — each slot shows the sample type label, a compatibility score, and a sample card with waveform visualization
- Kit score badge is color-coded: green (>=0.7), yellow (>=0.4), red (<0.4)
- Genre and vibe pills display above the kit slots
- Click any waveform to play the sample; scrub by clicking along the waveform
- Expand any tool call to see raw input/output JSON — shows CLAP queries, scoring breakdowns, and the greedy selection process

---

## Things No Other Chatbot Can Do

### Natural Language Audio Search

> "Find me a warm, breathy pad with a slow attack"

The agent encodes this text description into a 512-dim CLAP embedding and finds nearest audio neighbors via pgvector cosine similarity. "Breathy" and "slow attack" are sonic descriptors mapped to audio content — not keyword matching on filenames.

**What to watch for:**

- "Searching samples..." spinner → checkmark
- Results list with sample filenames, types, keys, BPMs, and IDs
- If song context is set, the query is automatically enriched with the vibe (expand the tool call to see the enriched query)

**Variations:**

- *"Shimmering, crystalline hi-hat with a tight decay"*
- *"Deep sub bass with analog warmth"*
- *"Glitchy, stuttered vocal chop"*

### Audio-to-Audio Similarity

> "Find samples that sound like `[sample_id]`"

Uses a custom-trained CNN on mel spectrograms to find spectrally similar samples. This is true audio-to-audio similarity — the CNN learns library-specific spectral features that CLAP's text-audio space can't capture.

**What to watch for:**

- "Finding similar samples..." spinner → checkmark
- Results are spectrally similar but may have different names or categories — the CNN sees frequency content, not metadata

**Why it matters:** Two search modalities that complement each other. CLAP bridges human language to audio. CNN bridges audio to audio. Together they cover queries that neither could handle alone.

### Upload a Reference Track

1. Click the **paperclip button** in the chat input
2. Select a WAV file — a preview chip appears with a loading spinner
3. Once the chip shows the filename with auto-analyzed metadata (type, key, BPM), type:

> "Find library samples that match this vibe"

The system generates a CLAP embedding for the upload and searches the library in the same embedding space — audio-to-audio cosine similarity against your own reference track.

**What to watch for:**

- Preview attachment chip transitions from loading spinner to complete state with metadata pills
- "Finding similar samples..." spinner → checkmark
- Results ranked by CLAP audio-to-audio similarity, excluding other uploads

### Interactive Pair Evaluation

> "Show me a pair to evaluate starting from `[sample_id]` — try matching it with a snare"

The agent finds candidates via CNN similarity (top 15), filters by the requested type, scores each candidate across key/BPM/type/spectral dimensions, and selects the candidate closest to a 0.6 score — plausible but not obvious, to make the evaluation interesting.

**What to watch for:**

- Pair-verdict block renders: two side-by-side sample cards, each with filename, metadata pills, and an interactive waveform
- Compatibility score displayed between the cards (e.g., "0.62/1.0")
- **"Works"** button (green, thumbs-up) and **"Doesn't work"** button (red, thumbs-down) appear below
- Clicking a verdict button updates the button state and auto-sends a message to the agent
- The agent calls `record_verdict`, which persists the verdict and triggers **background feature extraction** — computing 6 relational audio features (spectral overlap, onset alignment, timbral contrast, harmonic consonance, spectral centroid gap, RMS energy ratio)
- Agent confirms the verdict and reports the running total

**The feedback loop:** Verdicts accumulate into learned pairing preferences. After enough data, the system generates `PairRule` entries (e.g., "For kick-pad pairs, prefer spectral_distance > 0.5") that are injected into the agent's system prompt for future sessions.

### Pitch and Tempo Transformation

> "That pad sounds great but it's in the wrong key. Match `[sample_id]` to my song context."

The agent resolves the target key/BPM from the persisted song context, computes the semitone delta via circle-of-fifths logic, handles cross-mode transformations (major↔minor via relative keys), and runs pitch-shift/time-stretch.

**What to watch for:**

- "Transforming sample..." spinner → checkmark
- Response includes transformation details (e.g., "Shifted from E minor to D minor: -2 semitones")
- An **audio block** renders inline with a waveform player for the transformed audio — click to preview
- If the pitch shift is large (>5 semitones), a warning about potential artifacts appears
- Transformed audio is cached on the server — subsequent requests for the same transformation are instant

---

## End-to-End Workflows

### Full Production Session

A 6-step workflow demonstrating conversational memory and context-awareness across an entire session.

**Step 1 — Set the vibe:**

> "I'm working on a lo-fi hip hop beat in A minor at 85 BPM, warm and dusty vibes"

- Agent calls `set_song_context` with key=A minor, bpm=85, genre=lo-fi hip hop, vibe=warm and dusty
- Song context badge appears in the header with 4 pills

**Step 2 — Context-enriched search:**

> "Find me a mellow bass loop"

- Agent calls `search_by_description` — expand the tool call to see the query enriched with "warm and dusty" vibe automatically
- Results are influenced by the persisted vibe context

**Step 3 — Analyze and check compatibility:**

> "What key is `[bass_id]` in? Will it work with a pad in C major?"

- Agent calls `analyze_sample` then `check_key_compatibility`
- Two sequential tool calls, each with spinner → checkmark
- Key compatibility explains the circle-of-fifths distance and whether the keys are relative major/minor pairs

**Step 4 — Build a kit:**

> "Build me a kit — kick, snare, hihat, and a vinyl crackle texture"

- Agent calls `build_kit` with the song context from step 1 automatically applied
- Kit block renders with slots, compatibility scores, and genre/vibe badges
- Song context badge still visible in header from step 1

**Step 5 — Transform the kit:**

> "Transform the kit to match my song context"

- Agent calls `transform_kit` with the slots from step 4, resolving targets from song context (A minor, 85 BPM)
- Kit block re-renders with transformed audio URLs — each loop is pitch-shifted and/or time-stretched
- Response lists per-slot transforms (e.g., "bass: D minor → A minor (-5 semitones), 90 → 85 BPM")
- One-shots are included as-is (no transform needed)

**Step 6 — Preview the full kit:**

> "Let me hear the full kit together"

- Agent calls `preview_kit` with the transformed slots
- Audio block renders with a single mixed preview — all samples layered together
- Click to play the full kit as one track

**What to watch for across the session:** Song context persists through all 6 steps without repetition. Refresh the page mid-session — the context badge reappears because it's stored in the thread's JSONB column, not the browser session.

**Tip:** `match_to_context` is also available for transforming individual samples outside of a kit workflow.

### Sample Curation and Pair Training

A feedback loop: find samples, explore neighbors, evaluate pairs, build system knowledge. Includes rapid pairing mode for fast verdict collection.

**Step 1 — Find starting material:**

> "Find me aggressive, distorted kicks"

- Agent calls `search_by_description` — CLAP semantic search
- Results render as playable sample cards with waveforms

**Step 2 — Inspect in the detail view:**

Navigate to the Sample Library page and click the magnifying glass on the kick you found.

- Detail panel opens alongside the list with full metadata, waveform, and mel spectrogram
- Toggle to "CNN View" to see the exact 2-second, 128-mel-bin input the CNN processes
- Scroll down to "Similar Samples" — these are the CNN's nearest spectral neighbors with similarity percentages
- Play similar samples inline to audition them without leaving the panel

**Step 3 — Evaluate a pairing:**

> "Show me a pair to evaluate with `[kick_id]` — try matching it with a snare"

- Agent calls `present_pair` with candidate_type=snare
- Candidates are found via CLAP search enriched with song context (vibe, genre, key, BPM)
- Pair-verdict block renders with side-by-side cards and verdict buttons
- **Play Together** button layers both samples for audition as a mix
- Click "Works" or "Doesn't work"

**Step 4 — Rapid pairing mode:**

> "Start a pairing session with kicks and basses"

- Agent calls `present_pair` with anchor_type=kick, candidate_type=bass (no specific sample ID)
- A random kick is selected as anchor, bass candidate found via context-aware CLAP search
- After clicking a verdict, click **Next Pair** to immediately get another pair
- Each pair uses a new random anchor for diverse training data
- Repeat rapidly to build up verdicts — the preference model auto-trains after 15+

**Step 5 — Check what the system learned:**

After 15+ verdicts (mix of approvals and rejections):

> "What have you learned from my feedback?"

- Agent calls `show_preferences`
- Response includes a natural-language summary of learned feature importances
- Example: "You strongly prefer pairs with **distinct timbral character** (importance: 28%) and favor pairs that occupy **different frequency registers** (importance: 19%)"
- The preference model auto-trains in the background after every 5th verdict (starting at 15)
- Learned preferences are also injected into the agent's system prompt, so future recommendations are informed by your taste

### Reference Track Workflow

Start from your own music and work outward.

**Step 1 — Upload and analyze:**
Attach a WAV via the paperclip button, then:

> "What can you tell me about this track?"

- Agent calls `analyze_sample` on the uploaded sample
- Response shows detected key, BPM, duration, and type classification

**Step 2 — Find library matches:**

> "Find samples in the library that sound like my upload"

- Agent calls `find_similar_to_upload`
- Results ranked by CLAP audio-to-audio cosine similarity

**Step 3 — Set context and build:**

> "Set my song context to match this reference track and build me a full kit"

- Agent calls `set_song_context` with the reference track's detected key/BPM
- Then `build_kit` to assemble a kit informed by the reference
- Song context badge updates, kit block renders

---

## Quick Demos

**Semantic search:** *"Find a shimmering, crystalline hi-hat"* — CLAP text-to-audio search. Watch the spinner and results list.

**Key compatibility:** *"Are D minor and F major compatible?"* — circle-of-fifths check. Response explains they're relative major/minor pairs (highly compatible, score 0.95).

**Complement suggestion:** *"Suggest a bass that complements `[pad_id]`"* — CLAP search + key/BPM filtering. Results show key compatibility annotations (checkmarks for same/relative keys).

**Rate a pair:** *"How compatible are `[sample_a_id]` and `[sample_b_id]`?"* — multi-dimensional breakdown showing key, BPM, type complementarity, and spectral scores with a natural-language summary.

**Sample detail view:** Navigate to the Sample Library page, click the magnifying glass on any sample card. The list splits to reveal a detail panel with full metadata, interactive waveform, mel spectrogram (toggle between Full and CNN View to see what the model sees during inference), and CNN-similar samples ranked by similarity percentage. Play similar samples inline to audition them.

**Quick kit:** *"Build me a minimal techno kit — just kick, hihat, and bass"* — 3-slot kit block with fast rendering.

**Context-aware search:** *"I'm in G major at 140 BPM. Find me an uplifting lead"* — sets song context then searches with vibe enrichment, all in one turn.

**Transform a kit:** *"Transform the kit to match my song context"* — pitch-shifts and time-stretches all loops in the kit to the target key/BPM. One-shots pass through unchanged.

**Preview a kit:** *"Let me hear the full kit together"* — layers all kit samples into a single mixed audio preview for auditioning the full arrangement.

---

## Tips for Presenters

- **Start fresh:** Each workflow assumes a new chat thread (no prior song context) unless noted. Click the new chat button in the sidebar.
- **Sample IDs:** Replace `[sample_id]` placeholders with actual IDs from your library — every tool result includes sample IDs you can reference.
- **Expand tool calls:** Click the collapsible tool call indicators to show input parameters and raw output. This demonstrates the agent's reasoning and the multi-modal retrieval pipeline.
- **Audio playback:** Click waveforms to play samples; scrub by clicking along the waveform. Multiple samples can be played in sequence.
- **Pair verdicts:** The thumbs up/down buttons auto-send a message to the agent — you don't need to type anything after clicking.
- **Context persistence:** Refresh the page mid-session to show that song context survives page reloads (persisted in the thread's JSONB column, not browser state).
- **Dark mode:** Toggle the theme to show the full dark mode experience — waveforms, badges, and kit blocks all adapt.
