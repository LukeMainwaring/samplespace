# Demo Workflows

Prompts and workflows that showcase what SampleSpace can do that general-purpose chatbots (ChatGPT, Claude) cannot. Organized by uniqueness and impressiveness.

---

## The Best Single Demo

> "I'm producing a disco tech house track in A minor at 124 BPM. Build me a kit with a drum loop, bass, and pad."

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
- Results list with numbered indices (#1, #2, #3...), sample filenames, types, keys, and BPMs
- If song context is set, the query is automatically enriched with the vibe (expand the tool call to see the enriched query)
- Users can reference results naturally: "find more like #3" or "the second one sounds great"

**Variations:**

- *"Shimmering, crystalline hi-hat with a tight decay"*
- *"Deep sub bass with analog warmth"*
- *"Glitchy, stuttered vocal chop"*

### Audio-to-Audio Similarity

> "Find samples that sound like #2"

Uses a custom-trained CNN on mel spectrograms to find spectrally similar samples. This is true audio-to-audio similarity — the CNN learns library-specific spectral features that CLAP's text-audio space can't capture. Reference any sample from a previous search result by its number.

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

### Ad-hoc Pair Evaluation

A single, targeted evaluation where the user picks a specific sample and asks for a complementary match.

**Step 1 — Set the vibe:**

> "I'm making a minimalist house song with swaggy vibes in B Minor at 125 BPM"

**Step 2 — Find samples:**

> "Find me punchy, distorted kick one-shots"

**Step 3 — Explore neighbors:**

> "Find more samples that sound like #9"

**Step 4 — Evaluate a pairing:**

> "Show me a pair to evaluate — match kick #5 with a snare one-shot"

- Agent infers `is_loop=False` from "one-shot" and filters candidates accordingly
- Pair-verdict block renders: two side-by-side sample cards with waveforms and a mixed preview
- Scoring summary explains why the pair scored the way it did
- Click **"Works"** (green) or **"Doesn't work"** (red)
- Verdict is recorded; no automatic next pair (single evaluation)

**What to watch for:**

- Compatibility score and scoring summary displayed below the mixed preview
- Clicking a verdict renders a clean pill in the chat (not raw IDs)
- The agent calls `record_verdict`, which persists the verdict and triggers **background feature extraction** — computing 6 relational audio features

### Pairing Session — Rapid Training

A sustained session for fast verdict collection. The agent auto-presents the next pair after each verdict.

**Step 1 — Set the vibe:**

> "I'm making a progressive house song with retro vibes in G Major at 128 BPM"

**Step 2 — Start a session:**

> "Start a pairing session with kick loops and bass loops"

- Agent sets `is_loop=True`, filtering to loops only
- Random kick anchor selected, bass candidate found via context-aware CLAP search
- Pair-verdict block renders with side-by-side cards and mixed preview

**Step 3 — Rapid verdicts:**

- Click **"Works"** or **"Doesn't work"** → agent records the verdict and immediately presents the next pair
- Each pair uses a new random anchor for diverse training data
- Repeat rapidly to build up verdicts — the preference model auto-trains after 15+

**Step 4 — Check what the system learned:**

After 15+ verdicts (mix of approvals and rejections):

> "What have you learned from my feedback?"

- Agent calls `show_preferences`
- Response includes a natural-language summary of learned feature importances
- Example: "You strongly prefer pairs with **distinct timbral character** (importance: 28%) and favor pairs that occupy **different frequency registers** (importance: 19%)"
- The preference model auto-trains in the background after every 5th verdict (starting at 15)
- Learned preferences are also injected into the agent's system prompt, so future recommendations are informed by your taste

### Pitch and Tempo Transformation

After searching for samples:

> "That pad sounds great but it's in the wrong key. Transform #4 to match my song context."

The agent resolves the target key/BPM from the persisted song context, computes the semitone delta via circle-of-fifths logic, handles cross-mode transformations (major↔minor via relative keys), and runs pitch-shift/time-stretch via [Rubber Band](https://breakfastquay.com/rubberband/) R3 (the highest-quality engine, invoked directly as a subprocess). Percussive types (kick, snare, hihat, clap, cymbal, percussion, drum, fx) skip pitch-shifting — only BPM time-stretching is applied — since pitch-shifting degrades transient-heavy content.

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

> "I'm working on a hip hop house beat in D minor at 120 BPM, warm and dusty vibes"

- Agent calls `set_song_context` with key=D minor, bpm=120, genre=hip hop house, vibe=warm and dusty
- Song context badge appears in the header with 4 pills

**Step 2 — Context-enriched search:**

> "Find me a groovy bass loop"

- Agent calls `search_by_description` — expand the tool call to see the query enriched with "warm and dusty" vibe automatically
- Results are influenced by the persisted vibe context
- Each result is numbered (#1, #2, #3...) for easy reference

**Step 3 — Build a kit:**

> "Build me a kit with bass loop #10, a drum loop, and a guitar loop"

- Agent calls `build_kit` with the song context from step 1 automatically applied
- Kit block renders with slots, compatibility scores, and genre/vibe badges
- Song context badge still visible in header from step 1

**Step 4 — Transform the kit:**

> "Transform the kit to match my song context"

- Agent calls `transform_kit` with the slots from step 3, resolving targets from song context (D minor, 120 BPM)
- Kit block re-renders with transformed audio URLs — tonal loops are pitch-shifted and/or time-stretched; percussive loops (drums, hihats, etc.) are only time-stretched (pitch-shifting is skipped to preserve transient quality)
- Response lists per-slot transforms (e.g., "bass: D minor → A minor (-5 semitones), 90 → 85 BPM")
- One-shots are included as-is (no transform needed); percussive loops note "Pitch-shift skipped — percussive sample type."

**Step 5 — Preview the full kit:**

> "Let me hear the full kit together"

- Agent calls `preview_kit` — it automatically resolves transformed audio from the song context (no URL threading needed; the preview function looks up cached transforms directly)
- Audio block renders with a single mixed preview — all samples layered together with crossfade at loop boundaries
- Click to play the full kit as one track

**What to watch for across the session:** Song context persists through all 5 steps without repetition. Refresh the page mid-session — the context badge reappears because it's stored in the thread's JSONB column, not the browser session.

**Tip:** `match_to_context` is also available for transforming individual samples outside of a kit workflow.

### Sample Curation and Pair Training

A feedback loop: find samples, explore neighbors, evaluate pairs, build system knowledge. Combines ad-hoc evaluation with rapid pairing sessions.

**Step 1 — Find starting material:**

> "Find me aggressive, distorted kicks"

- Agent calls `search_by_description` — CLAP semantic search
- Results render as numbered, playable sample cards with waveforms

**Step 2 — Explore neighbors:**

> "Find more samples that sound like #1"

- Agent resolves #1 from the previous results, then calls `find_similar_samples` — CNN audio-to-audio similarity
- New results are also numbered for continued referencing

**Step 3 — Inspect in the detail view:**

Navigate to the Sample Library page and click the magnifying glass on one of the kicks.

- Detail panel opens alongside the list with full metadata, waveform, and mel spectrogram
- Toggle to "CNN View" to see the exact 2-second, 128-mel-bin input the CNN processes
- Scroll down to "Similar Samples" — these are the CNN's nearest spectral neighbors with similarity percentages
- Play similar samples inline to audition them without leaving the panel

**Step 4 — Ad-hoc evaluation:**

> "Show me a pair to evaluate — match that kick with a snare one-shot"

- Agent calls `present_pair` with candidate_type=snare, is_loop=False
- Candidates are filtered to one-shots and found via CLAP search enriched with song context
- Pair-verdict block renders with side-by-side cards, mixed preview, and scoring summary
- Click "Works" or "Doesn't work" — verdict renders as a clean pill in the chat

**Step 5 — Start a pairing session:**

> "Start a pairing session with kick loops and bass loops"

- Agent calls `present_pair` with anchor_type=kick, candidate_type=bass, is_loop=True
- A random kick loop is selected as anchor, bass loop candidate found via context-aware CLAP search
- Click a verdict → agent records it and immediately presents the next pair
- Each pair uses a new random anchor for diverse training data
- Repeat rapidly to build up verdicts — the preference model auto-trains after 15+

**Step 6 — Check what the system learned:**

After 15+ verdicts (mix of approvals and rejections):

> "What have you learned from my feedback?"

- Agent calls `show_preferences`
- Response includes a natural-language summary of learned feature importances
- Example: "You strongly prefer pairs with **distinct timbral character** (importance: 28%) and favor pairs that occupy **different frequency registers** (importance: 19%)"
- Learned preferences are injected into the agent's system prompt, so future recommendations are informed by your taste

### Reference Track Workflow

Upload your own music, set the song context from it, find complementary samples, and preview them together.

**Step 1 — Upload a reference track:**
Upload a WAV via the **Candidate Samples** panel on the right side. A metadata dialog appears after upload — optionally correct the auto-detected key, BPM, and loop/one-shot classification, then save or skip.

**Step 2 — Find the upload in chat:**

> "Find my southern twang house upload"

- Agent calls `find_upload` — searches uploaded samples by filename
- A playable sample card renders inline with key, BPM, and waveform visualization

**Step 3 — Set song context from the upload:**

> "Set the song context from this track. Genre is 'house', vibe is 'southern rock'"

- Agent calls `set_context_from_upload` with the upload's ID, plus the user-provided genre and vibe
- Key and BPM are extracted from the upload's detected metadata; genre and vibe come from the user
- Song context badge appears in the chat header with pills for key, BPM, genre, and vibe

**Step 4 — Find complementary samples:**

> "Find me a bass loop that goes well with this reference track"

- Agent calls `search_by_description` — CLAP search enriched with song context vibe
- Results render as numbered, playable sample cards
- Expand the tool call to see the vibe-enriched query

**Step 5 — Preview a pair together:**

> "Preview bass loop #6 with my reference track"

- Agent calls `present_pair` — side-by-side sample cards with a "Play Together" mixed preview
- Compatibility score displayed between the cards

**Step 6 — Transform and combine:**

> "Transpose and time-match it to fit the track context, then present the pair together so I can hear it combined"

- Agent calls `match_to_context` to pitch-shift and time-stretch the bass loop to the song context key/BPM
- Then calls `preview_kit` to layer the transformed bass with the reference track
- A combined audio preview renders inline — click to hear both samples together

**What to watch for across the session:** The song context badge persists through all steps. Each tool call shows a spinner → checkmark. Expand any tool call to see raw input/output JSON. The workflow flows naturally from upload → context → search → preview → transform without the user needing to manage IDs or metadata manually.

---

## Quick Demos

**Semantic search:** *"Find a shimmering, crystalline hi-hat"* — CLAP text-to-audio search. Watch the spinner and results list.

**Analyze and check compatibility:** After a search, ask *"What key is #1 in? Will it work with a pad in C major?"* — agent resolves #1 from the results, calls `analyze_sample` then `check_key_compatibility`. Two sequential tool calls, each with spinner → checkmark. Key compatibility explains circle-of-fifths distance and relative major/minor relationships.

**Key compatibility:** *"Are D minor and F major compatible?"* — circle-of-fifths check. Response explains they're relative major/minor pairs (highly compatible, score 0.95).

**Complement by reference:** *"Suggest a bass that complements #3"* — after a search, reference any result by number. CLAP search + key/BPM filtering. Results show key compatibility annotations.

**Rate a pair:** *"How compatible are #1 and #5?"* — multi-dimensional breakdown showing key, BPM, type complementarity, and spectral scores with a natural-language summary.

**Sample detail view:** Navigate to the Sample Library page, click the magnifying glass on any sample card. The list splits to reveal a detail panel with full metadata, interactive waveform, mel spectrogram (toggle between Full and CNN View to see what the model sees during inference), and CNN-similar samples ranked by similarity percentage. Play similar samples inline to audition them.

**Quick kit:** *"Build me a minimal techno kit — just kick, hihat, and bass"* — 3-slot kit block with fast rendering.

**Context-aware search:** *"I'm in G major at 140 BPM. Find me an uplifting lead"* — sets song context then searches with vibe enrichment, all in one turn.

**Transform by reference:** *"Transform #2 to match my song context"* — pitch-shifts and/or time-stretches the sample to the target key/BPM. Percussive types skip pitch-shifting (only BPM-stretched). Listen to the result inline.

**Transform a kit:** *"Transform the kit to match my song context"* — pitch-shifts and time-stretches tonal loops in the kit to the target key/BPM. One-shots pass through unchanged; percussive loops are only time-stretched (pitch-shift skipped).

**Preview a kit:** *"Let me hear the full kit together"* — layers all kit samples into a single mixed audio preview for auditioning the full arrangement.

---

## Tips for Presenters

- **Start fresh:** Each workflow assumes a new chat thread (no prior song context) unless noted. Click the new chat button in the sidebar.
- **Reference by number:** Search results are numbered (#1, #2, #3...). Use these in follow-up prompts: "find more like #2", "transform #4 to match my context", "how compatible are #1 and #3?"
- **Reference by name:** You can also use filenames: "transform warm-pad.wav to match my context". The agent will look it up.
- **Expand tool calls:** Click the collapsible tool call indicators to show input parameters and raw output. This demonstrates the agent's reasoning and the multi-modal retrieval pipeline.
- **Audio playback:** Click waveforms to play samples; scrub by clicking along the waveform. Multiple samples can be played in sequence.
- **Pair verdicts:** The thumbs up/down buttons auto-send a verdict to the agent — you don't need to type anything. In pairing sessions, the next pair appears automatically after each verdict.
- **Context persistence:** Refresh the page mid-session to show that song context survives page reloads (persisted in the thread's JSONB column, not browser state).
- **Dark mode:** Toggle the theme to show the full dark mode experience — waveforms, badges, and kit blocks all adapt.
