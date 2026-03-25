# Roadmap

Remaining features and improvements for SampleSpace.

## Recently Completed

### Song Context (Thread-Backed)

Persistent per-thread song metadata (key, BPM, genre, vibe) that the agent reads, writes, and uses to contextualize searches. See `docs/feature-brainstorm.md` Feature 1 for full details.

- `set_song_context` agent tool with partial merge and get-or-create pattern
- Dynamic system prompt injection of active context
- CLAP search enriched with song context vibe; complement suggestions use context key as fallback
- Frontend `SongContextBadge` in chat header with automatic refresh via TanStack Query invalidation

### Pair Compatibility Scoring

Multi-dimensional compatibility score between two samples (key, BPM, type complementarity, CNN distance). See `docs/feature-brainstorm.md` Feature 2.

- `rate_pair` agent tool with `PairScore` schema (4 dimensions, dynamic weight rebalancing)
- `services/pair_scoring.py` with circle-of-fifths key scoring, BPM normalization, type complementarity matrix, context-dependent spectral interpretation
- `services/music_theory.py` extracted for reuse across tools and services

### Sample Pairing & Feedback Loop (Stages 1-3)

Interactive feedback loop where the agent presents sample pairs for evaluation and learns from user verdicts. See `docs/feature-brainstorm.md` Feature 5.

- `present_pair` agent tool: finds complementary candidates via CNN similarity, scores them via pair scoring, picks candidates in the "interesting" range (0.5-0.8) for maximum learning signal
- `record_verdict` agent tool: persists verdicts with pair score snapshot, fires background relational audio feature extraction
- `services/pair_features.py`: 6 librosa-based relational features (spectral overlap, onset alignment, timbral contrast, harmonic consonance, spectral centroid gap, RMS energy ratio)
- `pair_verdicts` and `pair_rules` database tables with Alembic migration
- Frontend `PairVerdictBlock` Streamdown renderer with side-by-side `WaveformViz` players and thumbs up/down buttons
- `ChatActionsProvider` React context threads `sendMessage` to nested Streamdown renderers
- Dynamic system prompt injection for learned pair rules (returns empty until stages 4-6 are built)
- Stages 4-6 (pattern analysis, rule extraction, rule application) deferred until ~20+ verdicts collected

## Upcoming Features

## UI Features

### Sample Detail View

Dedicated view showing full metadata, similar samples, and audio visualization for a single sample.

- Create a `/samples/[id]` dynamic route in the App Router
- Display: filename, type, key, BPM, duration, waveform, mel spectrogram image
- Show CNN-similar samples via `GET /api/samples/{id}/similar` (endpoint already exists)
- Link from sample browser cards to the detail view

### Demo GIF

Record a short (~15s) GIF for the README showing the core loop: ask a question in chat, agent calls tools, results appear.

- Use a screen recorder (e.g., Kap) at ~15 FPS, 1200px wide
- Capture: type query -> agent streams response with tool calls -> sample results in sidebar
- Optimize with gifsicle or convert to WebM with a GIF fallback
- Add to README under the project title

## ML Improvements

### CNN Training Loss

The embedding head is currently not optimized during training — only the classification head's cross-entropy loss drives backprop. The embedding space is shaped incidentally by the shared backbone, not by an explicit similarity objective.

- Add a contrastive or triplet loss on the embedding head alongside the existing classification loss
- Triplet loss (anchor, positive from same class, negative from different class) would directly optimize the embedding space for similarity search
- Weight the combined loss: `total = classification_loss + lambda * embedding_loss` (start with `lambda=0.5`)
- Fix the docstring in `train.py` which already claims this dual-loss setup but doesn't implement it

### Data Augmentation

Current augmentations (time masking, frequency masking, gain adjustment) operate on the mel spectrogram. More aggressive augmentation would help with the small dataset (~75 samples).

- Add pitch shifting and time stretching on the raw waveform (before mel conversion) using `torchaudio.functional`
- Add random cropping — select a random 2s window from longer samples instead of always taking the first 2s
- Add mixup augmentation — blend two samples from the same class to create synthetic training examples

### Dataset Scaling

The current dataset (75 samples, 11 classes) will overfit. The architecture and pipeline are the focus, but there are paths to better embeddings.

- [NSynth](https://magenta.tensorflow.org/datasets/nsynth) (300K samples) is the natural scaling path for instrument classification
- [FSD50K](https://zenodo.org/record/4060432) for broader audio event classification
- Even doubling the current set to ~150 samples with more variety per class would meaningfully improve the embedding space

### CLAP Model

Currently using `laion/clap-htsat-unfused`. Potential improvements:

- Evaluate `laion/larger_clap_music` — trained on music-specific data, may produce better embeddings for this domain
- Fine-tune the CLAP audio projection head on the sample library (freeze audio backbone, train only the projection layer) to adapt embeddings to this specific collection
- Cache CLAP embeddings at ingestion time (already done) but also cache text embeddings for repeated queries using an LRU cache in the embedding service

### CNN Architecture

The current 4-block CNN is straightforward. Some directions to explore:

- Add residual connections (skip connections) to each ConvBlock to improve gradient flow
- Replace global average pooling with attention pooling — learn which time-frequency regions matter most
- Experiment with 1D convolutions on raw waveform (SampleCNN-style from the literature) as an alternative to mel spectrograms
- Add a lightweight MLP projection head after the embedding layer (as in SimCLR) to separate the representation space from the similarity search space
