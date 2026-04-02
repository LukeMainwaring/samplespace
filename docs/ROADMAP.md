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

### CNN Training & Architecture Overhaul

Modernized the CNN to produce properly trained embeddings and use current best practices.

- **Wider architecture**: Channel progression 1→64→128→256→512 (was 1→32→64→128→256), ~5.3M parameters. Residual connections with SE attention at each block.
- **2-layer projection head**: SimCLR-style projection (512→256→128 with ReLU + BatchNorm) replaces single linear layer. Improves contrastive embedding quality.
- **Supervised Contrastive Loss (SupCon)**: Combined with cross-entropy: `total = cls_loss + 0.5 * supcon_loss` (Khosla et al., NeurIPS 2020).
- **Cosine annealing with linear warmup**: 5-epoch warmup from 1% of target LR, then cosine decay to 1e-6. Replaces ReduceLROnPlateau.
- **Mixed precision training**: `torch.amp.autocast` + `GradScaler` on CUDA. Disabled on MPS (limited dtype coverage).
- **Gradient accumulation**: `--grad-accum` flag to simulate larger effective batch sizes for SupCon.
- **Early stopping**: Configurable patience (default 15 epochs). Prevents overfitting on small datasets.
- **MPS device support**: Auto-detects Apple Silicon for local training on MacBook.
- **TensorBoard logging**: Loss curves, LR schedule, per-class F1 scores, and embedding projector visualization (t-SNE/UMAP of 128-dim space). Logs to `data/runs/`.
- **Augmentation pipeline**: Pitch shift ±2 semitones, time stretch 0.9-1.1x, Gaussian noise injection (10-30 dB SNR), random EQ (±6 dB via equalizer biquad), SpecAugment (time/freq masking), random gain ±5 dB.
- **Defaults**: 100 epochs, batch size 64, AdamW (weight decay 1e-4), grad clipping (max_norm=1.0).
- **Batch inference**: `predict_batch()` for efficient multi-sample embedding generation.
- **pgvector HNSW indexes**: Added HNSW indexes on both `clap_embedding` and `cnn_embedding` columns for efficient cosine similarity search.

### Kit Builder

Assembles a complete multi-sample kit (kick + snare + hihat + bass + pad) using greedy pairwise optimization. See `docs/feature-brainstorm.md` Feature 6.

- `build_kit` agent tool with CLAP retrieval per type, fast inline compatibility scoring, CNN diversity penalty
- `services/kit_builder.py` with 3-phase greedy assembly (candidate retrieval, constrained selection, final pairwise scoring)
- Frontend `KitBlock` Streamdown renderer with per-slot playback and compatibility scores

## Upcoming Features

### Web-Based Sample Ingestion

Currently, samples are ingested from a local directory (`SAMPLE_LIBRARY_DIR`) via the `seed-samples` CLI script. This works well for personal use but doesn't scale to a hosted, multi-user setup.

Future direction:
- Web UI for uploading individual samples or bulk-importing from cloud storage (S3, Google Drive)
- Drag-and-drop ingestion with progress tracking and automatic metadata extraction
- Deferred until auth is implemented

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

### Data Augmentation

Active augmentations: Gaussian noise injection, random EQ, random crop (waveform-level), plus SpecAugment masking and random gain (spectrogram-level). Pitch shift (±2 semitones) and time stretch (0.9-1.1x) are implemented but disabled by default (`expensive_augment=False`) — they use `torch.stft` which deadlocks in macOS multiprocessing workers and adds ~10 min/epoch in single-threaded mode.

Next steps:

- **Re-enable time stretch first** — cheaper than pitch shift (single STFT pass vs STFT + resample), teaches tempo invariance which matters more for this use case. Benchmark per-epoch time with `num_workers=0` to see if it's tolerable (~3-4 min/epoch estimated)
- **Offline precomputation** — a CLI script that generates pitch-shifted/time-stretched waveform variants as `.pt` files, loaded at training time instead of computed live. One-time cost (~10-20 min), ~1.7 GB disk for 2,000 samples × ~4 variants each. Avoids the STFT-in-worker deadlock entirely
- **DataLoader parallelization** — `num_workers > 0` works on macOS only with cheap augmentations (no `torch.stft`). On CUDA/Linux, `num_workers=4` with all augmentations should work. Could also explore `torch.utils.data.DataLoader` with `prefetch_factor` tuning
- **Mixup augmentation** — blend two spectrograms from the same class to create synthetic training examples (operates on spectrograms, so no STFT issue)
- **Polarity inversion** — trivially flip waveform sign (free, teaches phase invariance)

### Dataset Scaling

Scaling to ~2,000 Splice samples across 15-16 classes. At this scale, SupCon loss becomes effective (enough same-class samples per batch) and the wider 512-channel architecture is justified. Further paths:

- [NSynth](https://magenta.tensorflow.org/datasets/nsynth) (300K samples) for instrument classification
- [FSD50K](https://zenodo.org/record/4060432) for broader audio event classification

### CLAP Model

Currently using `laion/clap-htsat-unfused`. Potential improvements:

- Evaluate `laion/larger_clap_music` — trained on music-specific data, may produce better embeddings for this domain
- Fine-tune the CLAP audio projection head on the sample library (freeze audio backbone, train only the projection layer) to adapt embeddings to this specific collection
- Cache CLAP embeddings at ingestion time (already done) but also cache text embeddings for repeated queries using an LRU cache in the embedding service

### CNN Architecture

Wider 512-channel backbone with residual connections, SE attention, and 2-layer SimCLR-style projection head are now implemented. Remaining directions to explore:

- Replace global average pooling with attention pooling — learn which time-frequency regions matter most
- Experiment with 1D convolutions on raw waveform (SampleCNN-style from the literature) as an alternative to mel spectrograms
