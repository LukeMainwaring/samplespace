# Roadmap

Remaining features and improvements for SampleSpace. All features from the original brainstorm (phases 1-3) are complete — see `docs/feature-brainstorm.md` for design rationale.

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

Next steps:

- **Re-enable time stretch first** — cheaper than pitch shift (single STFT pass vs STFT + resample), teaches tempo invariance which matters more for this use case. Benchmark per-epoch time with `num_workers=0` to see if it's tolerable (~3-4 min/epoch estimated)
- **Offline precomputation** — a CLI script that generates pitch-shifted/time-stretched waveform variants as `.pt` files, loaded at training time instead of computed live. One-time cost (~10-20 min), ~1.7 GB disk for 2,000 samples x ~4 variants each. Avoids the STFT-in-worker deadlock entirely
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

Remaining directions to explore:

- Replace global average pooling with attention pooling — learn which time-frequency regions matter most
- Experiment with 1D convolutions on raw waveform (SampleCNN-style from the literature) as an alternative to mel spectrograms

## Deferred

### Flywheel Stages 4-6

The pairing feedback loop (Feature 5) has stages 1-3 live: pair presentation, verdict collection, and background relational feature extraction. Stages 4-6 are deferred until ~20+ verdicts are collected per type pair:

- **Stage 4 (Pattern analysis)**: aggregate verdicts + features, find statistical patterns per type pair (e.g., "accepted kick+bass pairs have mean spectral overlap < 0.3")
- **Stage 5 (Rule extraction)**: convert patterns into `PairRule` records with confidence scores. Requires a `services/pair_analysis.py` module and `uv run analyze-pairs` management command
- **Stage 6 (Rule application)**: `inject_pair_rules` system prompt decorator is already wired up — just needs rules in the DB. Also integrate learned rules into `pair_scoring` service weights

### Other Deferred Items

- **`swap_kit_sample` agent tool** — swap a single slot in an existing kit. Users can rebuild kits or use search tools conversationally as a workaround
- **`POST /pairs/verdict` REST endpoint** — direct API access to verdict recording (currently agent-tool only)
- **`POST /pairs/score` REST endpoint** — direct API access to pair scoring (currently agent-tool only)
- **Auth system** — prerequisite for web-based sample ingestion and multi-user support
