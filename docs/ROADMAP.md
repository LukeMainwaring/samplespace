# Roadmap

Remaining features and improvements for SampleSpace. All original features (phases 1-3), the sample detail view, and Stage 4 preference model are complete. The next steps are active learning and preference-aware recommendations — see `docs/feature-brainstorm.md` for the full design.

## Upcoming Features

### Web-Based Sample Ingestion

Currently, samples are ingested from a local directory (`SAMPLE_LIBRARY_DIR`) via the `seed-samples` CLI script. This works well for personal use but doesn't scale to a hosted, multi-user setup.

Future direction:
- Web UI for uploading individual samples or bulk-importing from cloud storage (S3, Google Drive)
- Drag-and-drop ingestion with progress tracking and automatic metadata extraction
- Deferred until auth is implemented

## UI Features

### Demo GIF

Record a short (~15s) GIF for the README showing the core loop: ask a question in chat, agent calls tools, results appear.

- Use a screen recorder (e.g., Kap) at ~15 FPS, 1200px wide
- Capture: type query -> agent streams response with tool calls -> sample results in sidebar
- Optimize with gifsicle or convert to WebM with a GIF fallback
- Add to README under the project title

## ML Improvements

### Data Augmentation

Complete:

- **Speed perturbation** — ±5-10% via `torchaudio.functional.resample` with pre-defined small integer ratios (avoids the GCD kernel explosion of continuous rates). Always on during training.
- **Pitch perturbation** — ±1-2 semitones via resample with small integer ratios. Replaces the old `pitch_shift` (STFT + phase vocoder) which was too slow for live augmentation. The fixed-length spectrogram absorbs the duration change, leaving only the frequency shift.
- **Polarity inversion** — flip waveform sign (free, teaches phase invariance). Always on.
- **Mixup augmentation** — cross-class spectrogram blending with soft labels (default alpha 0.2). Uses a double forward pass: mixed spectrograms for CE loss, originals for SupCon.
- **Class-weighted sampling** — `WeightedRandomSampler` ensures equal class exposure per epoch. On by default.

Remaining:

- **DataLoader parallelization** — `num_workers > 0` with `forkserver` is now safe since all augmentations use fast resample (no `torch.stft`). Worth benchmarking on CUDA/Linux with `num_workers=4` and `prefetch_factor` tuning.
- **Offline precomputation** — less urgent now that live augmentation is fast (~17s/epoch for 944 samples). Could still be useful at >5,000 samples if data loading becomes the bottleneck again.

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

## Preference Learning

The pairing feedback loop collects verdicts and computes relational audio features. The preference learning system turns that data into a model that improves recommendations. See `docs/feature-brainstorm.md` for the full design and `docs/preference-learning-flow.md` for the data flow diagram.

- **Stage 4 — Preference Model** (complete): sklearn logistic regression trained on 10-dimensional feature vectors (4 pair score dimensions + 6 relational audio features). Auto-retrains every 5th verdict after 15 verdicts. `show_preferences` agent tool surfaces learned feature importances as natural-language explanations. Preferences are injected into the agent's system prompt.
- **Pair evaluation upgrades** (complete): CLAP-based context-aware retrieval (shared with kit builder), random anchor support for rapid pairing sessions, "Play Together" mixed audio preview, "Next Pair" button for fast verdict collection.
- **Stage 5 — Active Learning**: `present_pair` selects candidates where the model is most uncertain (P closest to 0.5), maximizing information gain per verdict.
- **Stage 6 — Preference-Aware Recommendations**: learned preferences feed into kit building (5th scoring dimension) and pair scoring (learned_preference dimension).
- **Confidence-Gated Automation**: after 30+ verdicts with 70%+ accuracy, the kit builder auto-approves high-confidence pairings and only asks about uncertain ones.

## Deferred

- **`swap_kit_sample` agent tool** — swap a single slot in an existing kit
- **`POST /pairs/verdict` and `POST /pairs/score` REST endpoints** — direct API access (currently agent-tool only)
- **Auth system** — prerequisite for web-based sample ingestion and multi-user support
- **Cross-session memory** — implicit preference learning from search/workflow patterns (exploratory, see brainstorm doc)
