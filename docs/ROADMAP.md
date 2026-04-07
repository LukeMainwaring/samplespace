# Roadmap

Planned features and improvements. See `docs/feature-brainstorm.md` for detailed designs of the preference learning stages.

## Next Up

### Active Learning

`present_pair` scores candidates through the preference model and picks the one where the model is most uncertain (P closest to 0.5), maximizing information gain per verdict. Falls back to heuristic selection when < 15 verdicts. See `docs/feature-brainstorm.md` for exploration/exploitation strategy.

### Preference-Aware Recommendations

Learned preferences feed into kit building (5th scoring dimension) and pair scoring (`learned_preference` dimension). System prompt injection is already implemented via `inject_preferences()`. Closes the flywheel: verdicts → model → better recommendations → more verdicts.

## Planned

- **Confidence-gated automation** — after 30+ verdicts with 70%+ accuracy, auto-approve high-confidence pairings during kit building, only ask about uncertain ones
- **Web-based sample ingestion** — upload UI for individual samples or bulk import from cloud storage (deferred until auth)
- **Demo GIF** — ~15s screen recording of core loop for README
- **DataLoader parallelization** — `num_workers > 0` with `forkserver`, benchmark on CUDA/Linux
- **CLAP improvements** — fine-tune projection head on sample library (freeze audio backbone), evaluate `laion/larger_clap_music`, LRU cache for repeated text queries
- **CNN architecture** — attention pooling, multi-resolution spectrograms (multiple FFT windows), pre-trained audio feature extractors (BEATs, AST) as backbone alternatives, SampleCNN-style 1D convolutions on raw waveform
- **Dataset scaling** — explore external datasets (NSynth, FSD50K) and offline augmentation precomputation for larger scale

## Deferred

- Auth system — prerequisite for web-based ingestion and multi-user support
- Cross-session memory — implicit preference learning from search/workflow patterns (see brainstorm doc)
