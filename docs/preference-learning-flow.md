# Preference Learning Flow

Data flow diagram for SampleSpace's preference learning system — from verdict collection through model training to agent integration.

```mermaid
flowchart TD
    subgraph Collection ["Data Collection (existing)"]
        A["present_pair\n(agent tool)"] -->|user clicks verdict| B["record_verdict\n(agent tool)"]
        B --> C[PairVerdict created]
        C --> D["Background task:\nextract pair features"]
        D --> E["PairVerdict.pair_features\n(6 relational audio features)"]
    end

    subgraph Training ["Model Training (Stage 4)"]
        E --> F{verdict_count >= 15\nAND count % 5 == 0?}
        F -->|yes| G[Fetch verdicts with features]
        G --> H["Build 10-dim feature vectors\n4 pair scores + 6 audio features"]
        H --> I["Train sklearn Pipeline\nStandardScaler + LogisticRegression"]
        I --> J[Evaluate via stratified k-fold CV]
        J --> K["Save to backend/data/models/\npreference_model.joblib\npreference_meta.json"]
    end

    subgraph Output ["Integration Points"]
        K --> L["inject_preferences()\nAgent system prompt enrichment"]
        K --> M["show_preferences tool\n'What have you learned\nfrom my feedback?'"]
        K -.->|"Stage 5 (future)"| N["Active learning\npresent_pair selects\nmost uncertain candidates"]
        K -.->|"Stage 6 (future)"| O["Preference-aware recs\nKit builder 5th dimension\nPair scoring integration"]
    end

    P["CLI: uv run train-preferences"] -->|manual trigger| G

    style Collection fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style Training fill:#16213e,stroke:#0f3460,color:#e0e0e0
    style Output fill:#0f3460,stroke:#533483,color:#e0e0e0
```

## Feature Vector (10 dimensions)

| # | Feature | Source | Range | Description |
|---|---------|--------|-------|-------------|
| 1 | key_score | pair_score_detail | [0, 1] | Key compatibility (circle of fifths) |
| 2 | bpm_score | pair_score_detail | [0, 1] | BPM compatibility (normalized) |
| 3 | type_score | pair_score_detail | [0, 1] | Type complementarity matrix |
| 4 | spectral_score | pair_score_detail | [0, 1] | CNN spectral distance/similarity |
| 5 | spectral_overlap | pair_features | [0, 1] | Frequency spectrum IoU |
| 6 | onset_alignment | pair_features | [0, 1] | Onset cross-correlation |
| 7 | timbral_contrast | pair_features | [0, 1] | MFCC cosine distance |
| 8 | harmonic_consonance | pair_features | [0, 1] | Chroma correlation |
| 9 | spectral_centroid_gap | pair_features | [0, 1] | Normalized centroid difference |
| 10 | rms_energy_ratio | pair_features | [0, 1] | Normalized log energy ratio |

Missing pair scores (e.g., key/BPM for one-shots) are imputed as 0.5 (neutral midpoint).
