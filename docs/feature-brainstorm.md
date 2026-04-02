# Preference Learning & Active Feedback

Design document for SampleSpace's human-in-the-loop learning system — the features that make the assistant get smarter with use.

## Background

SampleSpace already has the data collection infrastructure for a learning flywheel:

- **Pair verdicts** — users evaluate sample pairs (thumbs up/down) via `present_pair` and `record_verdict` agent tools. Each verdict is stored with the full pair score snapshot.
- **Relational audio features** — 6 librosa-based features computed asynchronously after each verdict: spectral overlap, onset alignment, timbral contrast, harmonic consonance, spectral centroid gap, RMS energy ratio.
- **Pair scoring** — 4-dimensional heuristic scoring (key, BPM, type complementarity, CNN spectral distance) with dynamic weight rebalancing.
- **System prompt injection** — `inject_pair_rules` decorator is wired up and ready to surface learned knowledge to the agent.

What's missing is the learning itself: turning collected verdicts into a model that improves recommendations. The features below close that loop.

---

## Stage 4: Preference Model

**What**: After accumulating ~15-20 verdicts, train a lightweight binary classifier that predicts whether a user will accept or reject a sample pair.

**Why**: The current pair scoring uses fixed heuristic weights (key 0.30, type 0.25, spectral 0.25, BPM 0.20). These weights are reasonable defaults, but they don't reflect what this specific user actually cares about. Some producers prioritize timbral contrast; others care more about harmonic consonance. A learned model captures this.

### Input Features (10-dimensional)

From pair scoring (4):
- Key compatibility score (0-1)
- BPM compatibility score (0-1)
- Type complementarity score (0-1)
- Spectral similarity score (0-1)

From relational audio features (6):
- Spectral overlap
- Onset alignment
- Timbral contrast
- Harmonic consonance
- Spectral centroid gap
- RMS energy ratio

### Model

Logistic regression via scikit-learn. Deliberately simple:
- Interpretable coefficients → direct feature importance extraction
- Trains in milliseconds on 20-100 samples
- No overfitting risk at small sample sizes (unlike neural nets)
- Upgrade path to small MLP if/when verdict count reaches hundreds

### Training Trigger

Retrain when: `verdict_count >= 15 AND verdict_count % 5 == 0` (every 5th verdict after the initial 15). Store the trained model as a pickled sklearn object in `data/models/preference_model.pkl`. Keep previous versions for drift comparison.

### Implementation

- `services/preference.py` — model training, prediction, feature importance extraction
- `PreferenceModel` table or filesystem-based storage for versioned model artifacts
- `uv run train-preferences` management command for manual retraining
- Auto-retrain hook in `record_verdict` tool (background task, non-blocking)

---

## Stage 5: Active Learning

**What**: When `present_pair` selects candidates for evaluation, score them through the preference model and pick the candidate where the model is most uncertain (predicted probability closest to 0.5).

**Why**: The current pair selection strategy picks candidates in the 0.5-0.8 heuristic score range — a reasonable proxy for "interesting," but it doesn't adapt. Active learning maximizes information gain per verdict: each evaluation teaches the model the most it can learn. The user reaches a good model with fewer interactions.

### How It Works

1. `present_pair` retrieves CNN-similar candidates (existing behavior)
2. Score each candidate through pair scoring (existing behavior)
3. **New**: if a preference model exists, also score each candidate through the model
4. Select the candidate closest to P(accept) = 0.5 (maximum uncertainty), with a tiebreaker favoring the heuristic "interesting" range
5. If no model exists yet (< 15 verdicts), fall back to current heuristic selection

### Exploration vs Exploitation

Pure uncertainty sampling can get stuck in a narrow region of feature space. Add a small exploration term:
- 80% of the time: pick the most uncertain candidate (exploitation of model uncertainty)
- 20% of the time: pick a random candidate from the viable pool (exploration of feature space)

This ensures the model sees diverse pairs and doesn't overfit to one type combination.

### Implementation

- Modify `present_pair` in `agents/tools/verdict_tools.py` to load the preference model and score candidates
- Add `predict_acceptance(features: list[float]) -> float` to `services/preference.py`
- No new agent tools — this is invisible to the user, the pair selection just gets smarter

---

## Stage 6: Preference-Aware Recommendations

**What**: Feed learned preferences back into the system at three integration points: kit building, pair scoring, and the agent's system prompt.

**Why**: This is where the flywheel closes. Collected verdicts → trained model → better recommendations → user trusts the system more → gives more verdicts. Without this stage, the learning is academic — it never affects what the user sees.

### Integration Point 1: Kit Builder

Add a 5th scoring dimension to `_fast_compatibility()` in `services/kit_builder.py`: the preference model's predicted acceptance probability. Weight it alongside existing dimensions during greedy assembly.

When the model doesn't exist yet, this dimension is simply absent (existing dynamic weight rebalancing handles missing dimensions).

### Integration Point 2: Agent System Prompt

The `inject_pair_rules` decorator (already wired up) surfaces a natural-language preference summary:

```
Based on your feedback (47 verdicts):
- You strongly prefer pairs with high timbral contrast (importance: 0.34)
- Spectral overlap matters most for kick+bass pairs — you reject pairs that compete for low frequencies
- Onset alignment has low importance for your pairings (0.08)
- Your overall acceptance rate is 62%, suggesting you have selective but not extreme taste
```

This gives the agent conversational awareness of the user's preferences, allowing it to explain recommendations and anticipate reactions.

### Integration Point 3: Pair Scoring

Add an optional `learned_preference` dimension to `PairScore` in `services/pair_scoring.py`. When the model exists, include it as a 5th dimension with a configurable weight (start at 0.20, redistributed from the other four). The `rate_pair` tool then shows both the heuristic score and the learned prediction.

### Implementation

- Modify `services/kit_builder.py` `_fast_compatibility()` to query preference model
- Modify `inject_pair_rules` in agent capabilities to generate natural-language summaries from feature importances
- Modify `services/pair_scoring.py` to include optional `learned_preference` dimension in `PairScore`
- Add `explain_preferences()` to `services/preference.py` — returns structured feature importances + natural-language summary

---

## Confidence-Gated Automation

**What**: When the preference model is highly confident about a pairing (P > 0.9 or P < 0.1), skip the verdict step and auto-approve/reject during kit building. Present verdicts only for uncertain cases.

**Why**: The system gradually needs less human input as it learns. Early on, every pairing is a question. After 50+ verdicts, the model is confident about most common type pairs and only asks about edge cases. This is the tangible payoff of the flywheel — the assistant gets faster and more autonomous.

### Guardrails

- Only activate after a minimum verdict count (e.g., 30) and minimum model accuracy (e.g., 70% on held-out verdicts)
- Always allow the user to override: "Show me the pairs you auto-approved in that kit"
- Log auto-decisions for later review (new column on kit results or a lightweight audit table)
- Agent communicates confidence level: "I'm 94% sure this kick+bass pair works for you based on your feedback"

### Implementation

- Add confidence thresholds to `services/preference.py` configuration
- Modify `services/kit_builder.py` to query the model during assembly and skip uncertain pairings
- Add an `auto_approved` flag to kit slot metadata
- Agent tool `explain_kit_decisions` (optional) — shows which pairs were auto-approved vs heuristic-selected

---

## Explainable Preferences

**What**: Surface the model's learned feature importances to the user as natural-language explanations.

**Why**: "The system learned from your feedback" is a black box. "You care most about timbral contrast (34%) and spectral separation (28%) when pairing drums with bass" is actionable and builds trust. For a portfolio, explainability is the difference between "I trained a model" and "I built an interpretable learning system."

### What to Surface

- **Global feature importances**: which of the 10 input features matter most across all verdicts
- **Per-type-pair importances**: "For kick+bass, spectral overlap is 3x more important than for pad+lead"
- **Acceptance rate trends**: overall and per type pair, with trajectory (improving? stable?)
- **Model confidence distribution**: "The model is confident about 70% of pairings, uncertain about 30%"

### Agent Integration

New agent tool: `show_preferences` — returns a formatted summary of learned preferences. The agent can reference this proactively ("Based on what I've learned about your taste...") or when asked ("What have you learned from my feedback?").

### Implementation

- `show_preferences` agent tool in `agents/tools/preference_tools.py`
- `services/preference.py` `explain()` method — extracts coefficients from logistic regression, computes per-type-pair breakdowns, formats as structured data
- Frontend: render as a preferences card (optional — agent text response may be sufficient initially)

---

## Exploratory: Cross-Session Memory

The preference model learns **pairing taste** from explicit feedback (verdicts). A complementary signal is **workflow preferences** from implicit behavior — patterns in how the user searches, what they ask for, and how they describe their music.

Examples of implicit signals:
- "Luke usually works in minor keys at 120-130 BPM" (derived from song context history across threads)
- "Luke prefers dark, gritty kicks over clean ones" (derived from CLAP search patterns)
- "When Luke says 'warm', he means analog-sounding, not lo-fi" (semantic preference calibration from search results the user engages with vs ignores)

### Why This Is Exploratory

1. **The preference model already closes the learning loop.** Adding implicit memory is a second learning system — valuable, but the marginal return is unclear until the first one is working.
2. **Attribution is harder.** Verdicts are unambiguous labeled data. Implicit signals are noisy — did the user click that sample because they liked it, or because it was first in the list?
3. **The portfolio story is cleaner with one flywheel.** Active learning from pair feedback is a crisp narrative. Implicit memory is a compelling extension but shouldn't dilute the core pitch.

### Possible Approaches

- **Mem0 integration** — managed memory layer that extracts and retrieves user preferences from conversation history. Low implementation cost (API integration), but it's a black box and doesn't tie into the audio feature pipeline.
- **Custom memory service** — track song context history, search query patterns, and verdict-adjacent signals (which search results lead to verdicts). More work, but fully integrated with the existing data model.
- **Hybrid** — use Mem0 for conversational memory ("Luke mentioned he's producing for a film soundtrack") and the custom preference model for audio-level taste. Different learning signals, different systems.

### When to Revisit

After the preference model is live and collecting data. The right question then becomes: "What does the memory layer know that the preference model doesn't?" If the answer is "not much," skip it. If it's "the user's genre preferences and workflow patterns," build it.

---

## Implementation Order

```
Verdict Collection (existing)
        |
        v
Feature Extraction (existing)
        |
        v
[Stage 4] Preference Model  ←── first priority
        |                        ~15-20 verdicts needed
        v                        sklearn logistic regression
[Stage 5] Active Learning    ←── second priority
        |                        modify present_pair candidate selection
        v
[Stage 6] Preference-Aware  ←── third priority
        |  Recommendations       kit builder + system prompt + pair scoring
        v
Confidence-Gated Automation  ←── after 30+ verdicts, 70%+ accuracy
        |
        v
Explainable Preferences      ←── show_preferences agent tool
        |
        v
Cross-Session Memory          ←── exploratory, revisit after model is live
```

Each stage is independently valuable and demoable. Stage 4 alone (a working preference model) is a meaningful milestone. Stages 5-6 make it impressive. Confidence gating and explainability make it a portfolio centerpiece.
