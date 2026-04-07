# Preference Learning & Active Feedback

Design document for SampleSpace's human-in-the-loop learning system — the features that make the assistant get smarter with use. See `docs/preference-learning-flow.md` for the data flow diagram.

## Stage 4: Preference Model (complete)

Sklearn `Pipeline(StandardScaler, LogisticRegression)` trained on 10-dimensional feature vectors (4 pair scores + 6 relational audio features). Auto-retrains every 5th verdict after 15. Saved to `backend/data/models/`. `show_preferences` agent tool surfaces learned feature importances as natural-language explanations. Preferences injected into agent system prompt via `inject_preferences()`. Implementation: `services/preference.py`, `agents/capabilities/pairing.py` (registration + instruction injection), `agents/tools/preference_tools.py` (tool implementations).

---

## Stage 5: Active Learning

**What**: When `present_pair` selects candidates for evaluation, score them through the preference model and pick the candidate where the model is most uncertain (predicted probability closest to 0.5).

**Why**: The current pair selection strategy picks candidates in the 0.5-0.8 heuristic score range — a reasonable proxy for "interesting," but it doesn't adapt. Active learning maximizes information gain per verdict: each evaluation teaches the model the most it can learn. The user reaches a good model with fewer interactions.

### How It Works

1. `present_pair` retrieves candidates via CLAP search (with song context) or CNN similarity (uses shared `candidate_search` module)
2. Score each candidate through pair scoring (existing behavior)
3. **New**: if a preference model exists, also score each candidate through the model
4. Select the candidate closest to P(accept) = 0.5 (maximum uncertainty), with a tiebreaker favoring the heuristic "interesting" range
5. If no model exists yet (< 15 verdicts), fall back to current heuristic selection

### Exploration vs Exploitation

Pure uncertainty sampling can get stuck in a narrow region of feature space. Add a small exploration term:
- 80% of the time: pick the most uncertain candidate (exploitation of model uncertainty)
- 20% of the time: pick a random candidate from the viable pool (exploration of feature space)

### Implementation

- Modify `present_pair` in `agents/tools/pair_tools.py` to load the preference model and score candidates
- Add `predict_acceptance(features: list[float]) -> float` to `services/preference.py`
- No new agent tools — this is invisible to the user, the pair selection just gets smarter

---

## Stage 6: Preference-Aware Recommendations

**What**: Feed learned preferences back into the system at three integration points: kit building, pair scoring, and the agent's system prompt.

**Why**: This is where the flywheel closes. Collected verdicts → trained model → better recommendations → user trusts the system more → gives more verdicts.

### Integration Point 1: Kit Builder

Add a 5th scoring dimension to `_fast_compatibility()` in `services/kit_builder.py`: the preference model's predicted acceptance probability. Weight it alongside existing dimensions during greedy assembly. When the model doesn't exist yet, this dimension is simply absent (existing dynamic weight rebalancing handles missing dimensions).

### Integration Point 2: Pair Scoring

Add an optional `learned_preference` dimension to `PairScore` in `services/pair_scoring.py`. When the model exists, include it as a 5th dimension with a configurable weight (start at 0.20, redistributed from the other four).

### Implementation

- Modify `services/kit_builder.py` `_fast_compatibility()` to query preference model
- Modify `services/pair_scoring.py` to include optional `learned_preference` dimension
- Add `predict_acceptance()` to `services/preference.py` if not already added in Stage 5

---

## Confidence-Gated Automation

**What**: When the preference model is highly confident about a pairing (P > 0.9 or P < 0.1), skip the verdict step and auto-approve/reject during kit building. Present verdicts only for uncertain cases.

**Why**: The system gradually needs less human input as it learns. After 50+ verdicts, the model is confident about most common type pairs and only asks about edge cases.

### Guardrails

- Only activate after 30+ verdicts and 70%+ model accuracy (held-out)
- Always allow user override: "Show me the pairs you auto-approved"
- Log auto-decisions for review (`auto_approved` flag on kit slot metadata)
- Agent communicates confidence: "I'm 94% sure this kick+bass pair works for you"

### Implementation

- Confidence thresholds in `services/preference.py` configuration
- Modify `services/kit_builder.py` to query the model during assembly
- Optional `explain_kit_decisions` agent tool

---

## Exploratory: Cross-Session Memory

The preference model learns **pairing taste** from explicit feedback. A complementary signal is **workflow preferences** from implicit behavior — patterns in searches, queries, and song context history.

### Why This Is Exploratory

1. The preference model already closes the learning loop. Implicit memory is a second system with unclear marginal return.
2. Attribution is harder — implicit signals are noisy.
3. The product story is cleaner with one flywheel.

### Possible Approaches

- **Mem0 integration** — managed memory layer for conversational preferences
- **Custom memory service** — track song context history, search patterns, verdict-adjacent signals
- **Hybrid** — Mem0 for conversational memory, custom model for audio-level taste

### When to Revisit

After the preference model is live and collecting data. The question: "What does the memory layer know that the preference model doesn't?"

---

## Implementation Order

```
[Stage 4] Preference Model      ✓ complete
        |
[Stage 5] Active Learning       ← next
        |
[Stage 6] Preference-Aware      ← kit builder + pair scoring
        |
Confidence-Gated Automation     ← after 30+ verdicts, 70%+ accuracy
        |
Cross-Session Memory             ← exploratory
```
