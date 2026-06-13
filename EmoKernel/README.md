# EmoKernel

Emotion analysis kernel with self-learning feedback loop and web-aware fallback for unknown expressions.

A Codex skill that doesn't just match keywords -- it understands emotion through a layered engine, learns from corrections, and searches the web when it encounters expressions it doesn't recognize.

---

## Architecture

`
User text
  |
  +-> EmotionAnalyzer          # Three-tier classification pipeline
  |     +-> sentence-embedding  # (when model available)
  |     +-> keyword + n-gram    # vote ensemble
  |     +-> web_lookup           # fallback: builtin slang -> multi-angle search -> character guess
  |
  +-> EmotionTree              # 7-root emotion taxonomy with automatic growth
  +-> MemoryManager            # Dual-layer: session queue (20 rounds) + persistent profile
  +-> SelfLearningManager      # Correction patterns, word promotion, version tracking
  |
  +-> EmotionEngine            # CLI entry point, coordinates all modules
`

## Features

- **7-emotion tree** -- joy, anger, sorrow, fear, love, disgust, desire -- each with 5 children. Unknown emotions trigger automatic branch growth.
- **Self-learning feedback loop** -- tell it "I'm not angry, I'm disappointed" once; it adjusts future analyses. Corrections are version-tracked.
- **Web-aware fallback** -- when keyword matching is uncertain, extracts unknown terms and searches the web (Tavily API). Multiple query angles are aggregated for robust results.
- **No-API fallback** -- even without Tavily, uses character-level emotion heuristics.
- **Dual-layer memory** -- short-term (session context, last 20 rounds) + long-term (persistent user profile with Welford statistics).
- **Psychology analysis** -- 11 cognitive distortion patterns (CBT), 4 attachment styles, defense mechanism classification.
- **Emotion smoothing** -- AI emotional state transitions smoothly with a dynamic alpha parameter, not abrupt jumps.
- **Graceful degradation** -- sentence-transformers model is optional. Falls back through keyword -> n-gram -> web lookup -> character guess without crashing.

## Quick Start

### 1. Install as a Codex skill

`ash
# Clone or copy EmoKernel into your Codex skills directory
# Then restart Codex
`

### 2. Run setup (optional)

`ash
python scripts/setup.py
`

The skill works without this step -- it falls back to keyword + web lookup automatically.

### 3. Configure web lookup (optional)

For web search capabilities, set a Tavily API key:

`ash
export TAVILY_API_KEY="your-api-key"
`

Without this key, the skill uses built-in slang + character-level heuristics.

## CLI Commands

| Command | Description |
|---------|-------------|
| process --text "..." | Analyze emotion and generate AI state (dialogue mode) |
| nalyze --text "..."  | Pure emotion analysis, no state change |
| psych --text "..."    | Psychology-level analysis (CBT distortions, attachment) |
| eedback --text "..." --detected "{}" --corrected "{}" | User correction feedback |
| slang --term "..."    | Look up an internet slang term |
| 	ree                  | View current emotion tree structure |
| 
eport                | Self-learning report |
| export                | Export learned knowledge |
| ersion               | Self-learning version |

## Mode Switching

The skill supports two modes controlled by user intent:

- **Dialogue Filter (default)** -- emotional state is reflected in responses. The engine smooths transitions between states.
- **Analysis Tool** -- triggered by explicit requests like "analyze this text." Returns structured emotion vectors without modifying AI state.

## Self-Learning Details

The learning system persists across sessions:

1. **User corrections** -- eedback command records the misclassification and learns the correction pattern
2. **Word promotion** -- frequently used expressions are automatically promoted to personalized keywords after 3+ occurrences
3. **Web lookup promotion** -- expressions discovered via web search are cached and promoted for future direct matching
4. **Cache correction** -- when user corrects a web-looked result, the cache entry is updated immediately

## Web Lookup Strategy

When keyword/n-gram analysis has low confidence (< 0.3) or text coverage is low (< 40%):

1. Extract terms not covered by the existing keyword lexicon
2. For each term, perform 3 parallel searches with different angles (general meaning, emotional tone, usage context)
3. Aggregate all 3 results into a unified emotion map
4. Cache the result for future matching
5. Promote the expression to the user's personalized keyword list

## Testing

`ash
python -m unittest discover -s tests -v
`

## Module Overview

| File | Purpose |
|------|---------|
| scripts/emotion_engine.py | Main engine, CLI entry, coordinates all modules |
| scripts/emotion_analyzer.py | Multi-label emotion classifier with 3-tier pipeline |
| scripts/emotion_tree.py | 7-root emotion taxonomy with automatic growth |
| scripts/memory_manager.py | Dual-layer memory (short-term + persistent profile) |
| scripts/self_learning_manager.py | Correction learning, word promotion, version tracking |
| scripts/emotion_lexicon.py | Keyword lexicon + psychology reference data |
| scripts/web_lookup.py | Multi-angle web search with caching and no-API fallback |
| scripts/ngram_classifier.py | Character n-gram classifier |
| scripts/model_cache.py | Shared sentence-transformers model cache |

## Requirements

- Python >= 3.10
- Optional: sentence-transformers for semantic embedding analysis
- Optional: TAVILY_API_KEY for web search fallback

The skill works without either optional dependency -- it degrades gracefully.

## License

MIT
