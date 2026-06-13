# EmoKernel 1.0 Universal

**Emotion analysis engine with self-learning, negation-aware keyword matching, and synchronized embedding + keyword voting.** Works with any AI, any programming language, any platform.

> **Universal Edition** — No vendor lock-in. Use with ChatGPT, Claude, Cursor, Gemini, Copilot, or directly from the command line.

---

## Quick Start

```bash
# 1. Install dependencies (optional — works without it)
pip install sentence-transformers

# 2. Analyse some text
python scripts/emotion_engine.py analyze --workspace /tmp/ek --text "I am really frustrated right now"
```

That is it. The engine falls back gracefully if `sentence-transformers` is not installed.

---

## Features

- **Synchronized voting (v1.0)**: embedding (70%) + enhanced keyword (30%) fused when semantic model is available
- **Negation-aware**: detects "not happy" and avoids false joy classification
- **Clause-level weighting**: final sentence 1.5x, adversative clauses 2x
- **7-emotion taxonomy**: joy, anger, sorrow, fear, love, disgust, desire — with automatic branch growth for unknown emotions
- **Self-learning**: user corrections are recorded; after 2+ similar corrections, weights adjust automatically
- **Psychological analysis**: 11 cognitive distortion patterns (CBT), 4 attachment styles, defence mechanism classification
- **Internet slang fallback**: built-in slang dictionary + optional web lookup via Tavily API
- **Dual-layer memory**: short-term (last 20 turns) + long-term (cross-session persistence)
- **Graceful degradation**: works fully with or without sentence-transformers — just lower precision

---

## Files

| File | Purpose |
|------|---------|
| `SYSTEM_PROMPT.md` | Prompt template for AI integration (copy into your AI's system prompt) |
| `USAGE.md` | Full CLI documentation, examples, and integration patterns |
| `scripts/emotion_engine.py` | Main engine — CLI entry point |
| `scripts/emotion_analyzer.py` | Multi-label emotion classifier with synchronized voting |
| `scripts/emotion_tree.py` | Auto-growing emotion taxonomy |
| `scripts/memory_manager.py` | Dual-layer memory |
| `scripts/self_learning_manager.py` | Correction learning, word promotion |
| `scripts/web_lookup.py` | Internet slang fallback |
| `scripts/ngram_classifier.py` | Character n-gram ensemble classifier |
| `scripts/model_cache.py` | Shared sentence-transformers model cache |

---

## Quick Reference

```bash
# Analyse with full context (dialogue mode)
python scripts/emotion_engine.py process --workspace <dir> --text "<message>"

# Analyse only (no state change)
python scripts/emotion_engine.py analyze --workspace <dir> --text "<text>"

# Psychology analysis
python scripts/emotion_engine.py psych --workspace <dir> --text "<text>"

# View emotion tree
python scripts/emotion_engine.py tree --workspace <dir>

# Submit correction feedback
python scripts/emotion_engine.py feedback --workspace <dir> --text "<text>" --detected '{}' --corrected '{}'

# Look up slang
python scripts/emotion_engine.py slang --workspace <dir> --term "<term>"

# Learning report
python scripts/emotion_engine.py report --workspace <dir>
```

---

## AI Integration

To use EmoKernel with ChatGPT, Claude, or any other AI:

1. Copy the contents of `SYSTEM_PROMPT.md` into your AI's system prompt
2. Ensure the AI can execute Python scripts or make CLI calls
3. The AI will now understand how to call the engine and interpret its output

For platforms that support code execution natively (Claude Artifacts, ChatGPT Code Interpreter, Cursor, etc.), paste the full prompt and the AI will invoke the engine automatically.

---

## Architecture

```
User text
  |
  +-> EmotionAnalyzer
  |     +-> sentence-embedding (when model available) —— 70%
  |     +-> enhanced keyword + n-gram vote ———————————— 30%
  |     +-> web_lookup (fallback for unknown expressions)
  |
  +-> EmotionTree — 7-root emotion taxonomy, auto-growth
  +-> MemoryManager — session queue (20 rounds) + persistent profile
  +-> SelfLearningManager — correction patterns, word promotion, version tracking
  |
  +-> EmotionEngine — CLI entry point, coordinates all modules
```

## License

MIT
