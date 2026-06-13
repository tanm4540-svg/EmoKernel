# EmoKernel — Usage Guide

EmoKernel is a standalone Python emotion analysis engine. It works with any programming environment or AI that can execute command-line tools.

## Requirements

- Python >= 3.10
- Optional: `sentence-transformers` for semantic embedding analysis
- Optional: `TAVILY_API_KEY` for web lookup of unknown slang

The engine works without either optional dependency — it degrades gracefully.

## Installation

```bash
# Clone or download EmoKernel, then:
cd EmoKernel-1.0-universal

# Optional: install sentence-transformers for better accuracy
pip install sentence-transformers

# Run setup (checks deps, initialises profile directory)
python scripts/setup.py
```

## CLI Commands

All commands require `--workspace <dir>`. The workspace directory stores emotion profiles, tree data, and session state.

### 1. Process (Dialogue Mode)

Analyse a user message and generate an AI emotional state with tone guidance:

```bash
python scripts/emotion_engine.py process --workspace /tmp/ek --text "I am so happy today!"
```

**Output** (JSON):
```json
{
  "user_emotion": {"joy": 0.85, "excitement": 0.12},
  "ai_emotion_state": {"关怀": 0.45, "愉悦": 0.40, "期待": 0.15},
  "trend": {"direction": "rising"},
  "emotion_instruction": "当前情绪：喜悦，强度偏高，情绪正在上扬。用户当前情绪偏向喜悦。表达要求：语气轻快活泼，可以撒娇，多用感叹句。",
  "tree_events": []
}
```

Use `emotion_instruction` to guide your response tone.

### 2. Analyse (Pure Analysis Mode)

Analyse text without modifying AI state:

```bash
python scripts/emotion_engine.py analyze --workspace /tmp/ek --text "I am so frustrated with this bug"
```

**Output** (JSON):
```json
{
  "text": "I am so frustrated with this bug",
  "emotion": {"anger": 0.65, "sadness": 0.20, "anxiety": 0.15},
  "tree_labels": ["喜悦","愤怒","悲伤","恐惧","爱","厌恶","渴望"],
  "timestamp": 1718000000.0
}
```

### 3. Psychology Analysis

Analyse text for cognitive distortions, attachment styles, and defence mechanisms:

```bash
python scripts/emotion_engine.py psych --workspace /tmp/ek --text "Everyone hates me, I always mess everything up"
```

**Output** (JSON):
```json
{
  "emotions": {"sadness": 0.6, "fear": 0.3},
  "cognitive_distortions": [
    {"name": "全有或全无思维", "description": "事物非黑即白", "matched_keyword": "总是"},
    {"name": "读心术", "description": "认定他人对自己有负面看法", "matched_keyword": "Everyone"}
  ],
  "attachment": {"pattern": "焦虑型", "suggestion": "焦虑型依恋需要安全感确认..."},
  "coping_suggestions": [
    "分享快乐可以增强积极体验",
    "练习自我安抚和信任建立"
  ],
  "analysis": "..."
}
```

### 4. User Feedback (Self-Learning)

Tell the engine it misclassified an emotion:

```bash
python scripts/emotion_engine.py feedback \
  --workspace /tmp/ek \
  --text "I'm not angry, I'm just disappointed" \
  --detected '{"anger": 0.7}' \
  --corrected '{"sadness": 0.6, "disappointment": 0.3}'
```

The engine records this correction and adjusts future analysis after 2+ similar corrections.

### 5. Internet Slang Query

Look up unknown slang terms:

```bash
python scripts/emotion_engine.py slang --workspace /tmp/ek --term "cooked"
```

Requires `TAVILY_API_KEY` environment variable for web search. Without it, falls back to built-in slang dictionary + character-level heuristics.

### 6. Emotion Tree

View the current emotion taxonomy:

```bash
python scripts/emotion_engine.py tree --workspace /tmp/ek
```

### 7. Learning Report

View self-learning stats (corrections, promoted words, version):

```bash
python scripts/emotion_engine.py report --workspace /tmp/ek
```

### 8. Export Knowledge

Export all learned corrections and promoted keywords:

```bash
python scripts/emotion_engine.py export --workspace /tmp/ek
```

## v1.0 Enhancements

- **Synchronized voting**: when sentence-transformers is available, embedding (70%) + enhanced keyword (30%) are fused
- **Negation awareness**: detects "not happy" and avoids misclassifying as joy
- **Clause weighting**: final clause 1.5x, adversative clauses 2x
- **Graceful degradation**: works fully without sentence-transformers

## Integration Examples

### With Python

```python
import subprocess
import json

def analyse_emotion(text):
    result = subprocess.run(
        ["python", "scripts/emotion_engine.py", "analyze",
         "--workspace", "/tmp/ek", "--text", text],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

emotion = analyse_emotion("I am really excited about this!")
print(emotion)  # {"text": "...", "emotion": {"joy": 0.8, ...}, ...}
```

### With shell script

```bash
#!/bin/bash
TEXT="I am so sad right now"
python scripts/emotion_engine.py process --workspace /tmp/ek --text "$TEXT"
```

### With any HTTP API

Wrap the CLI in a simple Flask/FastAPI server to expose it as a REST endpoint.
