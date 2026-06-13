# Changelog

## v1.0-universal — Universal AI Edition (2026-06-13)

Based on v1.0 with all its improvements, plus:

### What changed
- **Removed `SKILL.md`** — the Codex-specific plugin manifest is gone
- **Added `SYSTEM_PROMPT.md`** — a ready-to-paste system prompt template for any AI (ChatGPT, Claude, Cursor, Gemini, Copilot, etc.)
- **Added `USAGE.md`** — universal CLI documentation with integration examples in Python, shell, and REST
- **Rewrote `README.md`** — universal-first: works with any AI, any platform, no vendor lock-in

### What stayed the same
- All Python engine code is identical to v1.0
- All CLI commands and outputs are unchanged
- All emotion analysis, psychology analysis, self-learning, and web lookup features
- MIT license

### How to use with other AIs
1. Copy `SYSTEM_PROMPT.md` into the AI's system prompt / project rules / custom instructions
2. Ensure the AI can execute Python scripts (most coding AIs support this natively)
3. Done — the AI now knows how to call the engine and interpret its output
