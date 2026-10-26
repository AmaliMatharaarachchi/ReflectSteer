# Upstream Reflexion HotPotQA

Source: https://github.com/noahshinn/reflexion

Commit: `218cf0ef1df84b05ce379dd4a8e47f17766733a0`

Copied from `hotpotqa_runs/` with the existing Qwen compatibility edits: modern
LangChain imports, relative imports, injected LLMs instead of eager OpenAI
defaults, lazy optional OpenAI imports, and the malformed-action tuple fix.
Prompts, examples and agent method bodies otherwise retain the upstream code.
The upstream license is included in `LICENSE`.

This is a normal Python package without a nested Git repository. Deploy it with
the root runner and `data/`; no separate Reflexion clone is needed for inference.
