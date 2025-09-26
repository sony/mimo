# Paper Artifact - LLMAgentBanner

This directory provides a minimal, runnable artifact accompanying the paper.

What’s included
- Source code: `src/multiagent`, `src/prompts`, `src/okg`, `src/result_manager.py`
- Configuration: `config/config_llm.ini` (temporarily includes keys for reproduction; remove or switch to env vars after verification)
- Example data: `logos/001_ethicai.png` and `logos/001_ethicai_prompt.txt`
- Entry script: `test_naming_convention.py`
- One-click run script: `scripts/run_example.sh`

Quick start
1) Run the example (creates a venv and installs deps on first run)
```bash
bash scripts/run_example.sh
```

2) Run with a different logo
```bash
python test_naming_convention.py --logo wildcare
```
See available logos in `logos/`.

3) Dynamic styles (optional)
```bash
python test_naming_convention.py --logo ethicai --dynamic-styles --style-candidates 8
```

File naming convention
Final images follow:
```
generated_image_GraphicRevisor_Team{TeamID}_Round{RoundNumber}_{RevisionNumber}.png
```
The script prints verification details in the console.

Configuration and keys
- Main configuration: `config/config_llm.ini`
- Image model is configured via `IMAGE_GENERATION.MODEL` (e.g., `gemini` or `gpt`)
- Keys are temporarily present under `[KEYS]` for reproduction; after verifying the run, please remove them or use environment variables instead

Recommended environment variables (if you don’t want keys in the ini file):
- `GEMINI_API_KEY`
- If switching to OpenAI/Azure paths, the corresponding key and endpoint variables

Requirements
- Python 3.10+
- See `requirements.txt`

Directory structure
```
paper_artifact/
  ├── README.md
  ├── requirements.txt
  ├── scripts/
  │   └── run_example.sh
  ├── config/
  │   └── config_llm.ini
  ├── logos/
  │   ├── 001_ethicai.png
  │   └── 001_ethicai_prompt.txt
  ├── src/
  │   ├── multiagent/
  │   ├── prompts/
  │   ├── okg/
  │   └── result_manager.py
  └── test_naming_convention.py
```

Disclaimer
- Keys in the example configuration are only for paper reproduction. Do not commit them to public repositories. After confirming local runs, remove keys or switch to environment variables.
