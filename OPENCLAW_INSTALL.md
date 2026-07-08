# OpenClaw Installation Guide for Time Series Sandbox Skill

This guide explains how to install and use the local Time Series Sandbox skill with OpenClaw.

## 1. Prerequisites

- OpenClaw is installed and working.
- Python 3.10+ is available.
- Git is available.

## 2. Skill Location

The local skill lives in this repository at:

- `.agent/skills/time-series-sandbox/`

If OpenClaw uses a separate skills directory, copy or symlink this folder there.

Ensure this file exists after placement:

- `.agent/skills/time-series-sandbox/SKILL.md`

## 3. One-Line Project Install and Smoke Run

Use this exact command:

```bash
git clone https://github.com/NetManAIOps/sktime.git && cd sktime && python -m venv .venv && source .venv/bin/activate && python -m pip install -U pip && python -m pip install -e ".[all_extras]" && python -c "import sktime; print('Time Series Sandbox ready:', sktime.__version__)"
```

If you already cloned the repository:

```bash
cd sktime && python -m venv .venv && source .venv/bin/activate && python -m pip install -U pip && python -m pip install -e ".[all_extras]" && python -c "import sktime; print('Time Series Sandbox ready:', sktime.__version__)"
```

## 4. Optional Script-Based Setup

Run the provided script:

```bash
bash .agent/skills/time-series-sandbox/setup.sh
```

Or for an existing local repository path:

```bash
bash .agent/skills/time-series-sandbox/setup.sh /path/to/sktime
```

## 5. Validation Checklist

- OpenClaw can load and use `time-series-sandbox` skill.
- `import sktime` succeeds.
- Notebook examples are discoverable under `examples/` and `lectures/`.
- Colab links can be constructed with base:
  - `https://colab.research.google.com/github/NetManAIOps/sktime/blob/main`
- TSBox Sandbox Playground starts locally:
  - `python3 playground/server.py`
- Playground tests pass:
  - `python3 playground/test_playground.py`

## 6. Recommended Query Patterns for Users

- "Install and run Time Series Sandbox in one command"
- "What algorithm categories are implemented in Time Series Sandbox?"
- "What datasets are available?"
- "Do we have notebook reference cases for detection or forecasting?"
- "What is added in Time Series Sandbox compared to sktime?"

For the last question, if Feishu plugin is available, search Feishu KB first; then verify with `docs/` and source code paths.
