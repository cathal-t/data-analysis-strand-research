# Data Analysis Platform for Strand Research

This project is a data visualization and analysis toolkit for Dia-Stron tensile
experiments. It ingests exported dimensional and tensile `.txt` files, cleans
slot metadata, computes summary metrics, and writes Excel/HTML reports. A Dash
front-end mirrors the notebook workflow for interactive exploration.

## Features

- Parse Dia-Stron dimensional and tensile exports.
- Clean and normalize slot metadata.
- Compute summary metrics and statistics for experiments.
- Export results to Excel and HTML reports.
- Explore data interactively with a Dash UI.

## Requirements

- Python 3.11
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

```bash
# 1. Clone the repo and enter it
git clone https://github.com/cathal-t/data-analysis-strand-research.git
cd data-analysis-strand-research

# 2. Install dependencies
poetry install
```

### macOS setup (Homebrew)

```bash
# 1. Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Add Homebrew to your shell
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 3. Install Python and Poetry
brew install python
brew install poetry

# 4. Clone and install dependencies
git clone https://github.com/cathal-t/data-analysis-strand-research.git
cd data-analysis-strand-research
poetry install
```

## Quick start

Launch the Dash UI:

```bash
poetry run hairmech serve
```

The UI is available at `http://127.0.0.1:8050`.

## CLI usage

```bash
# Start the Dash app
poetry run hairmech serve
```

## Terminal launcher (macOS)

Create a double-clickable `.command` file that launches the app:

```bash
cat > run_hairmech.command << 'EOF'
#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Double-click this .command file to open Terminal, cd into your repo, and
# start the Hairmech Dash app via Poetry.
# ─────────────────────────────────────────────────────────────────────────────
# Change this to the path where you’ve cloned the repo:
REPO_PATH="data-analysis-strand-research"

cd "$REPO_PATH" || {
  echo ":x: Could not cd into $REPO_PATH"
  exit 1
}

git checkout main
git pull
poetry run hairmech serve
EOF
chmod +x run_hairmech.command
```

## Project layout

```
.
├─ src/hairmech/
│  ├─ cli.py                 # click entry-point for run/serve commands
│  ├─ analysis.py            # metric calculations & stats helpers
│  ├─ dimensional.py         # Dia-Stron dimensional loader → area map
│  ├─ dimensioncleaning.py   # helpers for cleaning dimensional exports
│  ├─ tensile.py             # tensile data parsing & stress/strain helpers
│  ├─ util.py                # shared maths/utilities
│  ├─ plots.py               # Plotly figures (overlay, violin grid)
│  ├─ io/
│  │  ├─ config.py           # YAML loader for experiment conditions
│  │  └─ export.py           # Excel writers for metrics/statistics
│  └─ ui/
│     ├─ __init__.py
│     └─ app.py              # Dash layout + callbacks (build_dash_app)
└─ tests/
   ├─ fixtures/demo_exp/     # tiny experiment fixture with config.yml
   ├─ test_cli.py            # CLI smoke tests
   ├─ test_analysis.py       # metric/statistics helpers
   ├─ test_dimensional.py    # dimensional loader
   ├─ test_tensile.py        # tensile parsing helpers
   ├─ test_util.py           # utility functions
   ├─ test_export.py         # Excel export sanity checks
   ├─ test_ui.py             # Dash UI smoke tests
   └─ test_ui_app.py         # UI layout helpers
```

## Development

```bash
# Run tests
poetry run pytest
```

## License

See [LICENSE](LICENSE).
