# Data Analysis Application for Strand Research
Data visualization and analysis toolkit for Dia-Stron tensile experiments.
The package ingests exported dimensional and tensile `.txt` files, cleans
slot metadata, computes summary metrics, and writes Excel/HTML reports. A
Dash front-end mirrors the notebook workflow for interactive exploration.

## Installation

```bash
# 1. Clone the repo and cd into it
git clone https://github.com/cathal-t/data-analysis-strand-research.git
cd data-analysis-strand-research

# 2. Poetry install
poetry install

# 3. Launch 
poetry run hairmech serve
```

The package installs a `hairmech` console script. All commands below assume
an active Poetry shell or any environment with the project dependencies.

## Installation (Mac)

```bash
# 1. Install Homebrew (if you don’t have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Add Homebrew to your shell (zsh is the default in modern macOS)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 3. Install Python 3 (Homebrew will give you the latest 3.x)
brew install python

# 4. Install Poetry
brew install poetry

# 5. Clone the repo and cd into it
git clone https://github.com/cathal-t/data-analysis-strand-research.git
cd data-analysis-strand-research

# 6. Poetry install
poetry install

# 7. Launch 
poetry run hairmech serve
```

## CLI usage

Launch the interactive Dash UI to explore the same pipeline in a
browser:

```bash
 cd data-analysis-strand-research
 git checkout main
 git pull
 poetry run hairmech serve
```
## Terminal usage (Mac)
```bash
cd ~/Desktop
cat > run_hairmech.command << 'EOF'
#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Double-click this .command file to open Terminal, cd into your repo, and
# start the Hair-mech Dash app via Poetry.
# ─────────────────────────────────────────────────────────────────────────────
# Change this to the path where you’ve cloned the repo:
REPO_PATH="data-analysis-strand-research"
# 1. cd into the repo
cd "$REPO_PATH" || {
  echo ":x: Could not cd into $REPO_PATH"
  exit 1
}
# 2. Make sure you’re on main and up to date
git checkout main
git pull
# 3. Launch the app
poetry run hairmech serve
EOF
chmod +x run_hairmech.command
```

The UI lives at `http://127.0.0.1:8050` and provides upload controls plus
tabs for overlay and violin visualisations.

## Project layout

```
hairmech/
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

