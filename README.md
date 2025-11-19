# Data Analysis Application for Strand Research
Data visualization and analysis toolkit for Dia-Stron tensile experiments.
The package ingests exported dimensional and tensile `.txt` files, cleans
slot metadata, computes summary metrics, and writes Excel/HTML reports. A
Dash front-end mirrors the notebook workflow for interactive exploration.

## Installation

```bash
poetry install
```

The package installs a `hairmech` console script. All commands below assume
an active Poetry shell or any environment with the project dependencies.

## CLI usage

Launch the interactive Dash UI to explore the same pipeline in a
browser:

```bash
 poetry install
 poetry run hairmech serve
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

## Development

Run the test suite with:

```bash
pytest
```
