# data-analysis-strand-research

### Folder Layout Version 0.1
hairmech/
├─ pyproject.toml          ← Poetry, build-system & deps
├─ README.md
├─ CHANGELOG.md
├─ Dockerfile
├─ src/
│  └─ hairmech/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ io.py
│     ├─ util.py
│     ├─ dimensional.py
│     ├─ tensile.py
│     ├─ analysis.py
│     ├─ plots.py
│     └─ app_dash.py
└─ tests/
   ├─ conftest.py
   ├─ data/                 ← tiny .txt fixtures
   ├─ test_dimensional.py
   ├─ test_tensile.py
   ├─ test_analysis.py
   └─ test_cli.py
