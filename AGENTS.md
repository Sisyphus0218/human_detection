# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 human-following and tracking pipeline. Runtime code lives in `human_detection/`, organized by responsibility: input sources, person detection/tracking, target matching/classification, pose and position estimation, visualization, and follow control. Hydra presets and defaults are under `configs/`; downloaded model checkpoints belong in `models/`; sample inputs go in `inputs/`; generated videos and registration crops go in `results/` or `outputs/`. Behavioral tests and their fixture CSV/GIF files are in `test/`. Utility scripts, including checkpoint download support, are in `scripts/`.

## Build, Test, and Development Commands

Create and activate the documented virtual environment, then install dependencies:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Download checkpoints with `python scripts/download_models.py`. Run the application from the repository root, for example:

```bash
python -m human_detection.main target.name=person_a source=video source.name=demo
```

Use `device=cpu`, `display_enabled=false`, or `video_enabled=false` for headless and CPU runs. Execute the test suite with `pytest` (or a focused file such as `pytest test/test_follow_controller.py`).

## Coding Style & Naming Conventions

Follow standard Python style (PEP 8), four-space indentation, and readable type annotations. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and lowercase YAML preset names. Keep pipeline components small and configuration-driven, and match existing import ordering and docstring conventions. Format or lint changed Python files with the project’s configured editor tooling before submitting.

## Testing Guidelines

Tests use pytest and focus on controller behavior, including target occlusion and detection dropouts. Name new files `test_<area>.py` and test functions `test_<behavior>`. Reuse or add compact fixtures under `test/`; avoid committing large model files or generated result videos. Run `pytest` before opening a pull request.

## Commit & Pull Request Guidelines

Recent history uses short conventional-style subjects such as `feat(...)`, `fix(...)`, `docs(...)`, and `refactor(...)`, sometimes with an emoji prefix. Follow that pattern: use an imperative, scoped subject and keep it focused. Pull requests should explain the behavior change, list validation commands and configuration assumptions, link related issues, and include screenshots or sample output when visualization changes. Keep generated artifacts, credentials, and local machine paths out of commits.

## Configuration & Data Safety

Do not commit model checkpoints, private input footage, or calibration data. Store local camera intrinsics, robot extrinsics, and runtime paths in untracked overrides or local configuration, and verify `.gitignore` coverage before sharing changes.
