# Deployment Guide

How to release a new version of `deep-semantic-search` to PyPI.

## Prerequisites

- Push access to `Harduex/deep-semantic-search` on GitHub
- PyPI trusted publishing configured for this repo (already set up via OIDC)
- No PyPI API token needed — the GitHub Actions workflow uses trusted publishing

## Release Steps

### 1. Update the version number

Bump the version in **both** of these files:

- `pyproject.toml` — line `version = "X.Y.Z"`
- `src/deep_semantic_search/__init__.py` — line `__version__ = "X.Y.Z"`

### 2. Update the changelog

Add a new section to `CHANGELOG.md` describing:

- Breaking changes
- New features
- Bug fixes
- Dependency changes
- Migration guide (if breaking)

### 3. Verify the package locally

```bash
# Lint
pip install ruff
ruff check src/ tests/

# Run tests
pip install -e ".[dev]"
pytest tests/ -v

# Test build
pip install build
python -m build
```

All checks must pass before tagging.

### 4. Commit and push

```bash
git add -A
git commit -m "feat: release vX.Y.Z — brief description"
git push origin master
```

### 5. Create and push the version tag

This triggers the PyPI publish workflow (`.github/workflows/publish.yml`).

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 6. Verify deployment

1. Check GitHub Actions: https://github.com/Harduex/deep-semantic-search/actions
   - The **Publish to PyPI** workflow should complete successfully
   - The **CI** workflow should also pass on the `master` push
2. Verify on PyPI: https://pypi.org/project/deep-semantic-search/
3. Test installation:
   ```bash
   pip install deep-semantic-search==X.Y.Z
   python -c "import deep_semantic_search; print(deep_semantic_search.__version__)"
   ```

## How It Works

### CI Workflow (`.github/workflows/ci.yml`)

Runs on every push/PR to `master`:
- **lint** — `ruff check src/ tests/`
- **test** — `pytest tests/ -v` on Python 3.10, 3.11, 3.12
- **build** — `python -m build` (runs after lint+test pass)

### Publish Workflow (`.github/workflows/publish.yml`)

Triggers on tags matching `v*`:
- Builds the package with `python -m build`
- Publishes to PyPI using `pypa/gh-action-pypi-publish` with OIDC trusted publishing

### Key detail

The publish workflow triggers on **git tags**, not on commits. If you push a commit without a tag, the package will not be published.

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Package not published | No `v*` tag pushed | Create and push the tag |
| CI not running | Workflow targets wrong branch | Ensure `ci.yml` targets `master` |
| Tests fail in CI | Missing dependency | Check `pyproject.toml` dependencies |
| Version conflict on PyPI | Version already exists | Bump the version number |

## Version History

| Version | Tag | Date |
|---------|-----|------|
| 3.0.0 | v3.0.0 | 2026-04-13 |
| 2.0.0 | v2.0.0 | — |
| 1.1.4 | v1.1.4 | — |
| 1.1.3 | v1.1.3 | — |
| 1.1.2 | v1.1.2 | — |
| 1.1.1 | v1.1.1 | — |
| 1.1.0 | v1.1.0 | — |
