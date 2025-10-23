# PyPI Release Guide

This guide explains how to publish the Applied Probability Framework to PyPI.

## Prerequisites

```bash
pip install build twine
```

## Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution packages
python -m build
```

This creates:
- `dist/applied_probability_framework-1.0.0.tar.gz` (source distribution)
- `dist/applied_probability_framework-1.0.0-py3-none-any.whl` (wheel)

## Test on TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ applied-probability-framework
```

## Publish to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Verify
pip install applied-probability-framework
```

## Post-Publication

Update README.md to replace "coming soon" with actual pip install command:

```markdown
### Installation

```bash
pip install applied-probability-framework
```
```

## Version Management

Update version in:
1. `setup.py` - `version="1.0.0"`
2. `src/python/core/__init__.py` - `__version__ = "1.0.0"`
3. Git tag: `git tag v1.0.0`

## Automated Release (GitHub Actions)

The `.github/workflows/release.yml` workflow automatically:
1. Builds the package
2. Creates GitHub release
3. (Can be configured to) Upload to PyPI on tag push

To trigger:
```bash
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin v1.0.1
```

## PyPI Project Page

Once published, the project will be available at:
https://pypi.org/project/applied-probability-framework/

## Best Practices

1. **Always test on TestPyPI first**
2. **Use semantic versioning** (MAJOR.MINOR.PATCH)
3. **Tag releases in git** matching PyPI versions
4. **Update CHANGELOG.md** with release notes
5. **Verify installation** in fresh environment

## Troubleshooting

### "File already exists" error
- You cannot re-upload the same version
- Increment version number and rebuild

### Import errors after installation
- Check `package_dir` in setup.py is correct
- Verify `MANIFEST.in` includes all necessary files

### Missing dependencies
- Check `install_requires` in setup.py
- Test in fresh virtual environment

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [TestPyPI](https://test.pypi.org/)
- [PyPI](https://pypi.org/)


