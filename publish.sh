#!/bin/bash

# Exit on any error
set -e

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Copy PyPI README to be used for packaging
echo "Using PyPI README..."
cp README.md.pypi README.md.original
cp README.md.pypi README.md

# Build source and wheel distributions
echo "Building distribution packages..."
python -m build

# Restore original README
echo "Restoring original README..."
mv README.md.original README.md

# Choose repository
echo "Where would you like to publish?"
echo "1) TestPyPI (for testing)"
echo "2) PyPI (production)"
echo "3) Skip upload"
read -p "Enter choice (1-3): " repo_choice

case $repo_choice in
1)
    echo "Uploading to TestPyPI..."
    python -m twine upload --repository testpypi dist/*
    echo "If successful, you can install with:"
    echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deep-semantic-search"
    ;;
2)
    echo "Uploading to PyPI..."
    python -m twine upload dist/*
    echo "If successful, you can install with:"
    echo "pip install deep-semantic-search"
    ;;
3)
    echo "Skipping upload to PyPI."
    echo "To upload manually later, run one of these commands:"
    echo "python -m twine upload dist/* (for PyPI)"
    echo "python -m twine upload --repository testpypi dist/* (for TestPyPI)"
    ;;
*)
    echo "Invalid choice. Skipping upload."
    ;;
esac

echo "Done!"
