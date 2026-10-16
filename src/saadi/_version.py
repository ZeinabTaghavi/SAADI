"""The single place where the package version is set.

``pyproject.toml`` reads it at build time (``[tool.setuptools.dynamic]``),
``saadi --version`` prints it, and the release workflow checks that a tag
``vX.Y.Z`` matches it.
"""

__version__ = "0.1.0"
