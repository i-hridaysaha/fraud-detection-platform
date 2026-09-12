"""The MLflow model-from-code entry point: what a registered version runs when it is loaded.

`scripts/register_model.py` logs this file as the model's code. MLflow copies it into the
version's artifacts and executes it on load, so the loaded object is `serving.FraudScorer`
built from the version's own files and nothing pickled. The package it imports is the installed
one; a version is therefore only as portable as the package it was registered under, which is
what the pinned dependencies in pyproject.toml are for.
"""

from __future__ import annotations

import mlflow

from fraud_platform.serving import as_pyfunc

mlflow.models.set_model(as_pyfunc())
