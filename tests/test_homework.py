"""Autograding script."""

import os


def test_01():
    """Test"""

    assert os.path.exists("models"), "models/ directory does not exist"
    assert os.path.exists("homework/src"), "models/ directory does not exist"
    assert os.path.exists(
        "models/estimator.pkl"
    ), "models/estimator.pkl file does not exist"
