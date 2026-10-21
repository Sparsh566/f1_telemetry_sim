"""
tests/test_perceptron_xor.py
============================
Unit tests for SinglePerceptron, decision boundary generation,
and the XOR non-linear separability limitation.
"""

import numpy as np
from ml.neural_network_numpy import SinglePerceptron


def test_perceptron_forward_and_step():
    perc = SinglePerceptron(in_features=2, seed=42)
    perc.weights = np.array([2.0, -1.0])
    perc.bias = 0.5

    # Input: [1.0, 1.0] -> z = 2(1) - 1(1) + 0.5 = 1.5 -> step=1
    X = np.array([[1.0, 1.0], [-1.0, 2.0]])
    steps = perc.predict_step(X)
    assert steps[0] == 1
    # [-1, 2] -> z = 2(-1) - 1(2) + 0.5 = -3.5 -> step=0
    assert steps[1] == 0


def test_perceptron_decision_boundary():
    perc = SinglePerceptron(in_features=2, seed=42)
    perc.weights = np.array([1.0, 2.0])
    perc.bias = -4.0

    # Line: 1*x1 + 2*x2 - 4 = 0 -> x2 = (4 - x1) / 2
    x1_vals, x2_vals = perc.get_decision_boundary((0.0, 4.0))
    assert x1_vals == [0.0, 4.0]
    assert x2_vals[0] == 2.0  # at x1=0, x2=2
    assert x2_vals[1] == 0.0  # at x1=4, x2=0


def test_xor_limitation_demonstration():
    res = SinglePerceptron.demonstrate_xor_limitation()
    # Single linear perceptron cannot separate XOR (accuracy should be ~50%, definitely < 100%)
    assert res["accuracy"] <= 0.75
    assert "hyperplane" in res["explanation"] or "linear" in res["explanation"]
