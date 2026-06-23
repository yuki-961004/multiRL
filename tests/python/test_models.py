import sys
import os
import math

# Add Python path so multiRL is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Python")))

import multiRL

def test_td_model():
    m = multiRL.TD()
    assert m["model"] == "TD"
    assert m["process"] == "process_model_free"
    assert m["params"]["free"]["alpha"] == 0.3
    assert m["params"]["free"]["beta"] == 0.5
    assert m["priors"]["alpha"]["type"] == "beta"

def test_rstd_model():
    m = multiRL.RSTD()
    assert m["model"] == "RSTD"
    assert m["params"]["free"]["alphaN"] == 0.3
    assert m["params"]["free"]["alphaP"] == 0.3

def test_utility_model():
    m = multiRL.Utility()
    assert m["model"] == "Utility"
    assert m["params"]["free"]["gamma"] == 0.5
