import sys
import os
import math
import pytest
import pandas as pd

# Add Python path so multiRL is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Python")))

import multiRL

def test_run_m_td_generate_false():
    data = pd.DataFrame({
        "Subject": [1, 1, 1],
        "Block": [1, 1, 1],
        "Trial": [1, 2, 3],
        "L_choice": ["A", "A", "C"],
        "R_choice": ["B", "B", "D"],
        "L_reward": [1.0, 1.0, 0.0],
        "R_reward": [0.0, 0.0, 1.0],
        "Sub_Choose": ["A", "B", "D"]
    })

    res = multiRL.run_m(
        data=data,
        colnames={
            "object": ["L_choice", "R_choice"],
            "reward": ["L_reward", "R_reward"],
            "action": "Sub_Choose",
        },
        behrule={
            "cue": ["A", "B", "C", "D"],
            "rsp": ["A", "B", "C", "D"],
        },
        params={
            "free": {"alpha": 0.3, "beta": 0.5},
            "fixed": {"threshold": 20.0}
        },
        priors={
            "alpha": {"type": "beta", "shape1": 2, "shape2": 2},
            "beta": {"type": "exponential", "rate": 1}
        },
        settings={
            "name": "TD",
            "mode": "fitting",
            "estimate": "MLE",
            "generate": False
        }
    )

    assert res is not None
    assert "metric" in res
    assert res["metric"]["LogL"] == pytest.approx(3.0 * math.log(0.5))
    assert "behave" in res["result"]
