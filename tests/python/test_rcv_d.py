import sys
import os
import pytest
import pandas as pd

# Add Python path so multiRL is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Python")))

import multiRL

@pytest.fixture(scope="module")
def data_sub():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/TAB.csv"))
    df = pd.read_csv(csv_path)
    return df[df["Subject"] == 1]

def test_rcv_d_mle(data_sub):
    rec = multiRL.rcv_d(
        estimator="mle",
        data=data_sub,
        colnames={
            "object": ["L_choice", "R_choice"],
            "reward": ["L_reward", "R_reward"],
            "action": "Sub_Choose",
        },
        behrule={
            "cue": ["A", "B", "C", "D"],
            "rsp": ["A", "B", "C", "D"],
        },
        generating=[multiRL.TD],
        candidates=[multiRL.TD],
        control={
            "sample": 2,
            "iter": 5
        }
    )
    assert rec is not None
