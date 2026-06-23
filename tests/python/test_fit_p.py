import sys
import os
import math
import pytest
import pandas as pd

# Add Python path so multiRL is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Python")))

import multiRL

@pytest.fixture(scope="module")
def data_sub():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/TAB.csv"))
    df = pd.read_csv(csv_path)
    return df[df["Subject"].isin([1, 2])]

def test_mle_fitting(data_sub):
    fit = multiRL.fit_p(
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
        models=[multiRL.TD, multiRL.RSTD],
        control={"iter": 5}
    )
    assert fit is not None
    assert fit["diagnostics"]["n_models"] == 2

def test_map_fitting(data_sub):
    fit = multiRL.fit_p(
        estimator="map",
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
        models=[multiRL.TD],
        control={"iter": [5, 1]}
    )
    assert fit is not None

def test_abc_fitting(data_sub):
    for red in [None, "PLS", "PCA"]:
        fit = multiRL.fit_p(
            estimator="abc",
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
            models=[multiRL.TD],
            control={
                "sample": 10,
                "train": 10,
                "tol": 0.5,
                "reduction": red
            }
        )
        assert fit is not None

def test_mcmc_fitting(data_sub):
    try:
        fit = multiRL.fit_p(
            estimator="mcmc",
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
            models=[multiRL.TD],
            control={
                "warmup": 2,
                "samples": 3,
                "chains": 1
            }
        )
        assert fit is not None
    except Exception as e:
        err_msg = str(e)
        if "Stan Math" in err_msg or "MCMC is not available" in err_msg:
            pytest.skip("Stan Math support not compiled")
        else:
            raise

def test_rnn_fitting(data_sub):
    try:
        fit = multiRL.fit_p(
            estimator="rnn",
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
            models=[multiRL.TD],
            control={
                "epochs": 1,
                "n_draws": 10,
                "batch_size": 5
            }
        )
        assert fit is not None
    except Exception as e:
        err_msg = str(e)
        if "LibTorch" in err_msg or "multiRL_torch_backend" in err_msg or "Cannot load" in err_msg:
            pytest.skip("LibTorch support not compiled")
        else:
            raise
