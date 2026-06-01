# %%
############################
# Development Package Setup
############################

# pip install -e ./Python --config-settings=build-dir="build"

import multiRL
import pandas

# %%

############################
# TD Policy-Off Benchmark
############################

data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

result = multiRL.run_m(
    data=data,
    behrule={
        "cue": ["A", "B", "C", "D"],
        "rsp": ["A", "B", "C", "D"],
    },
    colnames={
        "object": ["L_choice", "R_choice"],
        "reward": ["L_reward", "R_reward"],
        "action": "Sub_Choose",
    },
    params={
        "free": {
            "alpha": 0.3,
            "beta": 0.5,
        },
        "fixed": {
            "gamma": 1.0,
            "delta": 0.1,
            "epsilon": float("nan"),
            "zeta": 0.0,
        },
        "constant": {
            "seed": 123.0,
            "L": float("nan"),
            "penalty": 1.0,
            "Q0": float("nan"),
            "reset": float("nan"),
            "lapse": 0.01,
            "threshold": 20.0,
            "bonus": 0.0,
            "weight": 1.0,
            "capacity": 0.0,
            "sticky": 0.0,
        },
    },
    priors={
        "alpha": {
            "type": "beta",
            "shape1": 1.0,
            "shape2": 1.0,
        },
        "beta": {
            "type": "normal",
            "mean": 0.0,
            "sd": 3.0,
        },
    },
    settings={
        "system": ["RL"],
        "policy": "off",
        "name": "TD",
        "mode": "fitting",
        "estimate": "MLE",
    },
)

logL = result["metric"]["LogL"]

print(f"multiRL LogL: {logL:.5f}")

if round(logL, 2) != -317.45:
    raise RuntimeError("multiRL LogL is not -317.45 after rounding.")
