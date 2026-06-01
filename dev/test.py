# %%

"""Manual development script for the Python multiRL frontend."""

import pandas


############################
# Development Package Setup
############################

# pip install -e ./Python --config-settings=build-dir="build"

import multiRL


# %%

############################
# TD Policy-Off Benchmark
############################

data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

result = multiRL.run_m(
    object=data[["L_choice", "R_choice"]].values.tolist(),
    reward=data[["L_reward", "R_reward"]].values.tolist(),
    action=data["Sub_Choose"].tolist(),
    block=data["Block"].tolist(),
    trial=data["Trial"].tolist(),
    cue=["A", "B", "C", "D"],
    rsp=["A", "B", "C", "D"],
    params={
        "alpha": 0.3,
        "beta": 0.5,
        "gamma": 1.0,
        "delta": 0.1,
        "epsilon": float("nan"),
        "zeta": 0.0,
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
    free_names=["alpha", "beta"],
    system=["RL"],
    policy="off",
    name="TD",
    mode="fitting",
    estimate="MLE",
)

logL = result["metric"]["LogL"]

print(f"multiRL LogL: {logL:.5f}")

if round(logL, 2) != -317.45:
    raise RuntimeError("multiRL LogL is not -317.45 after rounding.")

