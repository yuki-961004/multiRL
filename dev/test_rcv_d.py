# Install from the Python package directory before running:
# python -m pip install -e Python

# %%
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")

models = [
    multiRL.TD,
    multiRL.RSTD,
    multiRL.Utility,
]

settings = [
    {"name": "TD"},
    {"name": "RSTD"},
    {"name": "Utility"},
]

mle = multiRL.rcv_d(
    estimator="mle",
    data=data,
    id=1,
    behrule={
        "cue": ["A", "B", "C", "D"],
        "rsp": ["A", "B", "C", "D"],
    },
    colnames={
        "subid": "Subject",
        "block": "Block",
        "trial": "Trial",
        "object": ["L_choice", "R_choice"],
        "reward": ["L_reward", "R_reward"],
        "action": "Sub_Choose",
    },
    models=models,
    settings=settings,
    lowers=[[0, 0], [0, 0, 0], [0, 0, 0]],
    uppers=[[1, 5], [1, 1, 5], [1, 5, 1]],
    control={
        "n_draws": 1,
        "seed": 1004,
        "threads": 1,
    },
    fit_control={
        "algorithm": "LN_BOBYQA",
        "local_algorithm": "LN_BOBYQA",
        "maxeval": 10,
        "seed": 1004,
    },
)

print(mle["recovery"])
print(mle["model_recovery"])

if len(mle["simulation"]) == 0:
    raise RuntimeError("Python rcv_d MLE returned no simulated rows.")

if len(mle["truth"]) != 3:
    raise RuntimeError("Python rcv_d MLE did not simulate three models.")

if len(mle["model_recovery"]) != 9:
    raise RuntimeError("Python rcv_d MLE did not fit all model pairs.")

# %%
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")

models = [
    multiRL.TD,
    multiRL.RSTD,
    multiRL.Utility,
]

settings = [
    {"name": "TD"},
    {"name": "RSTD"},
    {"name": "Utility"},
]

abc = multiRL.rcv_d(
    estimator="abc",
    data=data,
    id=1,
    behrule={
        "cue": ["A", "B", "C", "D"],
        "rsp": ["A", "B", "C", "D"],
    },
    colnames={
        "subid": "Subject",
        "block": "Block",
        "trial": "Trial",
        "object": ["L_choice", "R_choice"],
        "reward": ["L_reward", "R_reward"],
        "action": "Sub_Choose",
    },
    models=models,
    settings=settings,
    lowers=[[0, 0], [0, 0, 0], [0, 0, 0]],
    uppers=[[1, 5], [1, 1, 5], [1, 5, 1]],
    control={
        "n_draws": 1,
        "seed": 1004,
        "threads": 1,
        "scope": "shared",
    },
    fit_control={
        "samples": 10,
        "tol": 0.5,
        "method": "rejection",
        "reduction": "none",
        "threads": 1,
        "print_level": 0,
    },
)

print(abc["recovery"])
print(abc["model_recovery"])

if len(abc["simulation"]) == 0:
    raise RuntimeError("Python rcv_d ABC returned no simulated rows.")

if len(abc["truth"]) != 3:
    raise RuntimeError("Python rcv_d ABC did not simulate three models.")

if len(abc["model_recovery"]) != 9:
    raise RuntimeError("Python rcv_d ABC did not fit all model pairs.")

if abc["estimator"]["scope"] != "shared":
    raise RuntimeError("Python rcv_d ABC did not keep shared scope.")
