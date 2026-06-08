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
        "n_draws": 30,
        "seed": 1004,
        "threads": 1,
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

if len(mle["truth"]) != 90:
    raise RuntimeError("Python rcv_d MLE did not simulate three models.")

if len(mle["model_recovery"]) != 270:
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
        "n_draws": 30,
        "seed": 1004,
        "threads": 1,
        "scope": "shared",
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

if len(abc["truth"]) != 90:
    raise RuntimeError("Python rcv_d ABC did not simulate three models.")

if len(abc["model_recovery"]) != 270:
    raise RuntimeError("Python rcv_d ABC did not fit all model pairs.")

if abc["estimator"]["scope"] != "shared":
    raise RuntimeError("Python rcv_d ABC did not keep shared scope.")

# %%
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")

models = [
    multiRL.TD,
]

settings = [
    {"name": "TD"},
]

base_control = {
    "n_draws": 5,
    "seed": 1004,
    "threads": 1,
    "epochs": 1,
    "batch_size": 8,
    "units": 8,
    "layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.001,
    "model_type": "gru",
    "verbose": 0,
    "scope": "shared",
}

control_cpu = dict(base_control)
control_cpu["device"] = "cpu"

control_gpu = dict(base_control)
control_gpu["threads"] = 0
control_gpu["device"] = "cuda"

rnn_cpu = multiRL.rcv_d(
    estimator="rnn",
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
    lowers=[[0, 0]],
    uppers=[[1, 5]],
    control=control_cpu,
)

rnn_gpu = multiRL.rcv_d(
    estimator="rnn",
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
    lowers=[[0, 0]],
    uppers=[[1, 5]],
    control=control_gpu,
)

print(rnn_cpu["recovery"])
print(rnn_gpu["recovery"])

if len(rnn_cpu["simulation"]) == 0:
    raise RuntimeError("Python rcv_d RNN CPU returned no simulated rows.")

if len(rnn_gpu["simulation"]) == 0:
    raise RuntimeError("Python rcv_d RNN GPU returned no simulated rows.")

if len(rnn_cpu["truth"]) != len(rnn_gpu["truth"]):
    raise RuntimeError("Python rcv_d RNN CPU/GPU truth lengths differ.")

if len(rnn_cpu["recovery"]) != len(rnn_gpu["recovery"]):
    raise RuntimeError("Python rcv_d RNN CPU/GPU recovery lengths differ.")

for cpu_row, gpu_row in zip(rnn_cpu["recovery"], rnn_gpu["recovery"]):
    if cpu_row["parameter"] != gpu_row["parameter"]:
        raise RuntimeError("Python rcv_d RNN CPU/GPU parameters differ.")
    if abs(float(cpu_row["true"]) - float(gpu_row["true"])) > 1e-10:
        raise RuntimeError("Python rcv_d RNN CPU/GPU truth values differ.")
    if not float(cpu_row["recovered"]) == float(cpu_row["recovered"]):
        raise RuntimeError("Python rcv_d RNN CPU returned NaN.")
    if not float(gpu_row["recovered"]) == float(gpu_row["recovered"]):
        raise RuntimeError("Python rcv_d RNN GPU returned NaN.")

print("Python rcv_d RNN CPU/GPU device test passed.")
