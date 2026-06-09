# Optional dependency install from the repository root:
# $env:MULTIRL_ENABLE_RNN = "ON"
# $env:MULTIRL_ENABLE_MCMC = "ON"
# $env:MULTIRL_LIBTORCH_DIR = "./build/_deps/libtorch-src"
#
# python -m pip uninstall -y multiRL
# python -m pip install -e Python --no-build-isolation --verbose
#
# Restart the Python kernel after reinstalling. A running Python process keeps
# the old imported module and old _backend.pyd in memory.

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

# %%
# MLE estimator test
print("Running MLE test...")
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
    models=models[:1],
    settings=settings[:1],
    lowers=[[0, 0]],
    uppers=[[1, 5]],
    control={
        "n_draws": 50,
        "seed": 1004,
        "threads": 32,
        "algorithm": "GN_MLSL",
        "local_algorithm": "LN_BOBYQA",
        "maxeval": 10,
    },
)

# %%
# MAP estimator test
print("Running MAP test...")
map_fit = multiRL.rcv_d(
    estimator="map",
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
    models=models[:1],
    settings=settings[:1],
    lowers=[[0, 0]],
    uppers=[[1, 5]],
    control={
        "n_draws": 50,
        "seed": 1004,
        "threads": 32,
        "algorithm": "GN_MLSL",
        "local_algorithm": "LN_BOBYQA",
        "maxeval": 10,
    },
)

# %%
# ABC estimator test
print("Running ABC test...")
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
    models=models[:1],
    settings=settings[:1],
    lowers=[[0, 0]],
    uppers=[[1, 5]],
    control={
        "n_draws": 50,
        "seed": 1004,
        "threads": 32,
        "scope": "shared",
        "samples": 10,
        "tol": 0.5,
        "method": "rejection",
        "reduction": "none",
        "print_level": 0,
    },
)

# %%
# MCMC estimator test
print("Running MCMC test...")
mcmc = multiRL.rcv_d(
    estimator="mcmc",
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
    models=models[:1],
    settings=settings[:1],
    lowers=[[0, 0]],
    uppers=[[1, 5]],
    control={
        "n_draws": 50,
        "seed": 1004,
        "threads": 32,
        "chains": 2,
        "samples": 10,
        "warmup": 5,
    },
)

# %%
# RNN estimator test (LibTorch CPU vs GPU & architectures/losses)
print("Running RNN tests...")

# Verify different combinations of layers and loss functions
test_configs = [
    {"layer": "gru", "loss": "mse"},
    {"layer": "lstm", "loss": "nll"},
    {"layer": "bigru", "loss": "qrl"},
    {"layer": "bilstm", "loss": "mdn"},
]

for config in test_configs:
    layer = config["layer"]
    loss = config["loss"]
    print(f"  Testing RNN (layer={layer}, loss={loss})...")
    rnn_res = multiRL.rcv_d(
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
        models=models[:1],
        settings=settings[:1],
        lowers=[[0, 0]],
        uppers=[[1, 5]],
        control={
            "n_draws": 20,
            "epochs": 2,
            "units": 8,
            "layer": layer,
            "loss": loss,
            "device": "cpu",
            "scope": "shared",
        },
    )
    print(f"    Estimator info: {rnn_res['estimator']}")

print("All Python tests completed successfully!")
