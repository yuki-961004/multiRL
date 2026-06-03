# Install from the Python package directory before running:
# python -m pip install -e Python

# %%
import time

import multiRL
import pandas

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

iterations = 100
start_time = time.perf_counter()

for iteration in range(iterations):
    multiRL.run_m(
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

elapsed = time.perf_counter() - start_time
print(f"multiRL run_m 100 iterations elapsed: {elapsed:.4f}s")

# %%
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

mle = multiRL.estimate_mle(
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
    control={
        "algorithm": "NLOPT_LN_BOBYQA",
        "local_algorithm": "NLOPT_LN_BOBYQA",
        "maxeval": 100,
        "xtol_rel": 1e-8,
    },
)

print(mle["fit"])

# %%
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

map_fit = multiRL.estimate_map(
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
    priors={
        "alpha": {"type": "beta", "shape1": 2, "shape2": 2},
        "beta": {"type": "exp", "rate": 1},
    },
    control={
        "algorithm": "NLOPT_LN_BOBYQA",
        "local_algorithm": "NLOPT_LN_BOBYQA",
        "mle_maxeval": 100,
        "map_maxiter": 10,
        "map_tol": 1e-3,
        "map_patience": 10,
        "xtol_rel": 1e-8,
    },
)

print(map_fit["fit"])

# %%
# MCMC (requires Stan Math headers)
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

mcmc_result = multiRL.estimate_mcmc(
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
    priors={
        "alpha": {"type": "beta", "shape1": 2, "shape2": 2},
        "beta": {"type": "exp", "rate": 1},
    },
    control={
        "algorithm": "nuts",
        "warmup": 50,
        "samples": 50,
        "chains": 2,
        "thin": 1,
        "step_size": 0.05,
        "target_accept": 0.80,
        "max_tree_depth": 4,
        "seed": 1004,
    },
)

print(mcmc_result["fit"])

# %%
# ABC (uses the abcpp C++ backend)
import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

abc_result = multiRL.estimate_abc(
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
    lower=[0, 0],
    upper=[1, 1],
    control={
        "samples": 50,
        "tol": 0.2,
        "method": "rejection",
        "reduction": "none",
        "fake_block": 4,
        "seed": 1004,
        "threads": 1,
        "print_level": 0,
    },
)

print(abc_result["fit"])
print(abc_result["diagnostics"]["subjects"])

if abc_result["estimator"]["name"] != "ABC":
    raise RuntimeError("ABC estimator name is not ABC.")

if abc_result["estimator"]["backend"] != "abcpp":
    raise RuntimeError("ABC backend is not abcpp.")

if abc_result["diagnostics"]["subjects"][0]["fake_block"] != 4:
    raise RuntimeError("ABC fake_block was not reported correctly.")

if abc_result["diagnostics"]["subjects"][0]["n_blocks_used"] != 4:
    raise RuntimeError("ABC fake_block did not create four blocks.")
