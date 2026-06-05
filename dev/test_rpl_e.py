import multiRL
import pandas


data = pandas.read_csv("data/TAB.csv")
data = data[data["Subject"] == 1]

result_fit = multiRL.fit_p(
    data=data,
    estimator="mle",
    id=1,
    colnames={
        "subid": "Subject",
        "block": "Block",
        "trial": "Trial",
        "object": ["L_choice", "R_choice"],
        "reward": ["L_reward", "R_reward"],
        "action": "Sub_Choose",
    },
    behrule={
        "cue": ["A", "B", "C", "D"],
        "rsp": ["A", "B", "C", "D"],
    },
    models=[
        multiRL.TD,
        multiRL.Utility,
    ],
    control={
        "algorithm": "LN_BOBYQA",
        "local_algorithm": "LN_BOBYQA",
        "maxeval": 10,
        "seed": 1004,
    },
)

model_names = {row["model"] for row in result_fit["fit"]}
if len(model_names) != 2:
    raise RuntimeError("fit_p did not fit two models.")

result_rpl = multiRL.rpl_e(result_fit, option={"plot": False})

if not isinstance(result_rpl, dict):
    raise RuntimeError("rpl_e did not return a dictionary.")

required = {"input", "replay", "plot_data", "diagnostics"}
if not required.issubset(result_rpl.keys()):
    raise RuntimeError("rpl_e output does not contain all required fields.")

if "Human" not in {row["model"] for row in result_rpl["plot_data"]}:
    raise RuntimeError("rpl_e plot_data does not include Human.")

if "model" not in {row["source"] for row in result_rpl["plot_data"]}:
    raise RuntimeError("rpl_e plot_data does not include model replay.")

if result_rpl["diagnostics"]["policy"] != "on":
    raise RuntimeError("rpl_e did not report policy = on.")

print(result_fit["fit"])
print(result_rpl["plot_data"][:6])
print("Python rpl_e smoke test passed.")

# %%
# rcv_d + rpl_e recovery plotting

import multiRL
import pandas

data = pandas.read_csv("data/TAB.csv")

result_rcv = multiRL.rcv_d(
    estimator="mle",
    data=data,
    id=1,
    colnames={
        "subid": "Subject",
        "block": "Block",
        "trial": "Trial",
        "object": ["L_choice", "R_choice"],
        "reward": ["L_reward", "R_reward"],
        "action": "Sub_Choose",
    },
    behrule={
        "cue": ["A", "B", "C", "D"],
        "rsp": ["A", "B", "C", "D"],
    },
    models=[
        multiRL.TD,
        multiRL.RSTD,
        multiRL.Utility,
    ],
    settings=[
        {"name": "TD"},
        {"name": "RSTD"},
        {"name": "Utility"},
    ],
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

if len(result_rcv["recovery"]) == 0:
    raise RuntimeError("rcv_d recovery is empty.")

if len(result_rcv["model_recovery"]) == 0:
    raise RuntimeError("rcv_d model_recovery is empty.")

result_rpl_rcv = multiRL.rpl_e(result_rcv, option={"plot": False})

if "recovery" not in result_rpl_rcv:
    raise RuntimeError("rpl_e rcv_d output missing recovery.")

if "model_recovery" not in result_rpl_rcv:
    raise RuntimeError("rpl_e rcv_d output missing model_recovery.")

print(result_rcv["recovery"][:3])
print("Python rpl_e rcv_d recovery test passed.")
