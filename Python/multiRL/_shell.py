"""Shell wrappers for the Python multiRL frontend."""

from . import _shell_run_m


def run_m(
    data=None,
    id=None,
    colnames=None,
    behrule=None,
    funcs=None,
    params=None,
    priors=None,
    settings=None,
    engine="Cpp",
    object=None,
    reward=None,
    action=None,
    block=None,
    trial=None,
    cue=None,
    rsp=None,
    free_names=None,
    system=None,
    policy="off",
    name="TD",
    mode="fitting",
    estimate="MLE",
):
    if engine != "Cpp":
        raise ValueError("multiRL currently supports engine = 'Cpp'.")

    if funcs is not None and len(funcs) > 0:
        raise ValueError("multiRL currently supports built-in C++ functions.")

    if settings is None:
        settings = {}

    if behrule is not None:
        cue = behrule.get("cue", cue)
        rsp = behrule.get("rsp", rsp)

    name = settings.get("name", name)
    mode = settings.get("mode", mode)
    estimate = settings.get("estimate", estimate)
    policy = settings.get("policy", policy)
    system = settings.get("system", system)

    if system is None:
        system = ["RL"]

    if params is None:
        params = {}

    flat_params, free_names = _standardize_params(params, free_names)
    prior_spec = _standardize_priors(priors, free_names)

    if data is not None:
        features = _features_from_data(
            data=data,
            id=id,
            colnames=colnames,
        )
        object = features["object"]
        reward = features["reward"]
        action = features["action"]
        block = features["block"]
        trial = features["trial"]

    object = _matrix_to_list(object)
    reward = _numeric_matrix_to_list(reward)
    action = _vector_to_list(action)
    block = [int(value) for value in _vector_to_list(block)]
    trial = [int(value) for value in _vector_to_list(trial)]
    cue = _vector_to_list(cue)
    rsp = _vector_to_list(rsp)

    return _shell_run_m.shell_run_m(
        object=object,
        reward=reward,
        action=action,
        block=block,
        trial=trial,
        cue=cue,
        rsp=rsp,
        params=flat_params,
        free_names=free_names,
        system=system,
        prior_names=prior_spec["name"],
        prior_types=prior_spec["type"],
        prior_param1=prior_spec["param1"],
        prior_param2=prior_spec["param2"],
        prior_active=prior_spec["active"],
        policy=policy,
        name=name,
        mode=mode,
        estimate=estimate,
    )


def _standardize_params(params, free_names):
    if not isinstance(params, dict):
        raise TypeError("params must be a dict.")

    if any(key in params for key in ("free", "fixed", "constant")):
        free = dict(params.get("free", {}))
        fixed = dict(params.get("fixed", {}))
        constant = dict(params.get("constant", {}))
        out = {}
        out.update(free)
        out.update(fixed)
        out.update(constant)

        if free_names is None:
            free_names = list(free.keys())
    else:
        out = dict(params)
        if free_names is None:
            free_names = list(out.keys())

    return _numeric_params(out), list(free_names)


def _numeric_params(params):
    return {
        str(name): float(value)
        for name, value in params.items()
        if value is not None
    }


def _standardize_priors(priors, free_names):
    if priors is None or len(priors) == 0:
        return {
            "active": False,
            "name": [],
            "type": [],
            "param1": [],
            "param2": [],
        }

    if list(priors.keys()) != list(free_names):
        raise ValueError("priors names must match free parameter names.")

    names = []
    types = []
    param1 = []
    param2 = []

    for name, prior in priors.items():
        prior_type = _normalize_prior_type(prior["type"])
        names.append(str(name))
        types.append(prior_type)
        param1.append(_prior_arg1(prior, prior_type))
        param2.append(_prior_arg2(prior, prior_type))

    return {
        "active": True,
        "name": names,
        "type": types,
        "param1": param1,
        "param2": param2,
    }


def _normalize_prior_type(prior_type):
    prior_type = str(prior_type).lower()
    aliases = {
        "norm": "normal",
        "normal": "normal",
        "unif": "uniform",
        "uniform": "uniform",
        "lnorm": "lognormal",
        "lognormal": "lognormal",
        "cauchy": "cauchy",
        "beta": "beta",
        "exp": "exponential",
        "exponential": "exponential",
        "none": "none",
    }

    if prior_type not in aliases:
        raise ValueError("Unsupported prior type.")

    return aliases[prior_type]


def _prior_arg1(prior, prior_type):
    keys = {
        "normal": ("mean", "mu", "param1"),
        "uniform": ("min", "lower", "param1"),
        "lognormal": ("meanlog", "mean", "param1"),
        "cauchy": ("location", "mean", "param1"),
        "beta": ("shape1", "alpha", "param1"),
        "exponential": ("rate", "lambda", "param1"),
        "none": (),
    }[prior_type]

    if prior_type == "none":
        return float("nan")

    return _named_float(prior, keys)


def _prior_arg2(prior, prior_type):
    keys = {
        "normal": ("sd", "sigma", "param2"),
        "uniform": ("max", "upper", "param2"),
        "lognormal": ("sdlog", "sd", "param2"),
        "cauchy": ("scale", "sd", "param2"),
        "beta": ("shape2", "beta", "param2"),
        "exponential": (),
        "none": (),
    }[prior_type]

    if prior_type in ("exponential", "none"):
        return float("nan")

    return _named_float(prior, keys)


def _named_float(values, keys):
    for key in keys:
        if key in values:
            return float(values[key])

    raise ValueError("Structured prior is missing a required argument.")


def _features_from_data(data, id, colnames):
    if colnames is None:
        colnames = {}

    subid_name = colnames.get("subid", "Subject")
    block_name = colnames.get("block", "Block")
    trial_name = colnames.get("trial", "Trial")
    action_name = colnames.get("action", "Action")

    if id is not None:
        ids = id if isinstance(id, (list, tuple, set)) else [id]
        data = data[data[subid_name].isin(ids)]

    object_names = colnames.get("object")
    reward_names = colnames.get("reward")

    if object_names is None:
        object_names = [
            name for name in list(data.columns)
            if str(name).startswith("Object_")
        ]

    if reward_names is None:
        reward_names = [
            name for name in list(data.columns)
            if str(name).startswith("Reward_")
        ]

    return {
        "object": data[object_names],
        "reward": data[reward_names],
        "action": data[action_name],
        "block": data[block_name],
        "trial": data[trial_name],
    }


def _vector_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing vector input.")

    if hasattr(value, "tolist"):
        value = value.tolist()

    return list(value)


def _matrix_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing matrix input.")

    if hasattr(value, "values"):
        value = value.values

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [
        [str(cell) for cell in row]
        for row in value
    ]


def _numeric_matrix_to_list(value):
    if value is None:
        raise ValueError("run_m received a missing numeric matrix input.")

    if hasattr(value, "values"):
        value = value.values

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [
        [float(cell) for cell in row]
        for row in value
    ]
