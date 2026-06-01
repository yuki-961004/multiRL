"""Internal prior conversion helpers."""


def _modify_priors(priors, free_names):
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
        prior_type = _modify_prior_type(prior["type"])
        names.append(str(name))
        types.append(prior_type)
        param1.append(_modify_prior_arg1(prior, prior_type))
        param2.append(_modify_prior_arg2(prior, prior_type))

    return {
        "active": True,
        "name": names,
        "type": types,
        "param1": param1,
        "param2": param2,
    }


def _modify_prior_type(prior_type):
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


def _modify_prior_arg1(prior, prior_type):
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

    return _modify_prior_named_float(prior, keys)


def _modify_prior_arg2(prior, prior_type):
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

    return _modify_prior_named_float(prior, keys)


def _modify_prior_named_float(values, keys):
    for key in keys:
        if key in values:
            return float(values[key])

    raise ValueError("Structured prior is missing a required argument.")
