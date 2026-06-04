"""Shared helpers for model-free built-in model specifications."""


def _model_free_builtin(name, free, params, priors, lower, upper, settings):
    if params is None:
        params = {
            "free": dict(free),
            "fixed": _model_free_default_fixed(free.keys()),
            "constant": _model_free_default_constant(free.keys()),
        }
    if priors is None:
        priors = _model_free_default_priors(free.keys())
    if lower is None:
        lower = _model_free_default_lower(free.keys())
    if upper is None:
        upper = _model_free_default_upper(free.keys())
    if settings is None:
        settings = {}

    out_settings = {"name": name}
    out_settings.update(settings)

    return {
        "model": name,
        "process": "process_model_free",
        "params": params,
        "priors": priors,
        "lower": lower,
        "upper": upper,
        "settings": out_settings,
    }


def _model_free_default_fixed(free_names):
    values = {
        "gamma": 1,
        "delta": 0.1,
        "epsilon": float("nan"),
        "zeta": 0,
    }
    return {
        name: value
        for name, value in values.items()
        if name not in free_names
    }


def _model_free_default_constant(free_names):
    values = {
        "seed": 123,
        "L": float("nan"),
        "penalty": 1,
        "Q0": float("nan"),
        "reset": float("nan"),
        "lapse": 0.01,
        "threshold": 1,
        "bonus": 0,
        "weight": 1,
        "capacity": 0,
        "sticky": 0,
    }
    return {
        name: value
        for name, value in values.items()
        if name not in free_names
    }


def _model_free_default_priors(names):
    out = {}
    for name in names:
        if name == "beta":
            out[name] = {"type": "exponential", "rate": 1}
        else:
            out[name] = {"type": "beta", "shape1": 2, "shape2": 2}
    return out


def _model_free_default_lower(names):
    return {name: 0 for name in names}


def _model_free_default_upper(names):
    return {
        name: 5 if name == "beta" else 1
        for name in names
    }
