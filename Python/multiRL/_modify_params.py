"""Internal parameter conversion helpers."""


def _modify_params(params, free_names=None):
    if params is None:
        params = {}

    if not isinstance(params, dict):
        raise TypeError("params must be a dict.")

    default_flat = {
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
        "threshold": 1.0,
        "bonus": 0.0,
        "weight": 1.0,
        "capacity": 0.0,
        "sticky": 0.0,
    }

    out = dict(default_flat)

    if any(key in params for key in ("free", "fixed", "constant")):
        free = dict(params.get("free", {}))
        fixed = dict(params.get("fixed", {}))
        constant = dict(params.get("constant", {}))
        
        # update sequentially, prioritizing free parameters last so they override any duplicates
        out.update(fixed)
        out.update(constant)
        out.update(free)

        if free_names is None:
            free_names = list(free.keys())
    else:
        out.update(params)
        if free_names is None:
            free_names = list(params.keys())

    return _numeric_params(out), list(free_names)



def _numeric_params(params):
    return {
        str(name): float(value)
        for name, value in params.items()
        if value is not None
    }
