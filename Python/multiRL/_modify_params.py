"""Internal parameter conversion helpers."""


def _modify_params(params, free_names=None):
    if params is None:
        params = {}

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
