"""Fitting workflow shell for the Python multiRL frontend."""

from ._estimate_abc import estimate_abc
from ._estimate_map import estimate_map
from ._estimate_mcmc import estimate_mcmc
from ._estimate_mle import estimate_mle
from ._estimate_rnn import estimate_rnn


def fit_p(
    data=None,
    estimator="mle",
    id=None,
    colnames=None,
    behrule=None,
    models=None,
    funcs=None,
    params=None,
    priors=None,
    settings=None,
    lower=None,
    upper=None,
    control=None,
    object=None,
    reward=None,
    action=None,
    block=None,
    trial=None,
    cue=None,
    rsp=None,
    free_names=None,
    system=None,
    generate=False,
    name="TD",
    mode="fitting",
):
    estimator = _modify_fit_p_estimator(estimator)
    control = _modify_fit_p_control(control)
    if models is not None:
        return _fit_p_models(
            data=data,
            estimator=estimator,
            id=id,
            colnames=colnames,
            behrule=behrule,
            models=models,
            funcs=funcs,
            params=params,
            priors=priors,
            settings=settings,
            lower=lower,
            upper=upper,
            control=control,
        )

    dispatch = {
        "mle": estimate_mle,
        "map": estimate_map,
        "mcmc": estimate_mcmc,
        "abc": estimate_abc,
        "rnn": estimate_rnn,
    }
    result = dispatch[estimator](
        data=data,
        id=id,
        colnames=colnames,
        behrule=behrule,
        funcs=funcs,
        params=params,
        priors=priors,
        settings=settings,
        lower=lower,
        upper=upper,
        control=control,
        object=object,
        reward=reward,
        action=action,
        block=block,
        trial=trial,
        cue=cue,
        rsp=rsp,
        free_names=free_names,
        system=system,
        generate=generate,
        name=name,
        mode=mode,
    )
    return _tag_fit_p_result(result, estimator, control["scope"])


def _fit_p_models(
    data,
    estimator,
    id,
    colnames,
    behrule,
    models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper,
    control,
):
    specs = _modify_fit_p_models(
        models=models,
        funcs=funcs,
        params=params,
        priors=priors,
        settings=settings,
        lower=lower,
        upper=upper,
    )
    raw = {}
    fit = []

    for index, spec in enumerate(specs, start=1):
        model = _fit_p_model_name(spec, index)
        model_id = model + "_" + str(index)
        local_settings = dict(spec.get("settings") or {})
        local_settings["name"] = model
        result = fit_p(
            data=data,
            estimator=estimator,
            id=id,
            colnames=colnames,
            behrule=behrule,
            funcs=spec.get("funcs"),
            params=spec.get("params"),
            priors=spec.get("priors"),
            settings=local_settings,
            lower=spec.get("lower"),
            upper=spec.get("upper"),
            control=control,
        )
        if isinstance(result["fit"], list):
            rows = result["fit"]
        else:
            rows = [result["fit"]]
        for row in rows:
            row["model"] = model
            row["model_id"] = model_id
            fit.append(row)
        raw[model_id] = result

    return {
        "input": {
            "data": data,
            "colnames": colnames,
            "behrule": behrule,
            "models": specs,
            "estimator": estimator,
            "control": control,
        },
        "fit": fit,
        "raw": raw,
        "estimator": {
            "name": estimator.upper(),
            "shell": "fit_p",
        },
        "diagnostics": {
            "n_models": len(specs),
            "model_id": list(raw.keys()),
        },
    }


def _modify_fit_p_models(models, funcs, params, priors, settings, lower, upper):
    if not isinstance(models, list):
        models = [models]
    out = []
    for spec in models:
        if callable(spec):
            spec = spec()
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            spec = {"model": spec}
        out.append(
            {
                "model": spec.get("model"),
                "process": spec.get("process"),
                "funcs": spec.get("funcs", funcs),
                "params": spec.get("params", params),
                "priors": spec.get("priors", priors),
                "settings": spec.get("settings", settings),
                "lower": spec.get("lower", lower),
                "upper": spec.get("upper", upper),
            }
        )
    return out


def _fit_p_model_name(spec, index):
    settings = spec.get("settings") or {}
    if settings.get("name") is not None:
        return str(settings["name"])
    if spec.get("model") is not None:
        return str(spec["model"])
    return "model_" + str(index)


def _modify_fit_p_estimator(estimator):
    estimator = str(estimator).lower()
    supported = ["mle", "map", "mcmc", "abc", "rnn"]
    if estimator not in supported:
        raise ValueError(
            "Unknown estimator. Supported estimators in fit_p v0.5.0-12 "
            "are: " + ", ".join(supported) + "."
        )
    return estimator


def _modify_fit_p_control(control):
    if control is None:
        control = {}
    out = dict(control)
    out["scope"] = str(out.get("scope", "individual")).lower()
    supported = ["individual", "shared", "universal"]
    if out["scope"] not in supported:
        raise ValueError(
            "Unknown scope. Supported scopes in fit_p v0.5.0-12 are: "
            + ", ".join(supported) + "."
        )
    return out


def _tag_fit_p_result(result, estimator, scope):
    result["estimator"]["shell"] = "fit_p"
    result["estimator"]["scope"] = scope
    result["input"]["control"]["scope"] = scope
    result["input"]["settings"]["estimate"] = estimator.upper()
    return result
