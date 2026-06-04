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
    policy="off",
    name="TD",
    mode="fitting",
):
    estimator = _modify_fit_p_estimator(estimator)
    control = _modify_fit_p_control(control)

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
        policy=policy,
        name=name,
        mode=mode,
    )
    return _tag_fit_p_result(result, estimator, control["scope"])


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
