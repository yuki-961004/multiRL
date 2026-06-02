"""Internal shell for the Python multiRL frontend."""

from . import _shell_run_m as _cpp_shell_run_m
from ._modify_behrule import _modify_behrule
from ._modify_colnames import _modify_colnames
from ._modify_features import _modify_features
from ._modify_params import _modify_params
from ._modify_priors import _modify_priors
from ._modify_settings import _modify_settings


def run_m(
    data=None,
    id=None,
    colnames=None,
    behrule=None,
    funcs=None,
    params=None,
    priors=None,
    settings=None,
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
    return _shell_run_m(
        data=data,
        id=id,
        colnames=colnames,
        behrule=behrule,
        funcs=funcs,
        params=params,
        priors=priors,
        settings=settings,
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
        estimate=estimate,
    )


def _shell_run_m(
    data,
    id,
    colnames,
    behrule,
    funcs,
    params,
    priors,
    settings,
    object,
    reward,
    action,
    block,
    trial,
    cue,
    rsp,
    free_names,
    system,
    policy,
    name,
    mode,
    estimate,
):
    request = _modify_request(
        data=data,
        id=id,
        colnames=colnames,
        behrule=behrule,
        funcs=funcs,
        params=params,
        priors=priors,
        settings=settings,
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
        estimate=estimate,
    )

    return _cpp_shell_run_m.shell_run_m(
        object=request["object"],
        reward=request["reward"],
        action=request["action"],
        block=request["block"],
        trial=request["trial"],
        cue=request["behrule"]["cue"],
        rsp=request["behrule"]["rsp"],
        params=request["params"],
        free_names=request["free_names"],
        system=request["settings"]["system"],
        prior_names=request["priors"]["name"],
        prior_types=request["priors"]["type"],
        prior_param1=request["priors"]["param1"],
        prior_param2=request["priors"]["param2"],
        prior_active=request["priors"]["active"],
        policy=request["settings"]["policy"],
        name=request["settings"]["name"],
        mode=request["settings"]["mode"],
        estimate=request["settings"]["estimate"],
    )


def _modify_request(
    data,
    id,
    colnames,
    behrule,
    funcs,
    params,
    priors,
    settings,
    object,
    reward,
    action,
    block,
    trial,
    cue,
    rsp,
    free_names,
    system,
    policy,
    name,
    mode,
    estimate,
):
    settings = _modify_settings(
        settings=settings,
        system=system,
        policy=policy,
        name=name,
        mode=mode,
        estimate=estimate,
    )
    behrule = _modify_behrule(
        behrule=behrule,
        cue=cue,
        rsp=rsp,
    )
    flat_params, free_names = _modify_params(
        params=params,
        free_names=free_names,
    )
    prior_spec = _modify_priors(
        priors=priors,
        free_names=free_names,
    )

    if data is not None:
        colnames = _modify_colnames(data=data, colnames=colnames)
        features = _modify_features(
            data=data,
            id=id,
            colnames=colnames,
        )
        object = features["object"]
        reward = features["reward"]
        action = features["action"]
        block = features["block"]
        trial = features["trial"]

    return {
        "object": object,
        "reward": reward,
        "action": action,
        "block": block,
        "trial": trial,
        "behrule": behrule,
        "params": flat_params,
        "free_names": free_names,
        "priors": prior_spec,
        "settings": settings,
    }
