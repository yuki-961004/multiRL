"""Task sampling helpers for the Python multiRL frontend."""

from . import _backend as _cpp_task_sampler
from ._estimate_mle import _modify_bounds
from ._shell_run_m import _modify_request


def task_sampler(
    data=None,
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
    if control is None:
        control = {}

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
        estimate="SAMPLER",
    )

    free_names_list = request["free_names"]
    lower_bounds = _modify_bounds(lower, free_names_list, -float("inf"))
    upper_bounds = _modify_bounds(upper, free_names_list, float("inf"))

    cpp_result = _cpp_task_sampler.task_sampler(
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
        n_draws=int(control.get("n_draws", 100)),
        seed=int(control.get("seed", 123)),
        threads=int(control.get("threads", 0)),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return {
        "input": {
            "data": data,
            "colnames": request.get("colnames", colnames),
            "behrule": request.get("behrule", behrule),
            "funcs": request.get("funcs", funcs),
            "params": params,
            "priors": priors,
            "settings": request.get("settings", settings),
            "lower": lower,
            "upper": upper,
            "control": control,
            "features": {
                "object": request["object"],
                "reward": request["reward"],
                "action": request["action"],
                "block": request["block"],
                "trial": request["trial"],
            },
        },
        "data": cpp_result["data"],
        "metadata": cpp_result["metadata"],
    }
