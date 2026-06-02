"""Internal MAP estimator for the Python multiRL frontend."""

from . import _shell_run_m as _cpp_estimate_map
from ._shell_run_m import _modify_request


def estimate_map(
    data=None,
    id=None,
    colnames=None,
    behrule=None,
    funcs=None,
    params=None,
    priors=None,
    settings=None,
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
        estimate="MAP",
    )

    return _cpp_estimate_map.estimate_map(
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
        mle_maxeval=int(control.get("mle_maxeval", 10000)),
        map_maxiter=int(control.get("map_maxiter", 10)),
        map_tol=float(control.get("map_tol", 1e-3)),
        map_patience=int(control.get("map_patience", 10)),
        algorithm=str(control.get("algorithm", "GN_MLSL")),
        local_algorithm=str(
            control.get("local_algorithm", "LN_BOBYQA")
        ),
        xtol_rel=float(control.get("xtol_rel", 1e-6)),
        local_xtol_rel=float(control.get("local_xtol_rel", 1e-8)),
        seed=int(control.get("seed", 1004)),
    )
