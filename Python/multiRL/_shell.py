"""Shell wrappers for the Python multiRL frontend."""

from . import _shell_run_m


def shell_run_m(
    object,
    reward,
    action,
    block,
    trial,
    cue,
    rsp,
    params,
    free_names=None,
    system=None,
    policy="off",
    name="TD",
    mode="fitting",
    estimate="MLE",
):
    if free_names is None:
        free_names = list(params.keys())

    if system is None:
        system = ["RL"]

    return _shell_run_m.shell_run_m(
        object=object,
        reward=reward,
        action=action,
        block=block,
        trial=trial,
        cue=cue,
        rsp=rsp,
        params=params,
        free_names=free_names,
        system=system,
        policy=policy,
        name=name,
        mode=mode,
        estimate=estimate,
    )


run_m = shell_run_m

