"""RNN estimator implemented by the optional C++ LibTorch backend."""

from . import _backend
from ._estimate_mle import _modify_bounds
from ._shell_run_m import _modify_request


def estimate_rnn(
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
    model=None,
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
    if model is not None:
        raise ValueError(
            "estimate_rnn no longer accepts a Keras model object. "
            "Use the C++ LibTorch backend controls instead."
        )
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
        generate=generate,
        name=name,
        mode=mode,
        estimate="RNN",
    )
    request["settings"]["generate"] = True

    rnn_control = _modify_rnn_control(control)
    free_names_list = request["free_names"]
    lower_bounds = _modify_bounds(lower, free_names_list, -float("inf"))
    upper_bounds = _modify_bounds(upper, free_names_list, float("inf"))

    cpp_result = _backend.estimate_rnn_data(
        object=request["object"],
        reward=request["reward"],
        action=request["action"],
        block=request["block"],
        trial=request["trial"],
        subid=request["subid"],
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
        generate=request["settings"]["generate"],
        name=request["settings"]["name"],
        mode=request["settings"]["mode"],
        n_draws=int(rnn_control["n_draws"]),
        seed=int(rnn_control["seed"]),
        threads=int(rnn_control["threads"]),
        epochs=int(rnn_control["epochs"]),
        batch_size=int(rnn_control["batch_size"]),
        units=int(rnn_control["units"]),
        layers=int(rnn_control["layers"]),
        dropout=float(rnn_control["dropout"]),
        learning_rate=float(rnn_control["learning_rate"]),
        layer=str(rnn_control["layer"]),
        loss=str(rnn_control["loss"]),
        interop_threads=int(rnn_control["interop_threads"]),
        verbose=int(rnn_control["verbose"]),
        device=str(rnn_control["device"]),
        scope=str(rnn_control["scope"]),
        subject_embedding_size=int(rnn_control["subject_embedding_size"]),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return {
        "input": {
            "data": data,
            "colnames": colnames,
            "behrule": request["behrule"],
            "funcs": funcs,
            "params": params,
            "priors": priors,
            "settings": request["settings"],
            "lower": lower,
            "upper": upper,
            "control": rnn_control,
            "features": {
                "object": request["object"],
                "reward": request["reward"],
                "action": request["action"],
                "block": request["block"],
                "trial": request["trial"],
                "subid": request["subid"],
            },
        },
        "fit": cpp_result["fit"],
        "estimator": cpp_result["estimator"],
        "diagnostics": cpp_result["diagnostics"],
    }


def _modify_rnn_control(control):
    out = {
        "n_draws": 1000,
        "epochs": 20,
        "batch_size": 32,
        "validation_split": 0.0,
        "units": 32,
        "layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "seed": 123,
        "threads": 0,
        "interop_threads": 0,
        "backend": "torch",
        "layer": "gru",
        "loss": "mse",
        "verbose": 0,
        "device": "cpu",
        "scope": "individual",
        "subject_embedding_size": 8,
    }
    out.update(control)
    out["n_draws"] = int(out["n_draws"])
    out["epochs"] = int(out["epochs"])
    out["batch_size"] = int(out["batch_size"])
    out["validation_split"] = float(out["validation_split"])
    out["units"] = int(out["units"])
    out["layers"] = int(out["layers"])
    out["dropout"] = float(out["dropout"])
    out["learning_rate"] = float(out["learning_rate"])
    out["seed"] = int(out["seed"])
    out["threads"] = int(out["threads"])
    out["interop_threads"] = int(out["interop_threads"])
    out["backend"] = str(out["backend"]).lower()
    out["layer"] = str(out["layer"]).lower()
    out["loss"] = str(out["loss"]).lower()
    out["verbose"] = int(out["verbose"])
    out["device"] = str(out["device"]).lower()
    out["scope"] = str(out["scope"]).lower()
    out["subject_embedding_size"] = int(out["subject_embedding_size"])
    return out
