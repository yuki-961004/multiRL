"""RNN estimator implemented at the Python wrapper level."""

import math

from ._sampler import task_sampler
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
        estimate="RNN",
    )

    rnn_control = _modify_rnn_control(control)
    keras = _import_keras()
    numpy = _import_numpy()
    _set_seed(keras, rnn_control["seed"])

    sampler = task_sampler(
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
        priors=priors,
        policy=request["settings"]["policy"],
        name=request["settings"]["name"],
        mode=request["settings"]["mode"],
        lower=lower,
        upper=upper,
        control={
            "n_draws": rnn_control["n_draws"],
            "seed": rnn_control["seed"],
            "threads": rnn_control["threads"],
        },
    )

    X, y = _xy_from_sampler(
        rows=sampler["data"],
        cue=request["behrule"]["cue"],
        parameter_names=request["free_names"],
        numpy=numpy,
    )
    observed = _observed_sequence(
        request=request,
        numpy=numpy,
    )

    model_object = model
    if model_object is None:
        model_object = _default_model(
            keras=keras,
            n_features=int(X.shape[2]),
            n_params=len(request["free_names"]),
            control=rnn_control,
        )

    history = model_object.fit(
        X,
        y,
        epochs=rnn_control["epochs"],
        batch_size=rnn_control["batch_size"],
        validation_split=rnn_control["validation_split"],
        verbose=rnn_control["verbose"],
    )

    prediction = model_object.predict(
        observed,
        verbose=0,
    )
    estimates = prediction[0].tolist()

    fit = {"subid": "1", "status": 1}
    for index, name_value in enumerate(request["free_names"]):
        fit[name_value] = float(estimates[index])

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
            },
        },
        "fit": fit,
        "estimator": {
            "name": "RNN",
            "framework": "keras",
            "backend": _keras_backend(keras),
            "architecture": rnn_control["model_type"],
            "model": model_object,
            "control": rnn_control,
        },
        "diagnostics": {
            "training": _history_dict(history),
            "sampler": sampler["metadata"],
            "model": {
                "n_features": int(X.shape[2]),
                "n_sequences": int(X.shape[0]),
                "n_trials": int(X.shape[1]),
            },
        },
    }


def _modify_rnn_control(control):
    out = {
        "n_draws": 1000,
        "epochs": 20,
        "batch_size": 32,
        "validation_split": 0.2,
        "units": 32,
        "layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "seed": 123,
        "threads": 0,
        "model_type": "gru",
        "verbose": 0,
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
    out["model_type"] = str(out["model_type"]).lower()
    out["verbose"] = int(out["verbose"])
    return out


def _import_keras():
    try:
        import keras
    except ImportError as error:
        raise ImportError(
            "estimate_rnn requires Python Keras/keras3. "
            "Please install keras in the Python environment used by multiRL."
        ) from error
    return keras


def _import_numpy():
    try:
        import numpy
    except ImportError as error:
        raise ImportError(
            "estimate_rnn requires numpy in the Python environment used "
            "by multiRL."
        ) from error
    return numpy


def _set_seed(keras, seed):
    if hasattr(keras, "utils") and hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)


def _keras_backend(keras):
    if hasattr(keras, "backend") and hasattr(keras.backend, "backend"):
        return str(keras.backend.backend())
    return "unknown"


def _default_model(keras, n_features, n_params, control):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(None, n_features)))
    model.add(keras.layers.Masking(mask_value=0.0))

    for layer_index in range(max(control["layers"], 1)):
        return_sequences = layer_index < control["layers"] - 1
        if control["model_type"] == "lstm":
            model.add(
                keras.layers.LSTM(
                    control["units"],
                    return_sequences=return_sequences,
                )
            )
        elif control["model_type"] == "simple_rnn":
            model.add(
                keras.layers.SimpleRNN(
                    control["units"],
                    return_sequences=return_sequences,
                )
            )
        else:
            model.add(
                keras.layers.GRU(
                    control["units"],
                    return_sequences=return_sequences,
                )
            )

    if control["dropout"] > 0.0:
        model.add(keras.layers.Dropout(control["dropout"]))

    model.add(keras.layers.Dense(n_params, activation="linear"))
    optimizer = keras.optimizers.Adam(
        learning_rate=control["learning_rate"]
    )
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model


def _xy_from_sampler(rows, cue, parameter_names, numpy):
    groups = {}
    for row in rows:
        key = (row["draw"], row["subid"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    sequences = []
    targets = []
    for key in sorted(groups):
        trial_rows = sorted(groups[key], key=lambda value: value["trial"])
        sequences.append(
            [_row_features(row, cue, "simulation") for row in trial_rows]
        )
        targets.append(
            [float(trial_rows[0][name]) for name in parameter_names]
        )

    X = _pad_sequences(sequences, numpy)
    y = numpy.asarray(targets, dtype="float32")
    return X, y


def _observed_sequence(request, numpy):
    sequence = []
    for index, action_value in enumerate(request["action"]):
        row = {
            "action": action_value,
            "simulation": action_value,
            "reward": _chosen_reward(
                state=request["object"][index],
                reward=request["reward"][index],
                action=action_value,
            ),
            "block": request["block"][index],
            "trial": request["trial"][index],
        }
        for cue_value in request["behrule"]["cue"]:
            row["prob_" + cue_value] = 0.0
        sequence.append(_row_features(row, request["behrule"]["cue"], "action"))
    return _pad_sequences([sequence], numpy)


def _chosen_reward(state, reward, action):
    for index, state_value in enumerate(state):
        elements = str(state_value).split("_")
        if str(action) in elements and index < len(reward):
            return float(reward[index])
    return 0.0


def _row_features(row, cue, action_key):
    action_code = _action_code(row.get(action_key, ""), cue)
    values = [
        action_code,
        _safe_float(row.get("reward", 0.0)),
        _safe_float(row.get("block", 0.0)),
        _safe_float(row.get("trial", 0.0)),
    ]
    for cue_value in cue:
        values.append(_safe_float(row.get("prob_" + cue_value, 0.0)))
    return values


def _action_code(value, cue):
    if value in cue:
        return float(cue.index(value) + 1)
    return 0.0


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def _pad_sequences(sequences, numpy):
    n_rows = len(sequences)
    n_trials = max(len(sequence) for sequence in sequences)
    n_features = len(sequences[0][0])
    out = numpy.zeros((n_rows, n_trials, n_features), dtype="float32")
    for row, sequence in enumerate(sequences):
        out[row, :len(sequence), :] = numpy.asarray(
            sequence,
            dtype="float32",
        )
    return out


def _history_dict(history):
    out = {}
    for key, values in history.history.items():
        out[key] = [float(value) for value in values]
    return out
