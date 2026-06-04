"""RNN estimator implemented by Python Keras frontends."""

import math
import os

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
    keras = _import_keras(rnn_control)
    numpy = _import_numpy()
    _set_seed(keras, rnn_control["seed"])

    sampler = _estimate_rnn_data(
        request=request,
        priors=priors,
        lower=lower,
        upper=upper,
        control=rnn_control,
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
            "framework": rnn_control["backend"],
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


def _estimate_rnn_data(request, priors, lower, upper, control):
    free_names_list = request["free_names"]
    lower_bounds = _modify_bounds(lower, free_names_list, -float("inf"))
    upper_bounds = _modify_bounds(upper, free_names_list, float("inf"))

    cpp_result = _backend.estimate_rnn_data(
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
        n_draws=int(control["n_draws"]),
        seed=int(control["seed"]),
        threads=int(control["threads"]),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return {
        "input": {
            "behrule": request["behrule"],
            "params": request["params"],
            "priors": priors,
            "settings": request["settings"],
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
        "backend": "keras3",
        "keras_backend": "tensorflow",
        "model_type": "gru",
        "layer": None,
        "loss": "mse",
        "penalty": 0.0,
        "regularizer": "none",
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
    out["backend"] = str(out["backend"]).lower()
    out["keras_backend"] = str(out["keras_backend"]).lower()
    if out["layer"] is None:
        out["layer"] = out["model_type"]
    out["model_type"] = str(out["layer"]).lower()
    out["loss"] = str(out["loss"]).lower()
    out["penalty"] = float(out["penalty"])
    out["regularizer"] = str(out["regularizer"]).lower()
    out["verbose"] = int(out["verbose"])
    return out


def _import_keras(control):
    if control["backend"] in ["keras", "keras2", "tensorflow", "tf.keras"]:
        try:
            import tensorflow
        except ImportError as error:
            raise ImportError(
                "estimate_rnn backend='keras' requires tensorflow.keras."
            ) from error
        return tensorflow.keras

    os.environ.setdefault("KERAS_BACKEND", control["keras_backend"])
    try:
        import keras
    except ImportError as error:
        raise ImportError(
            "estimate_rnn backend='keras3' requires the keras package."
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
    if hasattr(keras.layers, "Masking"):
        model.add(keras.layers.Masking(mask_value=0.0))

    for layer_index in range(max(control["layers"], 1)):
        return_sequences = layer_index < control["layers"] - 1
        _add_recurrent_layer(
            keras=keras,
            model=model,
            model_type=control["model_type"],
            units=control["units"],
            return_sequences=return_sequences,
        )

    model.add(
        keras.layers.Dense(
            max(int(control["units"] / 2), 1),
            activation="relu",
            kernel_regularizer=_regularizer(keras, control),
        )
    )

    if control["dropout"] > 0.0:
        model.add(keras.layers.Dropout(control["dropout"]))

    model.add(
        keras.layers.Dense(
            _output_units(n_params, control["loss"]),
            activation="linear",
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=control["learning_rate"]
        ),
        loss=_loss(keras, n_params, control["loss"]),
        metrics=_metrics(control["loss"]),
    )
    return model


def _add_recurrent_layer(
    keras,
    model,
    model_type,
    units,
    return_sequences,
):
    if model_type in ["rnn", "simple_rnn", "simplernn"]:
        model.add(
            keras.layers.SimpleRNN(
                units,
                return_sequences=return_sequences,
            )
        )
    elif model_type == "lstm":
        model.add(
            keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
            )
        )
    elif model_type in ["birnn", "bisimple_rnn"]:
        model.add(
            keras.layers.Bidirectional(
                keras.layers.SimpleRNN(
                    units,
                    return_sequences=return_sequences,
                )
            )
        )
    elif model_type == "bigru":
        model.add(
            keras.layers.Bidirectional(
                keras.layers.GRU(
                    units,
                    return_sequences=return_sequences,
                )
            )
        )
    elif model_type == "bilstm":
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                )
            )
        )
    else:
        model.add(
            keras.layers.GRU(
                units,
                return_sequences=return_sequences,
            )
        )


def _regularizer(keras, control):
    penalty = control["penalty"]
    if penalty <= 0.0:
        return None
    if control["regularizer"] == "l1":
        return keras.regularizers.l1(penalty)
    if control["regularizer"] == "l2":
        return keras.regularizers.l2(penalty)
    if control["regularizer"] in ["l1_l2", "l12", "12"]:
        return keras.regularizers.l1_l2(l1=penalty, l2=penalty)
    return None


def _output_units(n_params, loss):
    if loss == "nll":
        return n_params * 2
    if loss == "qrl":
        return n_params * 3
    if loss == "mdn":
        return n_params * 9
    return n_params


def _loss(keras, n_params, loss):
    if loss == "mae":
        return "mean_absolute_error"
    if loss in ["hbr", "huber"]:
        return "huber_loss"
    if loss == "nll":
        return _gaussian_nll(keras, n_params)
    if loss == "qrl":
        return _quantile_loss(keras, n_params)
    if loss == "mdn":
        return _mixture_density_loss(keras, n_params)
    return "mean_squared_error"


def _metrics(loss):
    if loss == "mae":
        return ["mean_absolute_error"]
    if loss in ["hbr", "huber"]:
        return ["huber_loss"]
    if loss in ["nll", "qrl", "mdn"]:
        return None
    return ["mean_squared_error"]


def _ops(keras):
    if hasattr(keras, "ops"):
        return keras.ops
    return keras.backend


def _gaussian_nll(keras, n_params):
    ops = _ops(keras)

    def loss_func(y_true, y_pred):
        mu = y_pred[:, 0:n_params]
        log_var = y_pred[:, n_params:(2 * n_params)]
        precision = ops.exp(-log_var)
        value = 0.5 * precision * ops.square(y_true - mu) + 0.5 * log_var
        return ops.mean(ops.sum(value, axis=-1))

    return loss_func


def _quantile_loss(keras, n_params):
    ops = _ops(keras)
    q_values = [0.05, 0.50, 0.95]

    def loss_func(y_true, y_pred):
        total = 0.0
        for index, q_value in enumerate(q_values):
            start = index * n_params
            stop = (index + 1) * n_params
            pred = y_pred[:, start:stop]
            err = y_true - pred
            total = total + ops.mean(
                ops.maximum(q_value * err, (q_value - 1.0) * err),
                axis=-1,
            )
        return total

    return loss_func


def _mixture_density_loss(keras, n_params):
    ops = _ops(keras)
    n_mix = 3

    def loss_func(y_true, y_pred):
        pi_logits = ops.reshape(
            y_pred[:, 0:(n_params * n_mix)],
            (-1, n_params, n_mix),
        )
        mu = ops.reshape(
            y_pred[:, (n_params * n_mix):(2 * n_params * n_mix)],
            (-1, n_params, n_mix),
        )
        log_var = ops.reshape(
            y_pred[:, (2 * n_params * n_mix):(3 * n_params * n_mix)],
            (-1, n_params, n_mix),
        )
        mix_weights = ops.softmax(pi_logits, axis=-1)
        y_true_exp = ops.expand_dims(y_true, axis=-1)
        log_prob = (
            -0.5 * math.log(2.0 * math.pi)
            - 0.5 * log_var
            - 0.5 * ops.square(y_true_exp - mu) * ops.exp(-log_var)
        )
        weighted = ops.log(mix_weights + 1e-8) + log_prob
        if hasattr(ops, "logsumexp"):
            log_mix_prob = ops.logsumexp(weighted, axis=-1)
        else:
            max_value = ops.max(weighted, axis=-1, keepdims=True)
            log_mix_prob = (
                ops.log(ops.sum(ops.exp(weighted - max_value), axis=-1)) +
                ops.squeeze(max_value, axis=-1)
            )
        return ops.mean(-ops.sum(log_mix_prob, axis=-1))

    return loss_func


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
