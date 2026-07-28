"""Recovery workflow shell for the Python multiRL frontend."""

from . import _backend as _cpp_shell_rcv_d
from ._estimate_mle import _modify_bounds
from ._shell_fit_p import fit_p
from ._shell_fit_p import _modify_fit_p_estimator
from ._shell_run_m import _modify_request
import pandas
import warnings


def rcv_d(
    data=None,
    estimator="abc",
    id=None,
    colnames=None,
    behrule=None,
    generating=None,
    candidates=None,
    models=None,
    funcs=None,
    params=None,
    priors=None,
    settings=None,
    lower=None,
    upper=None,
    lowers=None,
    uppers=None,
    control=None,
    **kwargs,
):
    if "fit_control" in kwargs and kwargs["fit_control"] is not None:
        if control is None:
            control = {}
        control = dict(control)
        control.update(kwargs["fit_control"])
        warnings.warn(
            "rcv_d() now uses control for both sampling and fitting settings.",
            stacklevel=2,
        )
    if "sampler_control" in kwargs and kwargs["sampler_control"] is not None:
        if control is None:
            control = {}
        control = dict(control)
        control.update(kwargs["sampler_control"])
        warnings.warn(
            "rcv_d() now uses control for both sampling and fitting settings.",
            stacklevel=2,
        )

    if lowers is not None:
        lower = lowers
    if uppers is not None:
        upper = uppers

    estimator = _modify_fit_p_estimator(estimator)
    control = _modify_rcv_d_control(
        control=control,
        estimator=estimator,
    )

    generating = _modify_rcv_d_models(
        models=generating,
        fallback_models=models,
        funcs=funcs,
        params=params,
        priors=priors,
        settings=settings,
        lower=lower,
        upper=upper,
    )
    if candidates is None:
        candidates = generating
    else:
        candidates = _modify_rcv_d_models(
            models=candidates,
            fallback_models=None,
            funcs=funcs,
            params=params,
            priors=priors,
            settings=settings,
            lower=lower,
            upper=upper,
        )

    simulation_parts = []
    truth_parts = []
    fit_parts = []
    raw_fits = {}

    for generating_index, generating_spec in enumerate(generating, start=1):
        generating_name = _rcv_d_model_name(
            spec=generating_spec,
            index=generating_index,
            prefix="generating",
        )
        simulated = _rcv_d_simulate_model(
            data=data,
            id=id,
            colnames=colnames,
            behrule=behrule,
            spec=generating_spec,
            name=generating_name,
            control=control,
        )
        simulation_parts.extend(simulated["simulation"])
        truth_parts.extend(simulated["truth"])

        fit_data = _rcv_d_fit_data(simulated["simulation"])
        for candidate_index, candidate_spec in enumerate(candidates, start=1):
            candidate_name = _rcv_d_model_name(
                spec=candidate_spec,
                index=candidate_index,
                prefix="candidate",
            )
            local_control = _rcv_d_fit_control(
                estimator=estimator,
                scope=control.get("scope"),
                control=control,
            )
            candidate_settings = dict(candidate_spec.get("settings") or {})
            candidate_settings["name"] = candidate_name

            fit_result = fit_p(
                data=fit_data,
                estimator=estimator,
                id=None,
                colnames=_rcv_d_colnames(),
                behrule=behrule,
                funcs=candidate_spec.get("funcs"),
                params=candidate_spec.get("params"),
                priors=candidate_spec.get("priors"),
                settings=candidate_settings,
                lower=candidate_spec.get("lower"),
                upper=candidate_spec.get("upper"),
                control=local_control,
            )

            key = (
                generating_name
                + "::"
                + candidate_name
            )
            raw_fits[key] = fit_result
            fit_parts.extend(
                _rcv_d_fit_table(
                    fit=fit_result.get("fit", {}),
                    generating_model=generating_name,
                    candidate_model=candidate_name,
                    estimator=estimator,
                )
            )

    recovery = _rcv_d_recovery_table(truth_parts, fit_parts, estimator)
    model_recovery = _rcv_d_model_recovery_table(fit_parts)

    return {
        "input": {
            "data": data,
            "colnames": colnames,
            "behrule": behrule,
            "generating": generating,
            "candidates": candidates,
            "estimator": estimator,
            "control": control,
        },
        "simulation": simulation_parts,
        "truth": truth_parts,
        "fit": fit_parts,
        "recovery": recovery,
        "model_recovery": model_recovery,
        "raw": raw_fits,
        "estimator": {
            "name": estimator.upper(),
            "shell": "rcv_d",
            "scope": control.get("scope"),
        },
        "diagnostics": {
            "sampler": {
                "n_draws": control["n_draws"],
                "seed": control["seed"],
                "threads": control["threads"],
            },
            "fit_p": {
                "n_fits": len(raw_fits),
                "estimator": estimator,
            },
            "recovery": {
                "n_generating": len(generating),
                "n_candidates": len(candidates),
            },
        },
    }


def _modify_rcv_d_models(
    models,
    fallback_models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper,
):
    if models is None:
        models = fallback_models
    if models is None:
        models = [{}]
    if not isinstance(models, list):
        models = [models]
    if len(models) == 0:
        models = [{}]

    out = []
    n_models = len(models)
    for index, spec in enumerate(models):
        if callable(spec):
            spec = spec()
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            spec = {"model": spec}
        indexed = _rcv_d_indexed_fallback(
            index=index,
            n_models=n_models,
            funcs=funcs,
            params=params,
            priors=priors,
            settings=settings,
            lower=lower,
            upper=upper,
        )
        out.append(
            {
                "model": spec.get("model"),
                "funcs": _rcv_d_value_override(
                    spec,
                    "funcs",
                    indexed["funcs"],
                ),
                "params": spec.get("params", indexed["params"]),
                "priors": _rcv_d_value_override(
                    spec,
                    "priors",
                    indexed["priors"],
                ),
                "settings": _rcv_d_value_override(
                    spec,
                    "settings",
                    indexed["settings"],
                ),
                "lower": _rcv_d_value_override(
                    spec,
                    "lower",
                    indexed["lower"],
                ),
                "upper": _rcv_d_value_override(
                    spec,
                    "upper",
                    indexed["upper"],
                ),
            }
        )
    return out


def _rcv_d_indexed_fallback(
    index,
    n_models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper,
):
    return {
        "funcs": _rcv_d_indexed_value(funcs, index, n_models),
        "params": _rcv_d_indexed_value(params, index, n_models),
        "priors": _rcv_d_indexed_value(priors, index, n_models),
        "settings": _rcv_d_indexed_value(settings, index, n_models),
        "lower": _rcv_d_indexed_value(lower, index, n_models),
        "upper": _rcv_d_indexed_value(upper, index, n_models),
    }


def _rcv_d_indexed_value(value, index, n_models):
    if not isinstance(value, list):
        return value
    if len(value) != n_models:
        return value
    return value[index]


def _rcv_d_value_override(spec, key, fallback):
    if fallback is not None:
        if not hasattr(fallback, "__len__") or len(fallback) > 0:
            return fallback
    return spec.get(key, fallback)


def _modify_rcv_d_control(control, estimator):
    if control is None:
        control = {}

    out = {
        "n_draws": 30,
        "seed": 123,
        "threads": 0,
        "scope": "shared" if estimator in ["abc", "rnn"] else None,
    }
    out.update(control)
    out["n_draws"] = int(out["n_draws"])
    out["seed"] = int(out["seed"])
    out["threads"] = int(out["threads"])
    if out.get("scope") is not None:
        out["scope"] = str(out["scope"]).lower()
    return out


def _rcv_d_model_name(spec, index, prefix):
    settings = spec.get("settings") or {}
    if settings.get("name") is not None:
        return str(settings["name"])
    return prefix + "_" + str(index)


def _rcv_d_simulate_model(
    data,
    id,
    colnames,
    behrule,
    spec,
    name,
    control,
):
    settings = dict(spec.get("settings") or {})
    settings["name"] = name
    settings["mode"] = "simulating"
    settings["generate"] = True

    request = _modify_request(
        data=data,
        id=id,
        colnames=colnames,
        behrule=behrule,
        funcs=spec.get("funcs"),
        params=spec.get("params"),
        priors=spec.get("priors"),
        settings=settings,
        object=None,
        reward=None,
        action=None,
        block=None,
        trial=None,
        cue=None,
        rsp=None,
        free_names=None,
        system=None,
        generate=True,
        name=name,
        mode="simulating",
        estimate="RCV_D",
    )

    lower_bounds = _modify_bounds(
        spec.get("lower"),
        request["free_names"],
        -float("inf"),
    )
    upper_bounds = _modify_bounds(
        spec.get("upper"),
        request["free_names"],
        float("inf"),
    )

    return _cpp_shell_rcv_d.shell_rcv_d(
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
        generate=True,
        name=request["settings"]["name"],
        mode=request["settings"]["mode"],
        generating_model=name,
        n_draws=control["n_draws"],
        seed=control["seed"],
        threads=control["threads"],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def _rcv_d_fit_data(simulation):
    rows = []
    for row in simulation:
        out = dict(row)
        out["Subject"] = out.get("draw")
        out["Block"] = out.get("block")
        out["Trial"] = out.get("trial")
        out["Sub_Choose"] = out.get("action")
        out["L_choice"] = out.get("object_1")
        out["R_choice"] = out.get("object_2")
        out["L_reward"] = out.get("reward_1")
        out["R_reward"] = out.get("reward_2")
        rows.append(out)
    return pandas.DataFrame(rows)


def _rcv_d_colnames():
    return {
        "subid": "Subject",
        "block": "Block",
        "trial": "Trial",
        "object": ["L_choice", "R_choice"],
        "reward": ["L_reward", "R_reward"],
        "action": "Sub_Choose",
    }


def _rcv_d_fit_control(estimator, scope, control):
    out = dict(control)
    if estimator in ["abc", "rnn"] and scope is not None:
        out["scope"] = scope
    if out.get("scope") is None:
        out.pop("scope", None)
    return out


def _rcv_d_fit_table(fit, generating_model, candidate_model, estimator):
    if isinstance(fit, list):
        rows = fit
    else:
        rows = [fit]
    out = []
    for row in rows:
        item = dict(row)
        item["generating_model"] = generating_model
        item["candidate_model"] = candidate_model
        item["estimator"] = estimator
        out.append(item)
    return out


def _rcv_d_tag_draw(fit_result, draw):
    fit = fit_result.get("fit")
    if isinstance(fit, list):
        for row in fit:
            row["subid"] = str(draw)
    elif isinstance(fit, dict):
        fit["subid"] = str(draw)

    diagnostics = fit_result.get("diagnostics", {})
    subjects = diagnostics.get("subjects")
    if isinstance(subjects, list):
        for row in subjects:
            row["subid"] = str(draw)


def _rcv_d_recovery_table(truth, fit, estimator):
    rows = []
    for fit_row in fit:
        for truth_row in truth:
            if truth_row.get("generating_model") != fit_row.get(
                "generating_model"
            ):
                continue
            if str(truth_row.get("draw")) != str(fit_row.get("subid")):
                continue
            shared = set(truth_row.keys()).intersection(fit_row.keys())
            params = shared.difference(
                [
                    "generating_model",
                    "candidate_model",
                    "estimator",
                    "draw",
                    "subid",
                ]
            )
            for parameter in params:
                true_value = float(truth_row[parameter])
                recovered = float(fit_row[parameter])
                rows.append(
                    {
                        "generating_model": fit_row["generating_model"],
                        "candidate_model": fit_row["candidate_model"],
                        "draw": fit_row.get("subid"),
                        "subid": fit_row.get("subid"),
                        "parameter": parameter,
                        "true": true_value,
                        "recovered": recovered,
                        "error": recovered - true_value,
                        "abs_error": abs(recovered - true_value),
                        "estimator": estimator,
                    }
                )
    return rows


def _rcv_d_model_recovery_table(fit):
    rows = []
    for fit_row in fit:
        item = {
            "generating_model": fit_row.get("generating_model"),
            "candidate_model": fit_row.get("candidate_model"),
            "draw": fit_row.get("subid"),
            "subid": fit_row.get("subid"),
            "score": _rcv_d_score(fit_row),
            "selected": False,
        }
        rows.append(item)

    groups = {}
    for index, row in enumerate(rows):
        key = (row["generating_model"], row["draw"])
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        best = indices[0]
        for index in indices:
            if rows[index]["score"] > rows[best]["score"]:
                best = index
        rows[best]["selected"] = True
    return rows


def _rcv_d_score(row):
    for name in ["LogPo", "LogL"]:
        if row.get(name) is not None:
            return float(row[name])
    for name in ["NLL", "AIC"]:
        if row.get(name) is not None:
            return -float(row[name])
    return float("-inf")
