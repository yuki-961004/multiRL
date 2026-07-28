"""Replay-experiment workflow shell for the Python multiRL frontend."""

from . import _backend as _cpp_shell_rpl_e
from ._modify_priors import _modify_priors
import math


def rpl_e(result_fit, option=None):
    if option is None:
        option = {"plot": True}
    else:
        option = dict(option)
        option.setdefault("plot", True)

    if "model_recovery" in result_fit:
        return _rpl_e_recovery(result_fit, option)

    raw = _rpl_e_raw_fits(result_fit)
    replay = []
    plot_data = []
    human_seen = False

    for name, item in raw.items():
        fit_rows = _rpl_e_fit_rows(item, name)
        cpp = _rpl_e_call_cpp(item, fit_rows)
        replay.extend(cpp["replay"])

        local_plot = list(cpp["plot_data"])
        if human_seen:
            local_plot = [
                row
                for row in local_plot
                if row["source"] != "human"
            ]
        human_seen = True
        plot_data.extend(local_plot)

    out = {
        "input": {
            "fit": _rpl_e_trim_fit(result_fit),
            "option": option,
        },
        "replay": replay,
        "plot_data": plot_data,
        "plot": None,
        "diagnostics": {
            "n_models": len(raw),
            "n_subjects": len({row["subid"] for row in plot_data}),
            "generate": True,
            "replay_success": True,
        },
    }

    if option.get("plot") is True:
        out["plot"] = _rpl_e_plot_fitting(plot_data)
        try:
            out["plot"].show()
        except Exception:
            pass

    return out


def _rpl_e_recovery(result, option):
    out = {
        "input": {
            "recovery": result,
            "option": option,
        },
        "recovery": result.get("recovery"),
        "model_recovery": result.get("model_recovery"),
        "plot": None,
        "diagnostics": {
            "n_generating": len({
                row["generating_model"]
                for row in result.get("model_recovery", [])
            }),
            "n_candidates": len({
                row["candidate_model"]
                for row in result.get("model_recovery", [])
            }),
        },
    }
    if option.get("plot") is True:
        out["plot"] = _rpl_e_plot_recovery(result)
        _rpl_e_show_plot(out["plot"])
    return out


def _rpl_e_show_plot(plot):
    if isinstance(plot, dict):
        for item in plot.values():
            _rpl_e_show_plot(item)
        return
    if isinstance(plot, list):
        for item in plot:
            _rpl_e_show_plot(item)
        return
    try:
        plot.show()
    except Exception:
        pass


def _rpl_e_raw_fits(result):
    if result.get("raw") is not None:
        return result["raw"]
    name = result.get("input", {}).get("settings", {}).get("name", "model")
    return {name: result}


def _rpl_e_fit_rows(item, model_id):
    fit = item["fit"]
    if isinstance(fit, list):
        rows = fit
    else:
        rows = [fit]

    model = item.get("input", {}).get("settings", {}).get("name", model_id)
    out = []
    for row in rows:
        local = dict(row)
        local.setdefault("model", model)
        local.setdefault("model_id", model_id)
        local["subid"] = str(local.get("subid", "1"))
        out.append(local)
    return out


def _rpl_e_call_cpp(item, fit_rows):
    request = item["input"]
    features = request["features"]
    settings = request["settings"]
    priors = request["priors"]
    free_names = request["free_names"] if "free_names" in request else None

    if free_names is None:
        params = request["params"]
        free_names = list(params.get("free", {}).keys())
    else:
        params = request["params"]

    modified_priors = _rpl_e_priors(
        priors=request.get("priors"),
        free_names=free_names,
    )

    return _cpp_shell_rpl_e.shell_rpl_e(
        object=features["object"],
        reward=features["reward"],
        action=features["action"],
        block=features["block"],
        trial=features["trial"],
        subid=features.get("subid", ["1"] * len(features["action"])),
        cue=request["behrule"]["cue"],
        rsp=request["behrule"]["rsp"],
        params=_rpl_e_params(params),
        free_names=free_names,
        system=settings["system"],
        fit_rows=fit_rows,
        parameter_names=free_names,
        prior_names=modified_priors["name"],
        prior_types=modified_priors["type"],
        prior_param1=modified_priors["param1"],
        prior_param2=modified_priors["param2"],
        prior_active=modified_priors["active"],
        generate=True,
        name=settings["name"],
        mode=settings["mode"],
    )


def _rpl_e_priors(priors, free_names):
    if priors is None:
        return _modify_priors({}, free_names)
    if all(key in priors for key in ["name", "type", "param1", "param2"]):
        return priors
    return _modify_priors(priors, free_names)


def _rpl_e_params(params):
    if params is None:
        return {}
    if not isinstance(params, dict):
        return dict(params)
    if any(key in params for key in ["free", "fixed", "constant"]):
        out = {}
        out.update(params.get("free", {}))
        out.update(params.get("fixed", {}))
        out.update(params.get("constant", {}))
        return {
            str(name): float(value)
            for name, value in out.items()
            if value is not None
        }
    return {
        str(name): float(value)
        for name, value in params.items()
        if value is not None
    }


def _rpl_e_trim_fit(result):
    return {
        "fit": result.get("fit"),
        "estimator": result.get("estimator"),
        "diagnostics": result.get("diagnostics"),
    }


def _rpl_e_palette(count):
    base_colors = [
        "grey",
        "#053562",
        "#55c186",
        "#f0de36",
        "#f79d1e",
        "#e84a34",
        "#8b2f97",
    ]
    return base_colors[:count]

def _rpl_e_theme_apa():
    try:
        import plotnine
    except ImportError:
        return None
    return (
        plotnine.theme_bw()
        + plotnine.theme(
            panel_grid_major_x=plotnine.element_blank(),
            panel_grid_minor_x=plotnine.element_blank(),
            panel_grid_major_y=plotnine.element_blank(),
            panel_grid_minor_y=plotnine.element_blank(),
            legend_key=plotnine.element_rect(fill=None, color=None),
            strip_background=plotnine.element_rect(fill=None, color=None),
            panel_border=plotnine.element_blank(),
            axis_line=plotnine.element_line(),
        )
    )

def _rpl_e_plot_fitting(plot_data):
    try:
        import pandas
        import plotnine
    except ImportError as error:
        raise ImportError(
            "rpl_e plotting requires plotnine. Install plotnine or set "
            'option = {"plot": False}.'
        ) from error

    data = pandas.DataFrame(plot_data)
    models = sorted(data["model"].unique().tolist())
    n_models = len(models)
    palette = _rpl_e_palette(n_models)
    blocks = sorted(data["block"].unique().tolist())
    return (
        plotnine.ggplot(
            data,
            plotnine.aes(
                x="block",
                y="ratio",
                color="model",
                group="model",
            ),
        )
        + plotnine.stat_summary(
            geom="errorbar",
            fun_data="mean_se",
            width=0.2,
        )
        + plotnine.stat_summary(
            geom="line",
            fun_data="mean_cl_boot",
            size=1,
        )
        + plotnine.facet_wrap("~ action")
        + plotnine.scale_x_continuous(breaks=blocks) + plotnine.scale_y_continuous(limits=(0, 1))
        + plotnine.scale_color_manual(values=palette)
        + plotnine.labs(x="Block", y="Action Ratio", color="Model")
        + _rpl_e_theme_apa()
    )


def _rpl_e_plot_recovery(result):
    try:
        import pandas
        import plotnine
    except ImportError as error:
        raise ImportError(
            "rpl_e plotting requires plotnine. Install plotnine or set "
            'option = {"plot": False}.'
        ) from error

    recovery = pandas.DataFrame(result.get("recovery", []))
    model_recovery = pandas.DataFrame(result.get("model_recovery", []))
    recovery = recovery[
        recovery["true"].map(pandas.notna)
        & recovery["recovered"].map(pandas.notna)
        & (recovery["generating_model"] == recovery["candidate_model"])
    ].copy()
    recovery["plot_true"] = recovery["true"]
    recovery["plot_recovered"] = recovery["recovered"]
    beta = recovery["parameter"].astype(str).str.contains("beta")
    positive = (recovery["plot_true"] > 0) & (
        recovery["plot_recovered"] > 0
    )
    recovery = recovery[(~beta) | positive].copy()
    beta = recovery["parameter"].astype(str).str.contains("beta")
    recovery.loc[beta, "plot_true"] = recovery.loc[
        beta,
        "plot_true",
    ].map(math.log)
    recovery.loc[beta, "plot_recovered"] = recovery.loc[
        beta,
        "plot_recovered",
    ].map(math.log)
    ###########################
    # Preserve input model order
    ###########################
    # Extract model names from generating specs while preserving input order
    model_order = [
        spec.get("settings", {}).get("name", spec.get("model", ""))
        for spec in result.get("input", {}).get("generating", [])
    ]
    # Convert model columns to categorical with input order
    model_recovery["generating_model"] = pandas.Categorical(
        model_recovery["generating_model"],
        categories=model_order,
        ordered=True
    )
    model_recovery["candidate_model"] = pandas.Categorical(
        model_recovery["candidate_model"],
        categories=model_order,
        ordered=True
    )
    recovery["generating_model"] = pandas.Categorical(
        recovery["generating_model"],
        categories=model_order,
        ordered=True
    )

    recovery["panel"] = _rpl_e_parameter_panel(result, recovery)
    limits = _rpl_e_parameter_limits(result, recovery)
    priors = _rpl_e_parameter_prior_labels(result, recovery, limits)
    plots = {}
    for model in recovery["generating_model"].drop_duplicates().tolist():
        local = recovery[recovery["generating_model"] == model].copy()
        local_limits = limits[limits["generating_model"] == model].copy()
        local_priors = priors[priors["generating_model"] == model].copy()
        plots[str(model)] = (
            plotnine.ggplot(
                local,
                plotnine.aes(x="plot_true", y="plot_recovered"),
            )
            + plotnine.geom_point(color="#053562")
            + plotnine.geom_abline(
                slope=1,
                intercept=0,
                linetype="dashed",
                color="#55c186",
            )
            + plotnine.facet_wrap("~ panel", scales="free", nrow=1)
            + plotnine.geom_blank(
                data=local_limits,
                mapping=plotnine.aes(x="limit", y="limit"),
                inherit_aes=False,
            )
            + plotnine.geom_text(
                data=local_priors,
                mapping=plotnine.aes(x="x", y="y", label="label"),
                inherit_aes=False,
                ha="right",
                va="bottom",
                color="#053562",
            )
            + plotnine.labs(x="True", y="Recovered")
            + _rpl_e_theme_apa()
            + plotnine.theme(aspect_ratio=1)
        )

    return {
        "parameter": plots,
        "confusion": _rpl_e_plot_model_matrix(
            model_recovery=model_recovery,
            matrix_type="confusion",
        ),
        "inversion": _rpl_e_plot_model_matrix(
            model_recovery=model_recovery,
            matrix_type="inversion",
        ),
    }


def _rpl_e_parameter_panel(result, recovery):
    labels = []
    for _, row in recovery.iterrows():
        model = str(row["generating_model"])
        parameter = str(row["parameter"])
        local = recovery[
            (recovery["generating_model"] == model)
            & (recovery["parameter"] == parameter)
        ]
        correlation = local["plot_true"].corr(local["plot_recovered"])
        label = model + ": " + parameter + " (r = "
        label += f"{correlation:.2f}" + ")"
        labels.append(label)
    return labels


def _rpl_e_parameter_prior_labels(result, recovery, limits):
    import pandas

    rows = []
    for panel in recovery["panel"].drop_duplicates().tolist():
        local = recovery[recovery["panel"] == panel]
        local_limits = limits[limits["panel"] == panel]
        model = str(local["generating_model"].iloc[0])
        parameter = str(local["parameter"].iloc[0])
        label = _rpl_e_parameter_prior(result, model, parameter)
        if label == "":
            continue
        rows.append({
            "generating_model": model,
            "panel": panel,
            "x": local_limits["limit"].max(),
            "y": local_limits["limit"].min(),
            "label": label,
        })
    return pandas.DataFrame(
        rows,
        columns=["generating_model", "panel", "x", "y", "label"],
    )


def _rpl_e_parameter_limits(result, recovery):
    import pandas

    rows = []
    for panel in recovery["panel"].drop_duplicates().tolist():
        local = recovery[recovery["panel"] == panel]
        model = str(local["generating_model"].iloc[0])
        parameter = str(local["parameter"].iloc[0])
        lower = _rpl_e_parameter_bound(result, model, parameter, "lower")
        upper = _rpl_e_parameter_bound(result, model, parameter, "upper")
        if "beta" in parameter:
            if math.isfinite(lower) and lower > 0:
                lower = math.log(lower)
            else:
                lower = _rpl_e_log_magnitude_limit(local, "lower")
            if math.isfinite(upper) and upper > 0:
                upper = _rpl_e_log_magnitude_limit(local, "upper")
            else:
                upper = _rpl_e_log_magnitude_limit(local, "upper")
        if not math.isfinite(lower):
            lower = min(local["plot_true"].min(), local["plot_recovered"].min())
        if not math.isfinite(upper):
            upper = max(local["plot_true"].max(), local["plot_recovered"].max())
        rows.append({
            "generating_model": model,
            "panel": panel,
            "limit": lower,
        })
        rows.append({
            "generating_model": model,
            "panel": panel,
            "limit": upper,
        })
    return pandas.DataFrame(rows)


def _rpl_e_log_magnitude_limit(local, side):
    values = list(local["true"]) + list(local["recovered"])
    values = [
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) > 0
    ]
    if len(values) == 0:
        return float("nan")
    if side == "lower":
        exponent = math.floor(math.log10(min(values)))
    else:
        exponent = math.ceil(math.log10(max(values)))
    return math.log(10**exponent)


def _rpl_e_parameter_bound(result, model, parameter, side):
    specs = result.get("input", {}).get("generating", [])
    for spec in specs:
        settings = spec.get("settings") or {}
        name = settings.get("name", spec.get("model"))
        if str(name) != str(model):
            continue
        bound = spec.get(side) or {}
        if parameter in bound:
            try:
                return float(bound[parameter])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _rpl_e_parameter_prior(result, model, parameter):
    specs = result.get("input", {}).get("generating", [])
    for spec in specs:
        settings = spec.get("settings") or {}
        name = settings.get("name", spec.get("model"))
        if str(name) != str(model):
            continue
        priors = spec.get("priors") or {}
        prior = priors.get(parameter)
        if prior is None:
            return ""
        return _rpl_e_format_prior(parameter, prior)
    return ""


def _rpl_e_format_prior(parameter, prior):
    prior_type = str(prior.get("type", "")).lower()
    label = {
        "exponential": "exp",
        "normal": "norm",
        "uniform": "unif",
        "lognormal": "lnorm",
    }.get(prior_type, prior_type)
    values = _rpl_e_prior_values(prior)
    if len(values) == 0:
        return label
    return label + "(" + ", ".join(values) + ")"


def _rpl_e_prior_values(prior):
    values = []
    for name, value in prior.items():
        if name == "type" or value is None:
            continue
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
        except (TypeError, ValueError):
            pass
        values.append(str(value))
    return values


def _rpl_e_plot_model_matrix(model_recovery, matrix_type):
    try:
        import plotnine
    except ImportError as error:
        raise ImportError(
            "rpl_e plotting requires plotnine. Install plotnine or set "
            'option = {"plot": False}.'
        ) from error

    matrix = _rpl_e_model_matrix(model_recovery, matrix_type)
    return (
        plotnine.ggplot(
            matrix,
            plotnine.aes(
                x="generating_model",
                y="candidate_model",
                fill="fill_value",
            ),
        )
        + plotnine.geom_tile()
        + plotnine.geom_text(
            plotnine.aes(label="label"),
            color="white",
            fontweight="bold",
        )
        + plotnine.scale_fill_gradientn(
            colors=["#e84a34", "#f0de36", "#55c186"],
            limits=(0, 1),
            guide=None,
        )
        + plotnine.scale_x_discrete(name=None)
        + plotnine.scale_y_discrete(name=None)
        + plotnine.labs(
            title=_rpl_e_matrix_title(matrix_type),
        )
        + _rpl_e_theme_apa()
        + plotnine.theme(
            axis_title_x=plotnine.element_blank(),
            axis_title_y=plotnine.element_blank(),
            axis_line=plotnine.element_blank(),
            axis_ticks=plotnine.element_blank(),
        )
    )


def _rpl_e_matrix_title(matrix_type):
    if matrix_type == "confusion":
        return "Confusion Matrix"
    return "Inversion Matrix"


def _rpl_e_model_matrix(model_recovery, matrix_type):
    import pandas

    generating = model_recovery["generating_model"].drop_duplicates()
    candidate = model_recovery["candidate_model"].drop_duplicates()
    grid = pandas.MultiIndex.from_product(
        [generating, candidate],
        names=["generating_model", "candidate_model"],
    ).to_frame(index=False)

    selected = model_recovery[model_recovery["selected"] == True]
    if len(selected) == 0:
        count = pandas.DataFrame(
            columns=["generating_model", "candidate_model", "count"]
        )
    else:
        count = (
            selected.groupby(["generating_model", "candidate_model"])
            .size()
            .reset_index(name="count")
        )

    matrix = grid.merge(
        count,
        on=["generating_model", "candidate_model"],
        how="left",
    )
    matrix["count"] = matrix["count"].fillna(0)

    if matrix_type == "confusion":
        group = "generating_model"
    else:
        group = "candidate_model"

    if len(selected) == 0:
        denominator = pandas.DataFrame(columns=[group, "total"])
    else:
        denominator = selected.groupby(group).size().reset_index(name="total")

    matrix = matrix.merge(denominator, on=group, how="left")
    matrix["total"] = matrix["total"].fillna(0)
    matrix["value"] = 0.0
    valid = matrix["total"] > 0
    matrix.loc[valid, "value"] = (
        matrix.loc[valid, "count"] / matrix.loc[valid, "total"]
    )
    matrix["value"] = matrix["value"].round(2)
    matrix["fill_value"] = 1 - matrix["value"]
    diagonal = matrix["generating_model"] == matrix["candidate_model"]
    matrix.loc[diagonal, "fill_value"] = matrix.loc[diagonal, "value"]
    matrix["label"] = matrix["value"].map(lambda value: f"{value:.2f}")
    return matrix[
        [
            "generating_model",
            "candidate_model",
            "value",
            "fill_value",
            "label",
        ]
    ]
