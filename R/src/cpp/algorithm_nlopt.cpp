#include <multiRL/algorithm_nlopt.hpp>

#include <multiRL/info_nlopt.hpp>
#include <multiRL/modify_control.hpp>
#include <multiRL/process_model_free.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>

namespace multiRL {

namespace FreeValues {

void assign(
    Params& params,
    const std::vector<double>& free_values
) {
    for (std::size_t index = 0; index < params.free_names.size(); ++index) {
        params.values[params.free_names[index]] = free_values[index];
    }
}

std::vector<double> extract(const Params& params) {
    std::vector<double> out;
    out.reserve(params.free_names.size());

    for (const std::string& name : params.free_names) {
        out.push_back(params.get(name));
    }

    return out;
}

}  // namespace FreeValues

namespace Nlopt {

CriterionResult evaluate(
    const RunTask& task,
    const std::vector<double>& free_values
) {
    RunTask local_task = task;
    FreeValues::assign(local_task.params, free_values);
    return process_model_free(local_task).metric;
}

double score(
    const RunTask& task,
    const std::vector<double>& free_values
) {
    const CriterionResult metric = evaluate(task, free_values);

    if (task.settings.estimate == "MAP" &&
        !std::isnan(metric.log_posterior)) {
        return -metric.log_posterior;
    }

    return metric.nll;
}

#ifdef MULTIRL_HAS_NLOPT

double objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* data
) {
    (void) grad;

    const RunTask* task = static_cast<const RunTask*>(data);
    return score(*task, x);
}

nlopt::opt build(
    const NLoptControl& control,
    const std::size_t n_params
) {
    const std::string algorithm =
        normalize_nlopt_algorithm_name(control.algorithm);
    const std::string local_algorithm =
        normalize_nlopt_algorithm_name(control.local_algorithm);

    nlopt::opt opt(algorithm.c_str(), n_params);

    if (control.lower_bounds.size() == n_params) {
        opt.set_lower_bounds(control.lower_bounds);
    }
    if (control.upper_bounds.size() == n_params) {
        opt.set_upper_bounds(control.upper_bounds);
    }

    if (algorithm.find("MLSL") != std::string::npos ||
        algorithm.find("AUGLAG") != std::string::npos) {
        nlopt::opt local_opt(local_algorithm.c_str(), n_params);

        if (control.local_xtol_rel > 0.0) {
            local_opt.set_xtol_rel(control.local_xtol_rel);
        }
        if (control.ftol_rel > 0.0) {
            local_opt.set_ftol_rel(control.ftol_rel);
        }
        if (control.ftol_abs > 0.0) {
            local_opt.set_ftol_abs(control.ftol_abs);
        }
        if (control.xtol_abs > 0.0) {
            local_opt.set_xtol_abs(control.xtol_abs);
        }

        opt.set_local_optimizer(local_opt);
    }
    if (control.xtol_rel > 0.0) {
        opt.set_xtol_rel(control.xtol_rel);
    }
    if (control.ftol_rel > 0.0) {
        opt.set_ftol_rel(control.ftol_rel);
    }
    if (control.ftol_abs > 0.0) {
        opt.set_ftol_abs(control.ftol_abs);
    }
    if (control.xtol_abs > 0.0) {
        opt.set_xtol_abs(control.xtol_abs);
    }
    if (control.maxtime > 0.0) {
        opt.set_maxtime(control.maxtime);
    }
    if (control.stopval > 0.0) {
        opt.set_stopval(control.stopval);
    }
    if (control.maxeval > 0) {
        opt.set_maxeval(control.maxeval);
    }
    if (control.population > 0) {
        opt.set_population(control.population);
    }
    if (control.initial_step > 0.0) {
        opt.set_initial_step(control.initial_step);
    }

    return opt;
}

/* ========================================================================== *
 * Deterministic Multi-Start MLE                                              *
 * ========================================================================== *
 *                                                                             *
 * Random global algorithms (MLSL, CRS, ISRES, ESCH) use NLopt global RNG.     *
 * When called from parallel threads they share the same global RNG state      *
 * and produce non-reproducible results.                                       *
 *                                                                             *
 * This function replaces the random global algorithm with deterministic       *
 * multi-start: generate initial points via std::mt19937(seed), then run       *
 * LN_BOBYQA (deterministic) from each point, keep the best result.            *
 * Maintains MLSL-style global search semantics while being thread-safe.       *
 * ========================================================================== */

bool nlopt_algo_uses_global_rng(const std::string& algorithm) {
    const std::string normalized =
        normalize_nlopt_algorithm_name(algorithm);
    return normalized.find("MLSL") != std::string::npos ||
           normalized.find("CRS") != std::string::npos ||
           normalized.find("ISRES") != std::string::npos ||
           normalized.find("ESCH") != std::string::npos;
}

EstimateMleResult deterministic_mle(
    const RunTask& task,
    const NLoptControl& control
) {
    const std::vector<double> x0 = FreeValues::extract(task.params);
    const std::size_t n_params = x0.size();
    const int n_starts = std::max(
        10,
        static_cast<int>(n_params) * 5
    );

    std::mt19937 rng(
        static_cast<std::mt19937::result_type>(
            static_cast<unsigned long>(control.seed)
        )
    );

    EstimateMleResult best;
    best.params = task.params;
    best.optimum_value = std::numeric_limits<double>::infinity();

    /* Fallback local algorithm: use LN_BOBYQA if local_algorithm is          *
     * also a random global algorithm or is empty.                             */

    std::string local_algo = control.local_algorithm;
    if (local_algo.empty() || nlopt_algo_uses_global_rng(local_algo)) {
        local_algo = "LN_BOBYQA";
    }

    NLoptControl local_control = control;
    local_control.algorithm = local_algo;

    for (int start = 0; start < n_starts; ++start) {
        std::vector<double> candidate(n_params);

        for (std::size_t param_index = 0; param_index < n_params;
             ++param_index) {
            const double lower = param_index < control.lower_bounds.size()
                ? control.lower_bounds[param_index]
                : -1e6;
            const double upper = param_index < control.upper_bounds.size()
                ? control.upper_bounds[param_index]
                : 1e6;

            std::uniform_real_distribution<double> uniform(
                lower,
                upper
            );
            candidate[param_index] = uniform(rng);
        }

        try {
            nlopt::opt local_opt = build(local_control, n_params);
            local_opt.set_min_objective(
                objective,
                const_cast<RunTask*>(&task)
            );

            double local_minf = 0.0;
            const nlopt::result local_status = local_opt.optimize(
                candidate,
                local_minf
            );

            if (local_minf < best.optimum_value) {
                FreeValues::assign(best.params, candidate);
                best.metric = evaluate(task, candidate);
                best.optimum_value = local_minf;
                best.n_evals += static_cast<int>(
                    local_opt.get_numevals()
                );
                best.status = static_cast<int>(local_status);
                best.result_message =
                    NloptInfo::message(local_status);
                best.stop_reason =
                    NloptInfo::stop_reason(local_status);
            }
        } catch (const std::exception& error) {
            continue;
        }
    }

    if (!std::isfinite(best.optimum_value)) {
        FreeValues::assign(best.params, x0);
        best.metric = evaluate(task, x0);
        best.optimum_value = best.metric.nll;
        best.status = -1;
        best.result_message = "All multi-start attempts failed.";
        best.stop_reason = "exception";
    }

    return best;
}

#endif

}  // namespace Nlopt

}  // namespace multiRL
