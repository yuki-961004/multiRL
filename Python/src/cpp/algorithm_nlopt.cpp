#include <multiRL/algorithm_nlopt.hpp>

#include <multiRL/modify_control.hpp>
#include <multiRL/process_MDP_free.hpp>

#include <cmath>
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
    return process_MDP_free(local_task).metric;
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

#endif

}  // namespace Nlopt

}  // namespace multiRL
