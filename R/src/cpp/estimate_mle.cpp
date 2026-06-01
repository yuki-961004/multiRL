#include <multiRL/estimate_mle.hpp>

#include <multiRL/process_MDP_free.hpp>

#include <algorithm>
#include <stdexcept>

#ifdef MULTIRL_HAS_NLOPT
#include <nlopt.hpp>
#endif

namespace multiRL {

namespace {

void update_free_values(
    Params& params,
    const std::vector<double>& free_values
) {
    for (std::size_t index = 0; index < params.free_names.size(); ++index) {
        params.values[params.free_names[index]] = free_values[index];
    }
}

std::vector<double> extract_free_values(const Params& params) {
    std::vector<double> out;
    out.reserve(params.free_names.size());

    for (const std::string& name : params.free_names) {
        out.push_back(params.get(name));
    }

    return out;
}

CriterionResult evaluate_mle_task(
    const RunTask& task,
    const std::vector<double>& free_values
) {
    RunTask local_task = task;
    update_free_values(local_task.params, free_values);
    return process_MDP_free(local_task).metric;
}

#ifdef MULTIRL_HAS_NLOPT

double estimate_mle_objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* data
) {
    (void) grad;

    const RunTask* task = static_cast<const RunTask*>(data);
    return evaluate_mle_task(*task, x).nll;
}

nlopt::opt build_mle_optimizer(
    const NLoptControl& control,
    const std::size_t n_params
) {
    nlopt::opt opt(control.algorithm.c_str(), n_params);

    if (control.lower_bounds.size() == n_params) {
        opt.set_lower_bounds(control.lower_bounds);
    }
    if (control.upper_bounds.size() == n_params) {
        opt.set_upper_bounds(control.upper_bounds);
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

}  // namespace

EstimateMleResult estimate_mle(
    const RunTask& task,
    const NLoptControl& control
) {
    EstimateMleResult result;
    result.params = task.params;

    std::vector<double> x0 = extract_free_values(task.params);
    if (x0.empty()) {
        result.metric = process_MDP_free(task).metric;
        result.optimum_value = result.metric.nll;
        result.status = 0;
        result.result_message = "No free parameters.";
        result.stop_reason = "no_free_parameters";
        return result;
    }

#ifdef MULTIRL_HAS_NLOPT
    if (control.seed >= 0) {
        nlopt::srand(static_cast<unsigned long>(control.seed));
    }

    nlopt::opt opt = build_mle_optimizer(control, x0.size());
    opt.set_min_objective(estimate_mle_objective, const_cast<RunTask*>(&task));

    double minf = 0.0;
    nlopt::result status = nlopt::FAILURE;

    try {
        status = opt.optimize(x0, minf);
    } catch (const std::exception& error) {
        result.status = static_cast<int>(status);
        result.n_evals = opt.get_numevals();
        result.result_message = error.what();
        result.stop_reason = "exception";
        result.optimum_value = minf;
        return result;
    }

    update_free_values(result.params, x0);
    result.metric = evaluate_mle_task(task, x0);
    result.status = static_cast<int>(status);
    result.n_evals = opt.get_numevals();
    result.optimum_value = minf;
    result.result_message = "NLopt finished.";
    result.stop_reason = "nlopt";
    return result;
#else
    (void) control;
    result.metric = evaluate_mle_task(task, x0);
    result.optimum_value = result.metric.nll;
    result.result_message = "NLopt support is not enabled.";
    result.stop_reason = "nlopt_disabled";
    return result;
#endif
}

}  // namespace multiRL
