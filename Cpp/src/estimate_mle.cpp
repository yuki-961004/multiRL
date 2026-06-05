#include <multiRL/estimate_mle.hpp>

#include <multiRL/algorithm_nlopt.hpp>
#include <multiRL/info_nlopt.hpp>
#include <multiRL/modify_control.hpp>
#include <multiRL/process_model_free.hpp>

#include <cctype>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MULTIRL_HAS_NLOPT
#include <nlopt.hpp>
#endif

namespace multiRL {

namespace {

std::string upper_string(const std::string& value) {
    std::string out = value;
    for (char& item : out) {
        item = static_cast<char>(
            std::toupper(static_cast<unsigned char>(item))
        );
    }
    return out;
}

bool uses_nlopt_global_rng(const std::string& algorithm) {
    const std::string normalized = upper_string(
        normalize_nlopt_algorithm_name(algorithm)
    );

    return normalized.find("MLSL") != std::string::npos ||
           normalized.find("CRS") != std::string::npos ||
           normalized.find("ISRES") != std::string::npos ||
           normalized.find("ESCH") != std::string::npos;
}

EstimateMleResult estimate_mle_single(
    const RunTask& task,
    const MLEControl& control,
    const bool seed_nlopt
) {
    const NLoptControl& nlopt_control = control.nlopt;

    EstimateMleResult result;
    result.params = task.params;

    std::vector<double> x0 = FreeValues::extract(task.params);
    if (x0.empty()) {
        result.metric = process_model_free(task).metric;
        result.optimum_value = result.metric.nll;
        result.status = 0;
        result.result_message = "No free parameters.";
        result.stop_reason = "no_free_parameters";
        return result;
    }

#ifdef MULTIRL_HAS_NLOPT
    if (seed_nlopt && nlopt_control.seed >= 0) {
        nlopt::srand(static_cast<unsigned long>(nlopt_control.seed));
    }

    nlopt::opt opt = Nlopt::build(nlopt_control, x0.size());
    opt.set_min_objective(Nlopt::objective, const_cast<RunTask*>(&task));

    double minf = 0.0;
    nlopt::result status = nlopt::FAILURE;

    try {
        status = opt.optimize(x0, minf);
    } catch (const std::exception& error) {
        FreeValues::assign(result.params, x0);
        result.metric = Nlopt::evaluate(task, x0);
        result.status = static_cast<int>(status);
        result.n_evals = opt.get_numevals();
        result.result_message = error.what();
        result.stop_reason = "exception";
        result.optimum_value = minf;
        return result;
    }

    FreeValues::assign(result.params, x0);
    result.metric = Nlopt::evaluate(task, x0);
    result.status = static_cast<int>(status);
    result.n_evals = opt.get_numevals();
    result.optimum_value = minf;
    result.result_message = NloptInfo::message(status);
    result.stop_reason = NloptInfo::stop_reason(status);
    return result;
#else
    (void) control;
    result.metric = Nlopt::evaluate(task, x0);
    result.optimum_value = result.metric.nll;
    result.result_message = "NLopt support is not enabled.";
    result.stop_reason = "nlopt_disabled";
    return result;
#endif
}

}  // namespace

EstimateMleResult estimate_mle(
    const RunTask& task,
    const MLEControl& raw_control
) {
    const MLEControl control = modify_control(raw_control, "mle");
    return estimate_mle_single(task, control, true);
}

std::vector<EstimateMleResult> estimate_mle(
    const std::vector<RunTask>& tasks,
    const MLEControl& raw_control
) {
    const MLEControl control = modify_control(raw_control, "mle");
    const NLoptControl& nlopt_control = control.nlopt;

#ifdef _OPENMP
    if (nlopt_control.threads > 0) {
        omp_set_num_threads(nlopt_control.threads);
    }
#endif

#ifdef MULTIRL_HAS_NLOPT
    if (nlopt_control.seed >= 0) {
        nlopt::srand(static_cast<unsigned long>(nlopt_control.seed));
    }
#endif

    std::vector<EstimateMleResult> results(tasks.size());
    const int n_tasks = static_cast<int>(tasks.size());
    const bool use_serial_rng =
        n_tasks > 1 &&
        nlopt_control.threads == 1 &&
        uses_nlopt_global_rng(nlopt_control.algorithm);

    if (use_serial_rng) {
        for (int index = 0; index < n_tasks; ++index) {
            const std::size_t row = static_cast<std::size_t>(index);
            MLEControl local_control = control;
            if (nlopt_control.seed >= 0) {
                local_control.nlopt.seed = nlopt_control.seed + index;
            }
            results[row] = estimate_mle_single(
                tasks[row],
                local_control,
                true
            );
        }
        return results;
    }

#ifdef _OPENMP
#pragma omp parallel for if(n_tasks > 1)
#endif
    for (int index = 0; index < n_tasks; ++index) {
        const std::size_t row = static_cast<std::size_t>(index);
        results[row] = estimate_mle_single(tasks[row], control, false);
    }

    return results;
}

EstimateMleResult estimate_mle(
    const RunTask& task,
    const NLoptControl& raw_control
) {
    MLEControl control;
    control.nlopt = raw_control;
    return estimate_mle(task, control);
}

}  // namespace multiRL
