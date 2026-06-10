#include <multiRL/estimate_mle.hpp>

#include <multiRL/algorithm_nlopt.hpp>
#include <multiRL/modify_control.hpp>
#include <multiRL/process_model_free.hpp>

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

/* ========================================================================== *
 * Single-subject MLE evaluation                                              *
 * ========================================================================== *
 *                                                                             *
 * When seed >= 0, delegates to Nlopt::deterministic_mle() for thread-safe     *
 * reproducible optimization with random global algorithms.                     *
 * When seed < 0, uses NLopt directly without setting any seed.                *
 * ========================================================================== */

EstimateMleResult estimate_mle_single(
    const RunTask& task,
    const MLEControl& control
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
    /* ------------------------------------------------------------------ *
     * seed >= 0: use Nlopt::deterministic_mle() for reproducibility.      *
     * The deterministic path replaces random global algorithms (MLSL,     *
     * CRS, ISRES, ESCH) with a deterministic multi-start approach.        *
     * ------------------------------------------------------------------ */

    if (nlopt_control.seed >= 0) {
        return Nlopt::deterministic_mle(task, nlopt_control);
    }

    /* ------------------------------------------------------------------ *
     * seed < 0: directly call NLopt without setting a random seed.        *
     * Results will not be reproducible between runs.                      *
     * ------------------------------------------------------------------ */

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
        result.n_evals = static_cast<int>(opt.get_numevals());
        result.result_message = error.what();
        result.stop_reason = "exception";
        result.optimum_value = minf;
        return result;
    }

    FreeValues::assign(result.params, x0);
    result.metric = Nlopt::evaluate(task, x0);
    result.status = static_cast<int>(status);
    result.n_evals = static_cast<int>(opt.get_numevals());
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

/* ========================================================================== *
 * Public API: single-task MLE                                                *
 * ========================================================================== */

EstimateMleResult estimate_mle(
    const RunTask& task,
    const MLEControl& raw_control
) {
    const MLEControl control = modify_control(raw_control, "mle");
    return estimate_mle_single(task, control);
}

/* ========================================================================== *
 * Public API: multi-subject parallel MLE                                     *
 * ========================================================================== *
 *                                                                             *
 * Each subject gets its own control copy with seed offset by subject index.   *
 * The deterministic path in Nlopt::deterministic_mle() uses std::mt19937      *
 * seeded per-task, not NLopt global RNG, so 32-thread OpenMP is fully         *
 * reproducible.                                                               *
 * ========================================================================== */

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

    std::vector<EstimateMleResult> results(tasks.size());
    const int n_tasks = static_cast<int>(tasks.size());

#ifdef _OPENMP
#pragma omp parallel for if(n_tasks > 1)
#endif
    for (int index = 0; index < n_tasks; ++index) {
        const std::size_t row = static_cast<std::size_t>(index);

        /* -------------------------------------------------------------- *
         * Each task uses its own control copy with seed + index to avoid  *
         * competing for the same NLopt global RNG across threads.         *
         * -------------------------------------------------------------- */

        MLEControl local_control = control;
        if (nlopt_control.seed >= 0) {
            local_control.nlopt.seed = nlopt_control.seed + index;
        }

        results[row] = estimate_mle_single(
            tasks[row],
            local_control
        );
    }

    return results;
}

/* ========================================================================== *
 * Convenience overload: NLoptControl -> MLEControl                           *
 * ========================================================================== */

EstimateMleResult estimate_mle(
    const RunTask& task,
    const NLoptControl& raw_control
) {
    MLEControl control;
    control.nlopt = raw_control;
    return estimate_mle(task, control);
}

}  // namespace multiRL
