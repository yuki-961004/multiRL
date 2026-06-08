#ifdef MULTIRL_HAS_STAN

#include <multiRL/estimate_mcmc.hpp>

#include <multiRL/algorithm_stan.hpp>
#include <multiRL/modify_control.hpp>
#include <multiRL/process_model_free.hpp>

#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace multiRL {
namespace {

/* ========================================================================== *
 *                        Single Subject MCMC Runner
 * ========================================================================== */

SubjectMCMCResult run_subject_mcmc(
    const RunTask& task,
    const MCMCControl& control
) {
    SubjectMCMCResult out;
    out.subid = "1";
    out.cond = "default";

    if (task.params.free_names.empty()) {
        out.status = -1;
        out.result_message = "MCMC requires at least one free parameter.";
        out.stop_reason = "empty_parameter";
        return out;
    }

    // Extract initial values from task params
    std::vector<double> initial;
    initial.reserve(task.params.free_names.size());
    for (const std::string& name : task.params.free_names) {
        initial.push_back(task.params.get(name));
    }

    // Sanitize initial point within bounds
    StanAdapter::sanitize_initial_point(
        initial,
        control.lower_bounds,
        control.upper_bounds
    );

    // Build adapter and unconstrain initial values
    StanAdapter::Adapter adapter(task);
    const Eigen::VectorXd initial_unconstrained =
        adapter.unconstrain(initial);

    // Run each chain sequentially
    std::vector<HMCSamplerResult> chain_results;
    chain_results.reserve(static_cast<std::size_t>(control.chains));

    for (int chain = 0; chain < control.chains; ++chain) {
        if (control.algorithm == "nuts") {
            chain_results.push_back(
                NUTS::run_chain(
                    adapter,
                    initial_unconstrained,
                    control,
                    chain
                )
            );
        } else {
            chain_results.push_back(
                HMC::run_chain(
                    adapter,
                    initial_unconstrained,
                    control,
                    chain
                )
            );
        }
    }

    // Summarize results across chains
    out.n_evals = adapter.n_evals();
    out.n_chains = control.chains;
    out.warmup = control.warmup;
    out.thin = control.thin;
    out.leapfrog_steps = control.leapfrog_steps;

    // Collect all draws and log probs
    for (const HMCSamplerResult& chain_result : chain_results) {
        for (const std::vector<double>& draw : chain_result.draws) {
            out.samples.push_back(draw);
            ++out.n_draws;
        }
        for (double lp : chain_result.log_prob) {
            out.draw_logPost.push_back(lp);
        }
    }

    // Average accept rate across chains
    double total_accept_rate = 0.0;
    for (const HMCSamplerResult& chain_result : chain_results) {
        total_accept_rate += chain_result.accept_rate;
    }
    out.accept_rate = total_accept_rate /
        static_cast<double>(chain_results.size());
    out.step_size = chain_results[0].final_step_size;

    // Compute posterior mean as best_params
    if (!out.samples.empty()) {
        const std::size_t n_params = task.params.free_names.size();
        std::vector<double> means(n_params, 0.0);
        for (const std::vector<double>& draw : out.samples) {
            for (std::size_t j = 0; j < n_params; ++j) {
                means[j] += draw[j];
            }
        }
        for (std::size_t j = 0; j < n_params; ++j) {
            means[j] /= static_cast<double>(out.samples.size());
        }
        out.best_params = out.samples;

        // Evaluate posterior at posterior mean for fit metrics
        RunTask local_task = task;
        for (std::size_t j = 0; j < n_params; ++j) {
            local_task.params.values[
                task.params.free_names[j]
            ] = means[j];
        }

        const RunResult result = process_model_free(local_task);
        const CriterionPosterior posterior(task.priors);
        const CriterionResult metric = posterior.evaluate(
            local_task,
            result.result
        );

        out.logL = metric.log_likelihood;
        out.logPrior = metric.log_prior;
        out.logPost = metric.log_posterior;
        out.aic = metric.aic;
        out.bic = metric.bic;
    }

    // Determine overall status
    bool all_complete = true;
    for (const HMCSamplerResult& chain_result : chain_results) {
        if (chain_result.status <= 0) {
            all_complete = false;
            out.result_message = chain_result.result_message;
            out.stop_reason = chain_result.stop_reason;
        }
    }

    if (all_complete) {
        out.status = 1;
        out.result_message = "MCMC sampling finished.";
        out.stop_reason = "complete";
    }

    return out;
}

/* ========================================================================== *
 *                       Failed MCMC Result Helper
 * ========================================================================== */

SubjectMCMCResult failed_mcmc_result(
    const RunTask& task,
    const std::string& message,
    const std::string& stop_reason
) {
    SubjectMCMCResult out;
    out.subid = "1";
    out.cond = "default";
    out.status = -1;
    out.result_message = message;
    out.stop_reason = stop_reason;

    out.logL = missing_real();
    out.logPrior = missing_real();
    out.logPost = missing_real();
    out.aic = missing_real();
    out.bic = missing_real();

    return out;
}

}  // namespace

/* ========================================================================== *
 *                              Main Public API
 * ========================================================================== */

std::vector<SubjectMCMCResult> estimate_mcmc(
    const std::vector<RunTask>& tasks,
    const MCMCControl& raw_control
) {
    const MCMCControl control = modify_control(raw_control, "mcmc");

    if (tasks.empty()) {
        return {};
    }

#ifdef _OPENMP
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#endif

    const int n_tasks = static_cast<int>(tasks.size());
    std::vector<SubjectMCMCResult> results(
        static_cast<std::size_t>(n_tasks)
    );

    if (control.print_level > 0 && n_tasks > 0) {
        std::cout << "[MCMC] Starting sampling for "
                  << n_tasks << " subject(s)" << std::endl;
    }

    // Parallel subject-level MCMC with OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if(n_tasks > 1)
#endif
    for (int index = 0; index < n_tasks; ++index) {
        const std::size_t row = static_cast<std::size_t>(index);
        const RunTask& task = tasks[row];

        try {
            results[row] = run_subject_mcmc(task, control);
        } catch (const std::exception& error) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                std::cerr << "\n[MCMC Error] Subject "
                          << row + 1
                          << " fitting failed: "
                          << error.what() << "\n";
            }
            results[row] = failed_mcmc_result(
                task,
                error.what(),
                "exception"
            );
        }

        if (control.print_level > 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                std::cout << "[MCMC] Subject " << row + 1
                          << "/" << n_tasks
                          << " completed" << std::endl;
            }
        }
    }

    if (control.print_level > 0 && n_tasks > 0) {
        std::cout << "[MCMC] All subjects completed." << std::endl;
    }

    return results;
}

}  // namespace multiRL

#else

#include <multiRL/estimate_mcmc.hpp>

#include <stdexcept>

namespace multiRL {

std::vector<SubjectMCMCResult> estimate_mcmc(
    const std::vector<RunTask>& tasks,
    const MCMCControl& raw_control
) {
    (void) tasks;
    (void) raw_control;
    throw std::runtime_error(
        "estimate_mcmc requires Stan Math support. Rebuild multiRL with "
        "MULTIRL_ENABLE_MCMC=TRUE for R packages or "
        "MULTIRL_ENABLE_MCMC=ON for CMake builds."
    );
}

}  // namespace multiRL

#endif // MULTIRL_HAS_STAN
