#pragma once

#include <multiRL/algorithm_stan.hpp>
#include <multiRL/modify_control.hpp>
#include <multiRL/task_builder.hpp>

#include <string>
#include <vector>

namespace multiRL {

/* ========================================================================== *
 *                         Subject MCMC Result
 * ========================================================================== */

struct SubjectMCMCResult {
    std::string subid;
    std::string cond;
    int status = -1;
    int n_evals = 0;
    int n_draws = 0;
    int n_chains = 0;
    int warmup = 0;
    int thin = 0;
    int leapfrog_steps = 0;
    double accept_rate = 0.0;
    double step_size = 0.0;

    double logL = missing_real();
    double logPrior = missing_real();
    double logPost = missing_real();
    double aic = missing_real();
    double bic = missing_real();

    std::vector<std::vector<double>> best_params;
    std::vector<bool> lower_bounds;
    std::vector<bool> upper_bounds;
    std::vector<std::vector<double>> samples;
    std::vector<double> draw_logPost;

    std::string result_message;
    std::string stop_reason = "not_started";
};

/* ========================================================================== *
 *                         MCMC Public API
 * ========================================================================== */

std::vector<SubjectMCMCResult> estimate_mcmc(
    const std::vector<RunTask>& tasks,
    const MCMCControl& raw_control = MCMCControl()
);

}  // namespace multiRL