#include <multiRL/estimate_map.hpp>

#include <multiRL/criterion_prior.hpp>
#include <multiRL/modify_control.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace multiRL {

namespace {

/* -------------------------------------------------------------------------- *
 * Subject Result Helpers                                                     *
 * -------------------------------------------------------------------------- */

double map_score(const EstimateMleResult& result) {
    if (!std::isnan(result.metric.log_posterior)) {
        return result.metric.log_posterior;
    }

    return result.metric.log_likelihood;
}

std::unordered_map<std::string, double> free_param_map(
    const EstimateMleResult& result
) {
    std::unordered_map<std::string, double> out;

    for (const std::string& name : result.params.free_names) {
        out[name] = result.params.get(name);
    }

    return out;
}

std::vector<std::unordered_map<std::string, double>> free_param_maps(
    const std::vector<EstimateMleResult>& results
) {
    std::vector<std::unordered_map<std::string, double>> out;
    out.reserve(results.size());

    for (const EstimateMleResult& result : results) {
        out.push_back(free_param_map(result));
    }

    return out;
}

double sum_map_score(const std::vector<EstimateMleResult>& results) {
    double out = 0.0;

    for (const EstimateMleResult& result : results) {
        out += map_score(result);
    }

    return out;
}

CriterionResult sum_metric(const std::vector<EstimateMleResult>& results) {
    CriterionResult out;
    out.acc = 0.0;
    out.log_likelihood = 0.0;
    out.log_prior = 0.0;
    out.log_posterior = 0.0;
    out.nll = 0.0;
    out.aic = 0.0;
    out.bic = 0.0;

    if (results.empty()) {
        return out;
    }

    for (const EstimateMleResult& result : results) {
        out.acc += result.metric.acc;
        out.log_likelihood += result.metric.log_likelihood;
        out.nll += result.metric.nll;
        out.aic += result.metric.aic;
        out.bic += result.metric.bic;

        if (!std::isnan(result.metric.log_prior)) {
            out.log_prior += result.metric.log_prior;
        }
        if (!std::isnan(result.metric.log_posterior)) {
            out.log_posterior += result.metric.log_posterior;
        }
    }

    out.acc /= static_cast<double>(results.size());
    return out;
}

void apply_subject_results(
    std::vector<RunTask>& tasks,
    const std::vector<EstimateMleResult>& results
) {
    const std::size_t n_items = std::min(tasks.size(), results.size());

    for (std::size_t index = 0; index < n_items; ++index) {
        tasks[index].params = results[index].params;
    }
}

void apply_prior_group(
    std::vector<RunTask>& tasks,
    const PriorGroup& priors
) {
    for (RunTask& task : tasks) {
        task.priors = priors;
    }
}

void apply_estimate_name(
    std::vector<RunTask>& tasks,
    const std::string& estimate
) {
    for (RunTask& task : tasks) {
        task.settings.estimate = estimate;
    }
}

std::string signed_number(const double value) {
    if (value >= 0.0) {
        return "+";
    }

    return "";
}

void print_map_message(const MAPControl& control, const std::string& message) {
    if (control.mle.nlopt.print_level > 0) {
        std::cout << message << std::endl;
    }
}

/* -------------------------------------------------------------------------- *
 * Expectation Step                                                           *
 * -------------------------------------------------------------------------- */

std::vector<EstimateMleResult> em_e_step(
    std::vector<RunTask>& tasks,
    const MLEControl& control,
    const std::string& estimate
) {
    apply_estimate_name(tasks, estimate);
    return estimate_mle(tasks, control);
}

/* -------------------------------------------------------------------------- *
 * Maximization Step                                                          *
 * -------------------------------------------------------------------------- */

PriorGroup em_m_step(
    const PriorGroup& current_priors,
    const std::vector<EstimateMleResult>& results
) {
    CriterionPrior prior_engine(current_priors);
    prior_engine.update(free_param_maps(results));
    return prior_engine.group();
}

}  // namespace

EstimateMapResult estimate_map(
    const RunTask& task,
    const MAPControl& raw_control
) {
    return estimate_map(std::vector<RunTask>{task}, raw_control);
}

EstimateMapResult estimate_map(
    const std::vector<RunTask>& tasks,
    const MAPControl& raw_control
) {
    MAPControl control = modify_control(raw_control, "map");
    std::vector<RunTask> local_tasks = tasks;
    EstimateMapResult out;

    if (local_tasks.empty()) {
        out.stop_reason = "empty_tasks";
        return out;
    }

    print_map_message(
        control,
        "Initializing " + local_tasks.front().settings.name
    );

    std::vector<EstimateMleResult> current = em_e_step(
        local_tasks,
        control.mle,
        "MLE"
    );
    std::vector<EstimateMleResult> best = current;
    PriorGroup posteriors = em_m_step(local_tasks.front().priors, current);

    apply_subject_results(local_tasks, current);
    apply_prior_group(local_tasks, posteriors);

    double log_posterior = sum_map_score(current);
    double best_log_posterior = -std::numeric_limits<double>::infinity();
    double delta_log_posterior = 1.0;
    int patience = control.map_patience - 1;
    int stuck = 0;

    if (!std::isfinite(log_posterior)) {
        log_posterior = 0.0;
        print_map_message(
            control,
            "Infinite log-priors detected. Please adjust the priors."
        );
    }

    print_map_message(
        control,
        "Starting Expectation-Maximization Algorithm\n"
        "Log-Posterior Probability: " + std::to_string(log_posterior)
    );

    while (std::abs(delta_log_posterior) > control.map_tol) {
        current = em_e_step(local_tasks, control.mle, "MAP");

        const double sum_log_posterior = sum_map_score(current);

        if (!std::isfinite(sum_log_posterior)) {
            ++stuck;
            --patience;
            print_map_message(
                control,
                "Invalid log-priors detected. Please adjust the priors."
            );

            ++out.iterations;
            out.delta = delta_log_posterior;

            if (out.iterations >= control.map_maxiter) {
                print_map_message(
                    control,
                    "Iteration limit reached without convergence."
                );
                out.stop_reason = "maxiter";
                break;
            }

            if (stuck > 1 || patience == 0) {
                print_map_message(
                    control,
                    "EM-MAP seems to be stuck. You could try other priors or "
                    "just accept the best results for now."
                );
                out.stop_reason = "stuck";
                break;
            }

            continue;
        }

        if (delta_log_posterior ==
            log_posterior - sum_log_posterior) {
            ++stuck;
        }

        delta_log_posterior = sum_log_posterior - log_posterior;
        log_posterior = sum_log_posterior;

        posteriors = em_m_step(posteriors, current);
        apply_subject_results(local_tasks, current);
        apply_prior_group(local_tasks, posteriors);

        if (log_posterior > best_log_posterior) {
            best_log_posterior = log_posterior;
            ++patience;
            best = current;
        } else {
            --patience;
        }

        std::ostringstream message;
        message << "current: " << std::round(log_posterior * 100.0) / 100.0
                << ", "
                << "Delta: " << signed_number(delta_log_posterior)
                << std::round(delta_log_posterior * 1000.0) / 1000.0
                << ", "
                << "best: "
                << std::round(best_log_posterior * 100.0) / 100.0
                << ", "
                << "patience: " << patience;
        print_map_message(control, message.str());

        ++out.iterations;
        out.delta = delta_log_posterior;

        if (std::abs(delta_log_posterior) <= control.map_tol) {
            print_map_message(
                control,
                "Congrets~ EM-MAP finds solution!"
            );
            out.stop_reason = "converged";
            break;
        }

        if (out.iterations >= control.map_maxiter) {
            print_map_message(
                control,
                "Iteration limit reached without convergence."
            );
            out.stop_reason = "maxiter";
            break;
        }

        if (stuck > 1 || patience == 0) {
            print_map_message(
                control,
                "EM-MAP seems to be stuck. You could try other priors or "
                "just accept the best results for now."
            );
            out.stop_reason = "stuck";
            break;
        }
    }

    out.subjects = best;
    out.best = best.front();
    out.best.metric = sum_metric(best);
    out.priors = posteriors;
    out.best_log_posterior = best_log_posterior;

    if (out.stop_reason.empty()) {
        out.stop_reason = "finished";
    }

    return out;
}

}  // namespace multiRL
