#include <multiRL/estimate_abc.hpp>

#include <multiRL/process_MDP_free.hpp>
#include <multiRL/task_builder.hpp>

#include <abcpp/abc.hpp>
#include <abcpp/abcpp_impl.hpp>
#include <abcpp/options.hpp>
#include <abcpp/summary.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace multiRL {
namespace {

double finite_or(double value, double fallback) {
    return std::isfinite(value) ? value : fallback;
}

std::string task_subid(const RunTask& task) {
    if (!task.input.idinfo.empty() && !task.input.idinfo[0].empty()) {
        return task.input.idinfo[0][0];
    }
    return "1";
}

std::vector<double> observed_summary(const RunTask& task) {
    const std::size_t n_cues = task.behrule.cue.size();
    std::vector<double> counts(n_cues, 0.0);
    double reward_sum = 0.0;
    double valid_reward = 0.0;

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        const std::string& action = task.input.action[row];
        const auto cue_it = task.behrule.cue_index.find(action);
        if (cue_it != task.behrule.cue_index.end()) {
            counts[cue_it->second] += 1.0;
        }

        for (std::size_t option = 0; option < task.input.state[row].size();
             ++option) {
            bool found = false;
            for (const std::string& value : task.input.state[row][option]) {
                if (value == action) {
                    found = true;
                }
            }
            if (found && option < task.input.reward_table[row].size()) {
                reward_sum += task.input.reward_table[row][option];
                valid_reward += 1.0;
                break;
            }
        }
    }

    std::vector<double> out;
    out.reserve(2 + n_cues);
    out.push_back(1.0);
    out.push_back(valid_reward > 0.0 ? reward_sum / valid_reward : 0.0);

    const double denom = static_cast<double>(
        std::max<std::size_t>(task.input.n_rows, 1)
    );
    for (double count : counts) {
        out.push_back(count / denom);
    }

    return out;
}

std::vector<double> simulated_summary(
    const RunTask& task,
    const RunResult& result
) {
    const std::size_t n_cues = task.behrule.cue.size();
    std::vector<double> counts(n_cues, 0.0);
    double reward_sum = 0.0;
    double acc_sum = 0.0;
    double n_rows = 0.0;

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        const std::string& action = result.result.latent[row];
        const auto cue_it = task.behrule.cue_index.find(action);
        if (cue_it != task.behrule.cue_index.end()) {
            counts[cue_it->second] += 1.0;
        }
        if (action == task.input.action[row]) {
            acc_sum += 1.0;
        }
        reward_sum += finite_or(result.result.reward[row], 0.0);
        n_rows += 1.0;
    }

    std::vector<double> out;
    out.reserve(2 + n_cues);
    out.push_back(n_rows > 0.0 ? acc_sum / n_rows : 0.0);
    out.push_back(n_rows > 0.0 ? reward_sum / n_rows : 0.0);

    const double denom = std::max(n_rows, 1.0);
    for (double count : counts) {
        out.push_back(count / denom);
    }

    return out;
}

abcpp::Matrix to_matrix(
    const std::vector<std::vector<double>>& values
) {
    if (values.empty()) {
        return abcpp::Matrix();
    }

    abcpp::Matrix out(values.size(), values.front().size());
    for (std::size_t row = 0; row < values.size(); ++row) {
        for (std::size_t col = 0; col < values[row].size(); ++col) {
            out(row, col) = values[row][col];
        }
    }
    return out;
}

std::vector<abcpp::transform> transformations(
    std::size_t n_params
) {
    return std::vector<abcpp::transform>(
        n_params,
        abcpp::transform::none
    );
}

abcpp::AbcOptions options(
    const ABCControl& control,
    std::size_t n_params
) {
    abcpp::AbcOptions out;
    out.tol = control.tol;
    out.method = abcpp::parse_method(control.method);
    out.kernel = abcpp::parse_kernel(control.kernel);
    out.transformations = transformations(n_params);
    if (!out.transformations.empty()) {
        out.transf = out.transformations.front();
    }
    out.seed = control.seed;
    out.reduction.method = abcpp::parse_reduction(control.reduction);
    out.reduction.n_comp = static_cast<std::size_t>(
        std::max(control.n_comp, 0)
    );
    return out;
}

double sample_param(
    const std::string& name,
    double initial,
    double lower,
    double upper,
    std::mt19937& rng
) {
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        if (name == "alpha" || name == "lapse" || name == "weight") {
            lower = 0.0;
            upper = 1.0;
        } else if (name == "beta") {
            lower = 0.0;
            upper = 10.0;
        } else {
            const double span = std::max(std::abs(initial), 1.0);
            lower = initial - span;
            upper = initial + span;
        }
    }

    if (upper < lower) {
        std::swap(lower, upper);
    }
    if (upper == lower) {
        return lower;
    }

    std::uniform_real_distribution<double> dist(lower, upper);
    return dist(rng);
}

ABCSummaryStats summary_column(const abcpp::SummaryColumn& col) {
    ABCSummaryStats out;
    out.min = col.min;
    out.q_lower = col.q_lower;
    out.median = col.median;
    out.mean = col.mean;
    out.mode = col.mode;
    out.q_upper = col.q_upper;
    out.max = col.max;
    out.sd = col.sd;
    return out;
}

}  // namespace

ABCSubjectResult estimate_abc(
    const RunTask& raw_task,
    const ABCControl& raw_control
) {
    const ABCControl control = modify_control(raw_control, "abc");

    if (raw_task.params.free_names.empty()) {
        throw std::invalid_argument(
            "estimate_abc requires at least one free parameter."
        );
    }

    ABCSubjectResult out;
    out.subid = task_subid(raw_task);
    out.parameter_names = raw_task.params.free_names;
    out.observed_summary = observed_summary(raw_task);
    out.n_simulations = control.samples;
    out.n_comp_used = control.n_comp;

    std::mt19937 rng(control.seed);
    std::vector<std::vector<double>> param_rows;
    std::vector<std::vector<double>> summary_rows;
    param_rows.reserve(static_cast<std::size_t>(control.samples));
    summary_rows.reserve(static_cast<std::size_t>(control.samples));

    for (int sample = 0; sample < control.samples; ++sample) {
        RunTask sim_task = raw_task;
        sim_task.settings.policy = "on";
        sim_task.params.values["seed"] =
            static_cast<double>(control.seed + sample + 1);

        std::vector<double> param_row;
        param_row.reserve(sim_task.params.free_names.size());

        for (std::size_t index = 0; index < sim_task.params.free_names.size();
             ++index) {
            const std::string& name = sim_task.params.free_names[index];
            const double initial = sim_task.params.get(name);
            const double lower = index < control.lower_bounds.size()
                ? control.lower_bounds[index]
                : -std::numeric_limits<double>::infinity();
            const double upper = index < control.upper_bounds.size()
                ? control.upper_bounds[index]
                : std::numeric_limits<double>::infinity();
            const double value = sample_param(
                name,
                initial,
                lower,
                upper,
                rng
            );
            sim_task.params.values[name] = value;
            param_row.push_back(value);
        }

        const RunResult sim_result = process_MDP_free(sim_task);
        param_rows.push_back(param_row);
        summary_rows.push_back(simulated_summary(sim_task, sim_result));
    }

    abcpp::AbcResult abc_result = abcpp::fit(
        out.observed_summary,
        to_matrix(param_rows),
        to_matrix(summary_rows),
        options(control, raw_task.params.free_names.size())
    );
    abc_result.parameter_names = raw_task.params.free_names;

    const abcpp::SummaryResult summary = abcpp::summary(abc_result);
    out.parameter_summary.reserve(summary.columns.size());
    out.estimates.reserve(summary.columns.size());
    for (const abcpp::SummaryColumn& col : summary.columns) {
        out.parameter_summary.push_back(summary_column(col));
        out.estimates.push_back(col.mean);
    }

    out.accepted_distances = abc_result.distances;
    out.accepted_indices = abc_result.accepted_indices;
    out.n_accepted = static_cast<int>(abc_result.accepted_indices.size());
    out.status = abc_result.status == "ok" ? 1 : -1;
    out.message = abc_result.message;
    return out;
}

std::vector<ABCSubjectResult> estimate_abc(
    const std::vector<RunTask>& tasks,
    const ABCControl& raw_control
) {
    const ABCControl control = modify_control(raw_control, "abc");
    std::vector<ABCSubjectResult> out(tasks.size());

#ifdef _OPENMP
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#pragma omp parallel for schedule(dynamic) if(tasks.size() > 1)
#endif
    for (int index = 0; index < static_cast<int>(tasks.size()); ++index) {
        out[static_cast<std::size_t>(index)] =
            estimate_abc(tasks[static_cast<std::size_t>(index)], control);
    }

    return out;
}

}  // namespace multiRL
