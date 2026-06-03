#include <multiRL/estimate_abc.hpp>

#include <multiRL/process_model_free.hpp>
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
#include <unordered_map>

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

int nearest_divisor(const std::size_t n_rows, const int requested) {
    if (requested <= 1 || n_rows == 0) {
        return 0;
    }

    int best = 1;
    std::size_t best_distance = n_rows;
    for (std::size_t value = 1; value <= n_rows; ++value) {
        if (n_rows % value != 0) {
            continue;
        }

        const std::size_t distance = static_cast<std::size_t>(
            std::abs(static_cast<int>(value) - requested)
        );
        if (distance < best_distance) {
            best = static_cast<int>(value);
            best_distance = distance;
        }
    }
    return best;
}

std::vector<int> unique_sorted_blocks(const std::vector<int>& block) {
    std::vector<int> out = block;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<int> summary_blocks(
    const RunTask& task,
    const int fake_block
) {
    if (fake_block <= 1) {
        return task.input.block;
    }

    const int n_blocks = nearest_divisor(task.input.n_rows, fake_block);
    if (n_blocks <= 1) {
        return task.input.block;
    }

    const std::size_t chunk =
        task.input.n_rows / static_cast<std::size_t>(n_blocks);
    std::vector<int> out(task.input.n_rows, 1);
    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        out[row] = static_cast<int>(row / chunk) + 1;
    }
    return out;
}

std::vector<int> block_index(
    const std::vector<int>& block,
    const std::vector<int>& block_levels
) {
    std::unordered_map<int, int> lookup;
    for (std::size_t index = 0; index < block_levels.size(); ++index) {
        lookup[block_levels[index]] = static_cast<int>(index);
    }

    std::vector<int> out(block.size(), 0);
    for (std::size_t row = 0; row < block.size(); ++row) {
        out[row] = lookup[block[row]];
    }
    return out;
}

std::vector<double> observed_summary(
    const RunTask& task,
    const std::vector<int>& block
) {
    const std::size_t n_cues = task.behrule.cue.size();
    const std::vector<int> levels = unique_sorted_blocks(block);
    const std::vector<int> index = block_index(block, levels);
    const std::size_t n_blocks = levels.size();

    std::vector<double> rows(n_blocks, 0.0);
    std::vector<double> reward_sum(n_blocks, 0.0);
    std::vector<double> valid_reward(n_blocks, 0.0);
    std::vector<std::vector<double>> counts(
        n_blocks,
        std::vector<double>(n_cues, 0.0)
    );

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        const std::size_t block_row =
            static_cast<std::size_t>(index[row]);
        const std::string& action = task.input.action[row];
        rows[block_row] += 1.0;

        const auto cue_it = task.behrule.cue_index.find(action);
        if (cue_it != task.behrule.cue_index.end()) {
            counts[block_row][cue_it->second] += 1.0;
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
                reward_sum[block_row] += task.input.reward_table[row][option];
                valid_reward[block_row] += 1.0;
                break;
            }
        }
    }

    std::vector<double> out;
    out.reserve(n_blocks * (2 + n_cues));
    for (std::size_t block_row = 0; block_row < n_blocks; ++block_row) {
        out.push_back(1.0);
        out.push_back(
            valid_reward[block_row] > 0.0
                ? reward_sum[block_row] / valid_reward[block_row]
                : 0.0
        );

        const double denom = std::max(rows[block_row], 1.0);
        for (double count : counts[block_row]) {
            out.push_back(count / denom);
        }
    }

    return out;
}

std::vector<double> simulated_summary(
    const RunTask& task,
    const RunResult& result,
    const std::vector<int>& block
) {
    const std::size_t n_cues = task.behrule.cue.size();
    const std::vector<int> levels = unique_sorted_blocks(block);
    const std::vector<int> index = block_index(block, levels);
    const std::size_t n_blocks = levels.size();

    std::vector<double> rows(n_blocks, 0.0);
    std::vector<double> reward_sum(n_blocks, 0.0);
    std::vector<double> acc_sum(n_blocks, 0.0);
    std::vector<std::vector<double>> counts(
        n_blocks,
        std::vector<double>(n_cues, 0.0)
    );

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        const std::size_t block_row =
            static_cast<std::size_t>(index[row]);
        const std::string& action = result.result.latent[row];
        const auto cue_it = task.behrule.cue_index.find(action);
        if (cue_it != task.behrule.cue_index.end()) {
            counts[block_row][cue_it->second] += 1.0;
        }
        if (action == task.input.action[row]) {
            acc_sum[block_row] += 1.0;
        }
        reward_sum[block_row] += finite_or(result.result.reward[row], 0.0);
        rows[block_row] += 1.0;
    }

    std::vector<double> out;
    out.reserve(n_blocks * (2 + n_cues));
    for (std::size_t block_row = 0; block_row < n_blocks; ++block_row) {
        const double denom = std::max(rows[block_row], 1.0);
        out.push_back(acc_sum[block_row] / denom);
        out.push_back(reward_sum[block_row] / denom);

        for (double count : counts[block_row]) {
            out.push_back(count / denom);
        }
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
    std::size_t n_params,
    int n_comp_used
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
        std::max(n_comp_used, 0)
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
    const std::vector<int> block = summary_blocks(raw_task, control.fake_block);
    out.n_blocks_real = static_cast<int>(
        unique_sorted_blocks(raw_task.input.block).size()
    );
    out.n_blocks_used = static_cast<int>(unique_sorted_blocks(block).size());
    out.fake_block = control.fake_block;
    out.n_comp_requested = control.n_comp;
    out.n_comp_used = control.n_comp > 0 ? control.n_comp : out.n_blocks_used;
    out.observed_summary = observed_summary(raw_task, block);
    out.n_simulations = control.samples;

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

        const RunResult sim_result = process_model_free(sim_task);
        param_rows.push_back(param_row);
        summary_rows.push_back(simulated_summary(sim_task, sim_result, block));
    }

    abcpp::AbcResult abc_result = abcpp::fit(
        out.observed_summary,
        to_matrix(param_rows),
        to_matrix(summary_rows),
        options(control, raw_task.params.free_names.size(), out.n_comp_used)
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
