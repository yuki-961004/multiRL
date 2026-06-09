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
#include <string>
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

struct ABCBank {
    std::vector<std::vector<double>> param_rows;
    std::vector<std::vector<double>> summary_rows;
    int n_subjects = 1;
};

ABCBank simulate_bank(
    const RunTask& raw_task,
    const ABCControl& control,
    const std::vector<int>& block,
    const unsigned int seed_offset
) {
    ABCBank bank;
    bank.param_rows.reserve(static_cast<std::size_t>(control.samples));
    bank.summary_rows.reserve(static_cast<std::size_t>(control.samples));

    std::mt19937 rng(control.seed + seed_offset);

    for (int sample = 0; sample < control.samples; ++sample) {
        RunTask sim_task = raw_task;
        sim_task.settings.policy = "on";
        sim_task.params.values["seed"] =
            static_cast<double>(control.seed + seed_offset + sample + 1);

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
        bank.param_rows.push_back(param_row);
        bank.summary_rows.push_back(
            simulated_summary(sim_task, sim_result, block)
        );
    }

    return bank;
}

void append_bank(
    ABCBank& target,
    const ABCBank& source
) {
    if (
        !target.summary_rows.empty() &&
        !source.summary_rows.empty() &&
        target.summary_rows.front().size() != source.summary_rows.front().size()
    ) {
        throw std::invalid_argument(
            "scope = 'universal' requires compatible ABC summary lengths."
        );
    }

    target.param_rows.insert(
        target.param_rows.end(),
        source.param_rows.begin(),
        source.param_rows.end()
    );
    target.summary_rows.insert(
        target.summary_rows.end(),
        source.summary_rows.begin(),
        source.summary_rows.end()
    );
    target.n_subjects += source.n_subjects;
}

ABCSubjectResult fit_with_bank(
    const RunTask& target_task,
    const ABCControl& control,
    const ABCBank& bank,
    const std::vector<int>& block
) {
    ABCSubjectResult out;
    out.subid = task_subid(target_task);
    out.scope = control.scope;
    out.parameter_names = target_task.params.free_names;
    out.n_blocks_real = static_cast<int>(
        unique_sorted_blocks(target_task.input.block).size()
    );
    out.n_blocks_used = static_cast<int>(unique_sorted_blocks(block).size());
    out.fake_block = control.fake_block;
    out.n_comp_requested = control.n_comp;
    out.n_comp_used = control.n_comp > 0 ? control.n_comp : out.n_blocks_used;
    out.observed_summary = observed_summary(target_task, block);
    out.n_simulations = static_cast<int>(bank.param_rows.size());
    out.n_bank_subjects = bank.n_subjects;

    abcpp::AbcResult abc_result = abcpp::fit(
        out.observed_summary,
        to_matrix(bank.param_rows),
        to_matrix(bank.summary_rows),
        options(control, target_task.params.free_names.size(), out.n_comp_used)
    );
    abc_result.parameter_names = target_task.params.free_names;

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

bool same_task_structure(const RunTask& left, const RunTask& right) {
    return left.input.state == right.input.state &&
           left.input.reward_table == right.input.reward_table &&
           left.input.block == right.input.block &&
           left.input.trial == right.input.trial;
}

RunTask shared_target_task(
    const RunTask& templ,
    const RunTask& observed
) {
    if (templ.input.n_rows != observed.input.n_rows) {
        throw std::invalid_argument(
            "scope = 'shared' requires the same number of rows per task."
        );
    }

    RunTask out = templ;
    out.input.action = observed.input.action;
    out.input.idinfo = observed.input.idinfo;
    out.input.exinfo = observed.input.exinfo;
    return out;
}

RunTask pooled_task(const std::vector<RunTask>& tasks) {
    if (tasks.empty()) {
        throw std::invalid_argument("scope = 'universal' received no tasks.");
    }

    RunTask out = tasks.front();
    out.input.state.clear();
    out.input.reward_table.clear();
    out.input.action.clear();
    out.input.block.clear();
    out.input.trial.clear();
    out.input.idinfo.clear();
    out.input.exinfo.clear();
    out.input.n_rows = 0;

    for (const RunTask& task : tasks) {
        out.input.state.insert(
            out.input.state.end(),
            task.input.state.begin(),
            task.input.state.end()
        );
        out.input.reward_table.insert(
            out.input.reward_table.end(),
            task.input.reward_table.begin(),
            task.input.reward_table.end()
        );
        out.input.action.insert(
            out.input.action.end(),
            task.input.action.begin(),
            task.input.action.end()
        );
        out.input.block.insert(
            out.input.block.end(),
            task.input.block.begin(),
            task.input.block.end()
        );
        out.input.trial.insert(
            out.input.trial.end(),
            task.input.trial.begin(),
            task.input.trial.end()
        );
        out.input.idinfo.insert(
            out.input.idinfo.end(),
            task.input.idinfo.begin(),
            task.input.idinfo.end()
        );
        out.input.exinfo.insert(
            out.input.exinfo.end(),
            task.input.exinfo.begin(),
            task.input.exinfo.end()
        );
        out.input.n_rows += task.input.n_rows;
    }

    return out;
}

// Seed-aware estimate_abc for a single subject. Used by the public function
// (with seed_offset = 0) and by the OpenMP parallel loop (with a unique
// seed_offset per subject so each subject gets a distinct, reproducible bank).
ABCSubjectResult estimate_abc_seeded(
    const RunTask& raw_task,
    const ABCControl& raw_control,
    unsigned int seed_offset
) {
    const ABCControl control = modify_control(raw_control, "abc");

    if (raw_task.params.free_names.empty()) {
        throw std::invalid_argument(
            "estimate_abc requires at least one free parameter."
        );
    }

    const std::vector<int> block = summary_blocks(
        raw_task,
        control.fake_block
    );
    const ABCBank bank = simulate_bank(
        raw_task,
        control,
        block,
        seed_offset
    );
    ABCSubjectResult out = fit_with_bank(raw_task, control, bank, block);
    return out;
}

}  // namespace

ABCSubjectResult estimate_abc(
    const RunTask& raw_task,
    const ABCControl& raw_control
) {
    // seed_offset = 0U ensures backward-compatible single-subject behavior
    return estimate_abc_seeded(raw_task, raw_control, 0U);
}

std::vector<ABCSubjectResult> estimate_abc(
    const std::vector<RunTask>& tasks,
    const ABCControl& raw_control
) {
    const ABCControl control = modify_control(raw_control, "abc");

    if (tasks.empty()) {
        return std::vector<ABCSubjectResult>();
    }

    if (control.scope == "shared") {
        const RunTask& templ = tasks.front();
        const std::vector<int> block = summary_blocks(
            templ,
            control.fake_block
        );
        ABCBank bank = simulate_bank(templ, control, block, 0U);
        bank.n_subjects = 1;

        std::vector<ABCSubjectResult> out(tasks.size());
        for (std::size_t index = 0; index < tasks.size(); ++index) {
            RunTask target = shared_target_task(templ, tasks[index]);
            out[index] = fit_with_bank(target, control, bank, block);
            out[index].n_tasks = static_cast<int>(tasks.size());
            out[index].n_subjects = static_cast<int>(tasks.size());
            out[index].n_bank_subjects = 1;
            out[index].shared_template = true;
            out[index].structure_mismatch =
                !same_task_structure(templ, tasks[index]);
        }
        return out;
    }

    if (control.scope == "universal") {
        RunTask target = pooled_task(tasks);
        const std::vector<int> block = summary_blocks(
            target,
            control.fake_block
        );
        ABCBank bank;
        bank.n_subjects = 0;
        for (std::size_t index = 0; index < tasks.size(); ++index) {
            const std::vector<int> task_block = summary_blocks(
                tasks[index],
                control.fake_block
            );
            ABCBank subject_bank = simulate_bank(
                tasks[index],
                control,
                task_block,
                static_cast<unsigned int>(index * control.samples)
            );
            subject_bank.n_subjects = 1;
            append_bank(bank, subject_bank);
        }

        ABCSubjectResult result = fit_with_bank(target, control, bank, block);
        result.subid = "pooled";
        result.n_tasks = 1;
        result.n_subjects = static_cast<int>(tasks.size());
        result.n_bank_subjects = static_cast<int>(tasks.size());
        result.pooled = true;
        return std::vector<ABCSubjectResult>{result};
    }

    std::vector<ABCSubjectResult> out(tasks.size());

#ifdef _OPENMP
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#pragma omp parallel for schedule(dynamic) if(tasks.size() > 1)
#endif
    for (int index = 0; index < static_cast<int>(tasks.size()); ++index) {
        // Each subject gets a unique seed offset so the bank of simulated
        // data is distinct per subject and reproducible across runs.
        const unsigned int seed_offset = static_cast<unsigned int>(index)
            * static_cast<unsigned int>(control.samples);
        out[static_cast<std::size_t>(index)] = estimate_abc_seeded(
            tasks[static_cast<std::size_t>(index)],
            control,
            seed_offset
        );
        out[static_cast<std::size_t>(index)].n_tasks =
            static_cast<int>(tasks.size());
        out[static_cast<std::size_t>(index)].n_subjects =
            static_cast<int>(tasks.size());
    }

    return out;
}

// Seed-aware estimate_abc for a single subject. Used by the public function
// (with seed_offset = 0) and by the OpenMP parallel loop (with a unique
// seed_offset per subject so each subject gets a distinct, reproducible bank).
ABCSubjectResult estimate_abc_seeded(
    const RunTask& raw_task,
    const ABCControl& raw_control,
    unsigned int seed_offset
) {
    const ABCControl control = modify_control(raw_control, "abc");

    if (raw_task.params.free_names.empty()) {
        throw std::invalid_argument(
            "estimate_abc requires at least one free parameter."
        );
    }

    const std::vector<int> block = summary_blocks(
        raw_task,
        control.fake_block
    );
    const ABCBank bank = simulate_bank(
        raw_task,
        control,
        block,
        seed_offset
    );
    ABCSubjectResult out = fit_with_bank(raw_task, control, bank, block);
    return out;
}

}  // namespace multiRL
