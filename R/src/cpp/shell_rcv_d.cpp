#include <multiRL/shell_rcv_d.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace multiRL {

namespace {

TaskSamplerControl sampler_control(const RecoveryControl& control) {
    TaskSamplerControl out;
    out.n_draws = control.n_draws;
    out.seed = control.seed;
    out.threads = control.threads;
    out.lower_bounds = control.lower_bounds;
    out.upper_bounds = control.upper_bounds;
    return out;
}

std::string task_subid(const RunTask& task, const std::size_t row) {
    if (row < task.input.idinfo.size() && !task.input.idinfo[row].empty()) {
        return task.input.idinfo[row][0];
    }
    return "1";
}

std::size_t task_row_index(
    const RunTask& task,
    const int draw_row
) {
    if (task.input.n_rows == 0) {
        throw std::invalid_argument(
            "shell_rcv_d requires a task with at least one row."
        );
    }
    const int zero_based = draw_row - 1;
    return static_cast<std::size_t>(zero_based) % task.input.n_rows;
}

std::vector<double> params_for_draw(
    const TaskSamplerResult& sampled,
    const int draw
) {
    for (const TaskSamplerRow& row : sampled.rows) {
        if (row.draw == draw) {
            return row.params;
        }
    }
    return std::vector<double>();
}

std::string join_option(const std::vector<std::string>& option) {
    std::string out;
    for (std::size_t index = 0; index < option.size(); ++index) {
        if (index > 0) {
            out += "_";
        }
        out += option[index];
    }
    return out;
}

}  // namespace

RecoveryResult shell_rcv_d(
    const RunTask& task,
    const RecoveryControl& control,
    const std::string& generating_model
) {
    if (control.n_draws < 1) {
        throw std::invalid_argument(
            "shell_rcv_d requires control.n_draws >= 1."
        );
    }

    const TaskSamplerResult sampled = task_sampler(
        task,
        sampler_control(control)
    );

    RecoveryResult out;
    out.parameter_names = sampled.parameter_names;
    out.cue_names = sampled.cue_names;
    out.control = control;
    out.simulation.reserve(sampled.rows.size());
    out.truth.reserve(static_cast<std::size_t>(control.n_draws));

    std::unordered_map<int, bool> seen_draws;

    for (std::size_t index = 0; index < sampled.rows.size(); ++index) {
        const TaskSamplerRow& sampled_row = sampled.rows[index];
        const std::size_t task_row = task_row_index(
            task,
            static_cast<int>(index) + 1
        );

        RecoverySimulationRow row;
        row.generating_model = generating_model;
        row.draw = sampled_row.draw;
        row.subid = sampled_row.subid;
        row.block = task.input.block[task_row];
        row.trial = task.input.trial[task_row];
        row.object.reserve(task.input.state[task_row].size());
        for (const auto& option : task.input.state[task_row]) {
            row.object.push_back(join_option(option));
        }

        row.reward_table = task.input.reward_table[task_row];
        row.action = sampled_row.simulation;
        row.latent = sampled_row.latent;
        row.simulation = sampled_row.simulation;
        row.position = sampled_row.position;
        row.reward = sampled_row.reward;
        row.probability = sampled_row.probability;
        out.simulation.push_back(row);

        if (seen_draws.find(sampled_row.draw) == seen_draws.end()) {
            RecoveryTruthRow truth;
            truth.generating_model = generating_model;
            truth.draw = sampled_row.draw;
            truth.subid = task_subid(task, task_row);
            truth.params = params_for_draw(sampled, sampled_row.draw);
            out.truth.push_back(truth);
            seen_draws[sampled_row.draw] = true;
        }
    }

    return out;
}

}  // namespace multiRL
