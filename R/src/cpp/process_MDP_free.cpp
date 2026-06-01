#include <multiRL/process_MDP_free.hpp>

#include <multiRL/criterion_posterior.hpp>
#include <multiRL/funcs.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace multiRL {

namespace {

std::vector<std::string> split_object(const std::string& object) {
    std::vector<std::string> out;
    std::string current;

    for (char value : object) {
        if (value == '_') {
            out.push_back(current);
            current.clear();
        } else {
            current.push_back(value);
        }
    }

    out.push_back(current);
    return out;
}

}  // namespace

/* ========================================================================== *
 *                               Input Builder                                *
 * ========================================================================== */

Process1Input process_1_input(
    const StringMatrix& object,
    const DoubleMatrix& reward,
    const std::vector<std::string>& action,
    const std::vector<int>& block,
    const std::vector<int>& trial,
    const StringMatrix& idinfo,
    const StringMatrix& exinfo
) {
    if (object.empty()) {
        throw std::invalid_argument("process_1_input requires rows.");
    }

    const std::size_t n_rows = object.size();
    const std::size_t n_options = object[0].size();

    if (reward.size() != n_rows ||
        action.size() != n_rows ||
        block.size() != n_rows ||
        trial.size() != n_rows ||
        idinfo.size() != n_rows ||
        exinfo.size() != n_rows) {
        throw std::invalid_argument("process_1_input row sizes do not match.");
    }

    Process1Input input;
    input.state.resize(n_rows);
    input.reward_table = reward;
    input.action = action;
    input.block = block;
    input.trial = trial;
    input.idinfo = idinfo;
    input.exinfo = exinfo;
    input.n_rows = n_rows;
    input.n_options = n_options;

    for (std::size_t row = 0; row < n_rows; ++row) {
        if (object[row].size() != n_options ||
            reward[row].size() != n_options) {
            throw std::invalid_argument(
                "process_1_input option sizes do not match."
            );
        }

        input.state[row].resize(n_options);
        for (std::size_t option = 0; option < n_options; ++option) {
            input.state[row][option] = split_object(object[row][option]);
        }
    }

    return input;
}

/* ========================================================================== *
 *                              Behrule Builder                               *
 * ========================================================================== */

Process2Behrule process_2_behrule(
    const std::vector<std::string>& cue,
    const std::vector<std::string>& rsp
) {
    if (cue.empty() || rsp.empty()) {
        throw std::invalid_argument("behrule must contain cue and rsp.");
    }

    Process2Behrule behrule;
    behrule.cue = cue;
    behrule.rsp = rsp;

    for (std::size_t index = 0; index < cue.size(); ++index) {
        behrule.cue_index[cue[index]] = index;
    }

    return behrule;
}

/* ========================================================================== *
 *                            Record Initialization                           *
 * ========================================================================== */

Process3Record process_3_record(const RunTask& task) {
    const Process1Input& input = task.input;
    const Process2Behrule& behrule = task.behrule;
    const Settings& settings = task.settings;
    const Params& params = task.params;

    const std::size_t n_rows = input.n_rows;
    const std::size_t n_cues = behrule.cue.size();
    const std::size_t n_system = settings.system.size();
    const double q0 = params.get("Q0");
    const double initial_value = std::isnan(q0) ? 0.0 : q0;

    Process3Record record;

    record.value.resize(
        n_system,
        DoubleMatrix(n_rows + 1, std::vector<double>(n_cues, missing_real()))
    );

    for (std::size_t system_index = 0; system_index < n_system;
         ++system_index) {
        record.value[system_index][0] =
            std::vector<double>(n_cues, initial_value);
    }

    record.bias = DoubleMatrix(
        n_rows,
        std::vector<double>(n_cues, missing_real())
    );
    record.shown = record.bias;
    record.prob = record.bias;
    record.count = DoubleMatrix(
        n_rows + 1,
        std::vector<double>(n_cues, missing_real())
    );
    record.count[0] = std::vector<double>(n_cues, 0.0);

    record.behave = StringMatrix(
        n_rows + 1,
        std::vector<std::string>(4, "")
    );

    record.exploration = std::vector<double>(n_rows, missing_real());
    record.latent = std::vector<std::string>(n_rows, "");
    record.reward = std::vector<double>(n_rows, missing_real());
    record.utility = std::vector<double>(n_rows, missing_real());
    record.simulation = std::vector<std::string>(n_rows, "");
    record.position = std::vector<std::string>(n_rows, "");

    return record;
}

namespace {

/* ========================================================================== *
 *                              Trial Context                                 *
 * ========================================================================== */

TrialContext make_context(
    const RunTask& task,
    const Process3Record& record,
    const std::size_t row
) {
    TrialContext context;
    context.row = row;
    context.rownum = static_cast<int>(row + 1);
    context.shown = record.shown[row];
    context.count = record.count[row];
    context.idinfo = task.input.idinfo[row];
    context.exinfo = task.input.exinfo[row];
    context.behave = record.behave[row];
    context.cue = task.behrule.cue;
    context.rsp = task.behrule.rsp;
    context.state = task.input.state[row];
    return context;
}

void update_behave(
    Process3Record& record,
    const std::size_t row,
    const std::string& action,
    const std::string& latent,
    const std::string& simulation,
    const std::string& position
) {
    record.behave[row][0] = action;
    record.behave[row][1] = latent;
    record.behave[row][2] = simulation;
    record.behave[row][3] = position;

    record.behave[row + 1][0] = action;
    record.behave[row + 1][1] = latent;
    record.behave[row + 1][2] = simulation;
    record.behave[row + 1][3] = position;
}

/* ========================================================================== *
 *                            Action Selection                                *
 * ========================================================================== */

std::vector<double> record_shown(
    const std::vector<std::vector<std::string>>& state,
    const std::vector<std::string>& cue
) {
    std::vector<double> shown(cue.size(), missing_real());

    for (std::size_t cue_index = 0; cue_index < cue.size(); ++cue_index) {
        double position = 1.0;
        bool found = false;

        const std::size_t n_options = state.size();
        const std::size_t n_elements = state[0].size();
        for (std::size_t element = 0; element < n_elements; ++element) {
            for (std::size_t option = 0; option < n_options; ++option) {
                if (state[option][element] == cue[cue_index]) {
                    shown[cue_index] = position;
                    found = true;
                    break;
                }
                position += 1.0;
            }
            if (found) {
                break;
            }
        }
    }

    return shown;
}

int find_option_row(
    const std::vector<std::vector<std::string>>& state,
    const std::string& target
) {
    for (std::size_t row = 0; row < state.size(); ++row) {
        for (const std::string& value : state[row]) {
            if (value == target) {
                return static_cast<int>(row);
            }
        }
    }
    return -1;
}

std::string find_simulation(
    const std::vector<std::string>& state_row,
    const std::unordered_set<std::string>& rsp_set
) {
    for (const std::string& value : state_row) {
        if (rsp_set.find(value) != rsp_set.end()) {
            return value;
        }
    }
    return "";
}

std::string sample_choice(
    const std::vector<double>& prob,
    const std::vector<double>& shown,
    const std::vector<std::string>& cue,
    std::mt19937& rng
) {
    double total = 0.0;
    for (std::size_t index = 0; index < prob.size(); ++index) {
        if (!std::isnan(prob[index]) && !std::isnan(shown[index])) {
            total += prob[index];
        }
    }

    if (total <= 0.0) {
        throw std::runtime_error("No valid options to sample from.");
    }

    std::uniform_real_distribution<double> runif(0.0, total);
    const double target = runif(rng);
    double current = 0.0;

    for (std::size_t index = 0; index < prob.size(); ++index) {
        if (!std::isnan(prob[index]) && !std::isnan(shown[index])) {
            current += prob[index];
            if (current >= target) {
                return cue[index];
            }
        }
    }

    return cue.back();
}

}  // namespace

/* ========================================================================== *
 *                             Trial Processor                                *
 * ========================================================================== */

Process3Record process_4_output(
    const RunTask& task,
    Process3Record record
) {
    const Process1Input& input = task.input;
    const Process2Behrule& behrule = task.behrule;
    const Params& params = task.params;
    const Settings& settings = task.settings;

    std::unordered_set<std::string> rsp_set;
    for (const std::string& value : behrule.rsp) {
        rsp_set.insert(value);
    }

    std::mt19937 rng(
        static_cast<std::mt19937::result_type>(params.get("seed"))
    );

    for (std::size_t row = 0; row < input.n_rows; ++row) {
        const auto& state_row = input.state[row];
        record.shown[row] = record_shown(state_row, behrule.cue);
        TrialContext context = make_context(task, record, row);

        record.bias[row] = func_delta(context, params);

        record.exploration[row] = static_cast<double>(
            func_epsilon(context, params)
        );

        std::vector<std::vector<double>> qvalue(settings.system.size());
        for (std::size_t system_index = 0;
             system_index < settings.system.size();
             ++system_index) {
            qvalue[system_index].resize(behrule.cue.size(), missing_real());
            for (std::size_t cue_index = 0;
                 cue_index < behrule.cue.size();
                 ++cue_index) {
                if (std::isnan(record.shown[row][cue_index])) {
                    qvalue[system_index][cue_index] = missing_real();
                    continue;
                }

                double value =
                    record.value[system_index][row][cue_index] +
                    record.bias[row][cue_index];
                if (std::isnan(value)) {
                    value = 0.0;
                }
                qvalue[system_index][cue_index] = value;
            }
        }

        record.prob[row] = func_beta(
            context,
            qvalue,
            record.exploration[row],
            settings.system,
            params
        );

        int option_row = -1;
        if (settings.policy == "off") {
            record.latent[row] = input.action[row];
            record.simulation[row] = input.action[row];
            option_row = find_option_row(state_row, record.latent[row]);
        } else {
            record.latent[row] = sample_choice(
                record.prob[row],
                record.shown[row],
                behrule.cue,
                rng
            );
            option_row = find_option_row(state_row, record.latent[row]);
            if (option_row >= 0) {
                record.simulation[row] = find_simulation(
                    state_row[static_cast<std::size_t>(option_row)],
                    rsp_set
                );
            }
        }

        if (option_row < 0) {
            throw std::runtime_error(
                "Observed or sampled action is absent from state."
            );
        }

        record.position[row] = std::to_string(option_row + 1);
        update_behave(
            record,
            row,
            input.action[row],
            record.latent[row],
            record.simulation[row],
            record.position[row]
        );
        context.behave = record.behave[row];

        record.reward[row] =
            input.reward_table[row][static_cast<std::size_t>(option_row)];
        record.utility[row] = func_gamma(context, record.reward[row], params);

        auto cue_it = behrule.cue_index.find(record.latent[row]);
        if (cue_it == behrule.cue_index.end()) {
            throw std::runtime_error("Latent action is absent from cue.");
        }

        const std::size_t cue_index = cue_it->second;
        const bool is_nb = input.trial[row] == 1;
        const bool is_fp = record.count[row][cue_index] == 0.0;

        for (std::size_t system_index = 0;
             system_index < settings.system.size();
             ++system_index) {
            std::vector<double> next_values = func_zeta(
                context,
                is_nb,
                record.value[system_index][0],
                record.value[system_index][row],
                record.reward[row],
                record.utility[row],
                settings.system[system_index],
                params
            );

            double qi = record.value[system_index][row][cue_index];
            const double reset = params.get("reset");
            if (is_nb && !std::isnan(reset)) {
                qi = next_values[cue_index];
            }

            next_values[cue_index] = func_alpha(
                context,
                is_fp,
                qi,
                record.reward[row],
                record.utility[row],
                settings.system[system_index],
                params
            );

            record.value[system_index][row + 1] = next_values;

            const double q0 = params.get("Q0");
            if (std::isnan(q0) && is_fp) {
                record.value[system_index][0][cue_index] =
                    next_values[cue_index];
            }
        }

        const double reset = params.get("reset");
        if (is_nb && !std::isnan(reset)) {
            record.count[row + 1] =
                std::vector<double>(behrule.cue.size(), 0.0);
        } else {
            record.count[row + 1] = record.count[row];
        }
        record.count[row + 1][cue_index] += 1.0;

    }

    return record;
}

/* ========================================================================== *
 *                              Metric Builder                                *
 * ========================================================================== */

CriterionResult process_5_metric(
    const RunTask& task,
    const Process3Record& output
) {
    return criterion_posterior(task, output);
}

/* ========================================================================== *
 *                         Model-Free MDP Processor                           *
 * ========================================================================== */

RunResult process_MDP_free(const RunTask& task) {
    RunResult result;
    Process3Record record = process_3_record(task);
    result.result = process_4_output(task, record);
    result.metric = process_5_metric(task, result.result);
    return result;
}

}  // namespace multiRL
