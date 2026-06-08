#include <multiRL/process_model_free.hpp>

#include <multiRL/criterion_posterior.hpp>
#include <multiRL/modify_context.hpp>
#include <multiRL/modify_funcs.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace multiRL {



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

namespace {

/* ========================================================================== *
 *                             String Utilities                               *
 * ========================================================================== */

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

/* ========================================================================== *
 *                            Record Initialization                           *
 * ========================================================================== */

Process3Loop initialize_loop(const RunTask& task) {
    const Process1Input& input = task.input;
    const Process2Behrule& behrule = task.behrule;
    const Settings& settings = task.settings;
    const Params& params = task.params;

    const std::size_t n_rows = input.n_rows;
    const std::size_t n_cues = behrule.cue.size();
    const std::size_t n_system = settings.system.size();
    const double q0 = params.get("Q0");
    const double initial_value = std::isnan(q0) ? 0.0 : q0;

    Process3Loop output;

    output.value.resize(
        n_system,
        DoubleMatrix(n_rows + 1, std::vector<double>(n_cues, missing_real()))
    );

    for (std::size_t system_index = 0; system_index < n_system;
         ++system_index) {
        output.value[system_index][0] =
            std::vector<double>(n_cues, initial_value);
    }

    output.bias = DoubleMatrix(
        n_rows,
        std::vector<double>(n_cues, missing_real())
    );
    output.shown = output.bias;
    output.prob = output.bias;
    output.count = DoubleMatrix(
        n_rows + 1,
        std::vector<double>(n_cues, missing_real())
    );
    output.count[0] = std::vector<double>(n_cues, 0.0);

    output.behave = StringMatrix(
        n_rows + 1,
        std::vector<std::string>(4, "")
    );

    output.exploration = std::vector<double>(n_rows, missing_real());
    output.latent = std::vector<std::string>(n_rows, "");
    output.reward = std::vector<double>(n_rows, missing_real());
    output.utility = std::vector<double>(n_rows, missing_real());
    output.simulation = std::vector<std::string>(n_rows, "");
    output.position = std::vector<std::string>(n_rows, "");

    return output;
}

void update_behave(
    Process3Loop& output,
    const std::size_t row,
    const std::string& action,
    const std::string& latent,
    const std::string& simulation,
    const std::string& position
) {
    output.behave[row][0] = action;
    output.behave[row][1] = latent;
    output.behave[row][2] = simulation;
    output.behave[row][3] = position;

    output.behave[row + 1][0] = action;
    output.behave[row + 1][1] = latent;
    output.behave[row + 1][2] = simulation;
    output.behave[row + 1][3] = position;
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
 *                             Trial Loop                                     *
 * ========================================================================== */

Process3Loop process_3_loop(const RunTask& task) {
    const Process1Input& input = task.input;
    const Process2Behrule& behrule = task.behrule;
    const Params& params = task.params;
    const Settings& settings = task.settings;
    const FunctionConfig default_funcs = modify_funcs();
    const FunctionConfig& funcs = task.funcs.lrng_func
        ? task.funcs
        : default_funcs;

    Process3Loop output = initialize_loop(task);

    std::unordered_set<std::string> rsp_set;
    for (const std::string& value : behrule.rsp) {
        rsp_set.insert(value);
    }

    std::mt19937 rng(
        static_cast<std::mt19937::result_type>(params.get("seed"))
    );

    for (std::size_t row = 0; row < input.n_rows; ++row) {
        const auto& state_row = input.state[row];
        output.shown[row] = record_shown(state_row, behrule.cue);
        TrialContext context = modify_context(task, output, row);

        /* ------------------------------------------------------------------ *
         * Bias function: bias_func                                            *
         * ------------------------------------------------------------------ */

        output.bias[row] = funcs.bias_func(context, params);

        /* ------------------------------------------------------------------ *
         * Exploration function: expl_func                                     *
         * ------------------------------------------------------------------ */

        output.exploration[row] = static_cast<double>(
            funcs.expl_func(context, params)
        );

        std::vector<std::vector<double>> qvalue(settings.system.size());
        for (std::size_t system_index = 0;
             system_index < settings.system.size();
             ++system_index) {
            qvalue[system_index].resize(behrule.cue.size(), missing_real());
            for (std::size_t cue_index = 0;
                 cue_index < behrule.cue.size();
                 ++cue_index) {
                if (std::isnan(output.shown[row][cue_index])) {
                    qvalue[system_index][cue_index] = missing_real();
                    continue;
                }

                double value =
                    output.value[system_index][row][cue_index] +
                    output.bias[row][cue_index];
                if (std::isnan(value)) {
                    value = 0.0;
                }
                qvalue[system_index][cue_index] = value;
            }
        }

        /* ------------------------------------------------------------------ *
         * Choice probability function: prob_func                              *
         * ------------------------------------------------------------------ */

        modify_context_qvalue(
            context,
            qvalue,
            output.exploration[row],
            settings.system
        );

        output.prob[row] = funcs.prob_func(
            context,
            params
        );

        int option_row = -1;
        if (settings.policy == "off") {
            output.latent[row] = input.action[row];
            output.simulation[row] = input.action[row];
            option_row = find_option_row(state_row, output.latent[row]);
        } else {
            output.latent[row] = sample_choice(
                output.prob[row],
                output.shown[row],
                behrule.cue,
                rng
            );
            option_row = find_option_row(state_row, output.latent[row]);
            if (option_row >= 0) {
                output.simulation[row] = find_simulation(
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

        output.position[row] = std::to_string(option_row + 1);
        update_behave(
            output,
            row,
            input.action[row],
            output.latent[row],
            output.simulation[row],
            output.position[row]
        );
        modify_context_choice(context, output, row);

        output.reward[row] =
            input.reward_table[row][static_cast<std::size_t>(option_row)];

        /* ------------------------------------------------------------------ *
         * Utility function: util_func                                         *
         * ------------------------------------------------------------------ */

        context.reward = output.reward[row];
        output.utility[row] = funcs.util_func(context, params);

        auto cue_it = behrule.cue_index.find(output.latent[row]);
        if (cue_it == behrule.cue_index.end()) {
            throw std::runtime_error("Latent action is absent from cue.");
        }

        const std::size_t cue_index = cue_it->second;
        const bool is_nb = input.trial[row] == 1;
        const bool is_fp = output.count[row][cue_index] == 0.0;

        modify_context_outcome(
            context,
            output.reward[row],
            output.utility[row],
            is_nb,
            is_fp
        );

        for (std::size_t system_index = 0;
             system_index < settings.system.size();
             ++system_index) {
            modify_context_system(
                context,
                output.value[system_index][0],
                output.value[system_index][row],
                output.value[system_index][row][cue_index],
                settings.system[system_index]
            );

            /* -------------------------------------------------------------- *
             * Decay function: dcay_func                                      *
             * -------------------------------------------------------------- */

            std::vector<double> next_values = funcs.dcay_func(
                context,
                params
            );

            const double reset = params.get("reset");
            if (is_nb && !std::isnan(reset)) {
                context.qi = next_values[cue_index];
            }

            /* -------------------------------------------------------------- *
             * Learning function: lrng_func                                   *
             * -------------------------------------------------------------- */

            next_values[cue_index] = funcs.lrng_func(
                context,
                params
            );

            output.value[system_index][row + 1] = next_values;

            const double q0 = params.get("Q0");
            if (std::isnan(q0) && is_fp) {
                output.value[system_index][0][cue_index] =
                    next_values[cue_index];
            }
        }

        const double reset = params.get("reset");
        if (is_nb && !std::isnan(reset)) {
            output.count[row + 1] =
                std::vector<double>(behrule.cue.size(), 0.0);
        } else {
            output.count[row + 1] = output.count[row];
        }
        output.count[row + 1][cue_index] += 1.0;

    }

    return output;
}

/* ========================================================================== *
 *                         Model-Free MDP Processor                           *
 * ========================================================================== */

RunResult process_model_free(const RunTask& task) {
    RunResult result;

    auto output = process_3_loop(task);
    auto metric = criterion_posterior(task, output);

    result.result = output;
    result.metric = metric;
    return result;
}

}  // namespace multiRL
