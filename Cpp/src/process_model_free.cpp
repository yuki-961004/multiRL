#include <multiRL/process_model_free.hpp>

#include <multiRL/criterion_posterior.hpp>
#include <multiRL/modify_context.hpp>
#include <multiRL/modify_funcs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace multiRL {

namespace {

std::vector<std::string> split_object(const std::string& object);

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

/* ========================================================================== *
 *                              Hidden Features                               *
 * ========================================================================== */

double finite_or_zero(double value) {
    return std::isnan(value) ? 0.0 : value;
}

double progress_fraction(int value, int maximum) {
    if (maximum <= 1) {
        return 0.0;
    }
    return static_cast<double>(value - 1) / static_cast<double>(maximum - 1);
}

std::vector<double> finite_qvalues(const std::vector<double>& values) {
    std::vector<double> out;
    for (double value : values) {
        if (!std::isnan(value)) {
            out.push_back(value);
        }
    }
    return out;
}

double entropy_from_qvalues(const std::vector<double>& values) {
    const std::vector<double> finite = finite_qvalues(values);
    if (finite.size() <= 1) {
        return 0.0;
    }

    const double max_value =
        *std::max_element(finite.begin(), finite.end());
    double denominator = 0.0;
    for (double value : finite) {
        denominator += std::exp(value - max_value);
    }
    if (denominator <= 0.0) {
        return 0.0;
    }

    double entropy = 0.0;
    for (double value : finite) {
        const double prob = std::exp(value - max_value) / denominator;
        entropy -= prob * std::log(prob + 1e-12);
    }
    return entropy / std::log(static_cast<double>(finite.size()));
}

double previous_chosen_q(
    const std::vector<double>& qvalue,
    int previous_choice
) {
    if (previous_choice < 0) {
        return 0.0;
    }
    const std::size_t index = static_cast<std::size_t>(previous_choice);
    if (index >= qvalue.size()) {
        return 0.0;
    }
    return finite_or_zero(qvalue[index]);
}

double previous_unchosen_q(
    const std::vector<double>& qvalue,
    int previous_choice
) {
    if (previous_choice < 0 || qvalue.size() <= 1) {
        return 0.0;
    }

    if (qvalue.size() == 2) {
        const std::size_t index =
            static_cast<std::size_t>(1 - previous_choice);
        return index < qvalue.size() ? finite_or_zero(qvalue[index]) : 0.0;
    }

    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < qvalue.size(); ++index) {
        if (static_cast<int>(index) == previous_choice) {
            continue;
        }
        if (!std::isnan(qvalue[index])) {
            best = std::max(best, qvalue[index]);
        }
    }
    return std::isfinite(best) ? best : 0.0;
}

HiddenFeatures base_hidden_features(
    const RunTask& task,
    const Process3Loop& output,
    const HiddenState& state,
    std::size_t row,
    int max_block,
    int max_trial
) {
    HiddenFeatures features;
    features.progress = progress_fraction(
        static_cast<int>(row + 1),
        static_cast<int>(task.input.n_rows)
    );
    features.block = static_cast<double>(task.input.block[row]);
    features.trial = static_cast<double>(task.input.trial[row]);
    features.block_progress =
        progress_fraction(task.input.block[row], max_block);
    features.session_progress = features.progress;
    if (max_trial > 0) {
        features.log_trial_block =
            std::log(static_cast<double>(task.input.trial[row] + 1)) /
            std::log(static_cast<double>(max_trial + 1));
    }

    features.prev_reward = state.prev_reward;
    features.prev_repeat_sign = state.prev_repeat_sign;
    features.prev_switch = state.prev_switch;
    features.choice_streak = state.choice_streak;
    features.pe_prev = state.pe_prev;
    features.abs_pe_prev = std::abs(state.pe_prev);
    features.first_trial_in_block = task.input.trial[row] == 1 ? 1.0 : 0.0;
    features.prev_alpha = state.alpha_prev;
    features.prev_beta = state.beta_prev;
    features.prev_gamma = state.gamma_prev;
    features.prev_delta = state.delta_prev;
    features.prev_epsilon = state.epsilon_prev;
    features.prev_zeta = state.zeta_prev;
    features.prev_q0 = state.q0_prev;
    features.prev_lapse = state.lapse_prev;
    features.prev_sticky = state.sticky_prev;
    features.prev_weight = state.weight_prev;
    features.prev_threshold = state.threshold_prev;
    features.prev_bonus = state.bonus_prev;
    features.prev_reset = state.reset_prev;

    const std::vector<double>& count = output.count[row];
    features.before_count = count;
    features.valid_count_total = 0.0;
    for (double value : count) {
        if (!std::isnan(value)) {
            features.valid_count_total += value;
        }
    }

    return features;
}

void append_qvalue_statistics(
    const DoubleMatrix& qvalue,
    int chosen_index,
    std::vector<double>& maximum,
    std::vector<double>& mean,
    std::vector<double>& range,
    std::vector<double>& entropy,
    std::vector<double>& chosen,
    std::vector<double>& unchosen
) {
    maximum.clear();
    mean.clear();
    range.clear();
    entropy.clear();
    chosen.clear();
    unchosen.clear();

    for (const std::vector<double>& values : qvalue) {
        const std::vector<double> finite = finite_qvalues(values);
        if (finite.empty()) {
            maximum.push_back(missing_real());
            mean.push_back(missing_real());
            range.push_back(missing_real());
        } else {
            const double max_value =
                *std::max_element(finite.begin(), finite.end());
            const double min_value =
                *std::min_element(finite.begin(), finite.end());
            maximum.push_back(max_value);
            mean.push_back(
                std::accumulate(finite.begin(), finite.end(), 0.0) /
                static_cast<double>(finite.size())
            );
            range.push_back(max_value - min_value);
        }
        entropy.push_back(entropy_from_qvalues(values));
        chosen.push_back(previous_chosen_q(values, chosen_index));
        unchosen.push_back(previous_unchosen_q(values, chosen_index));
    }
}

void set_before_qvalue_features(
    HiddenFeatures& features,
    const HiddenState& state,
    const DoubleMatrix& qvalue
) {
    features.before_qvalue = qvalue;
    append_qvalue_statistics(
        qvalue,
        state.prev_choice_index,
        features.before_q_max,
        features.before_q_mean,
        features.before_q_range,
        features.before_q_entropy,
        features.before_q_chosen,
        features.before_q_unchosen
    );
}

void set_after_qvalue_features(
    HiddenFeatures& features,
    const DoubleMatrix& qvalue,
    std::size_t choice_index,
    const std::vector<double>& count
) {
    features.after_qvalue = qvalue;
    features.after_count = count;
    append_qvalue_statistics(
        qvalue,
        static_cast<int>(choice_index),
        features.after_q_max,
        features.after_q_mean,
        features.after_q_range,
        features.after_q_entropy,
        features.after_q_chosen,
        features.after_q_unchosen
    );
}

void reset_hidden_state(HiddenState& state) {
    state.prev_choice_index = -1;
    state.prev_reward = 0.0;
    state.prev_switch = 0.0;
    state.prev_repeat_sign = 0.0;
    state.choice_streak = 0.0;
    state.pe_prev = 0.0;
    state.alpha_prev = 0.0;
    state.beta_prev = 0.0;
    state.gamma_prev = 0.0;
    state.delta_prev = 0.0;
    state.epsilon_prev = 0.0;
    state.zeta_prev = 0.0;
    state.q0_prev = 0.0;
    state.lapse_prev = 0.0;
    state.sticky_prev = 0.0;
    state.weight_prev = 0.0;
    state.threshold_prev = 0.0;
    state.bonus_prev = 0.0;
    state.reset_prev = 0.0;
}

double parameter_or_zero(const Params& params, const std::string& name) {
    if (!params.has(name)) {
        return 0.0;
    }
    return finite_or_zero(params.get(name));
}

double effective_alpha(
    const RunTask& task,
    const Process3Loop& output,
    std::size_t row,
    std::size_t cue_index
) {
    const double q0 = task.params.get("Q0");
    if (std::isnan(q0) && output.count[row][cue_index] == 0.0) {
        return 0.0;
    }
    if (task.params.has("alpha")) {
        return parameter_or_zero(task.params, "alpha");
    }
    if (!task.params.has("alphaN") || !task.params.has("alphaP")) {
        return 0.0;
    }

    for (std::size_t system_index = 0;
         system_index < task.settings.system.size();
         ++system_index) {
        if (task.settings.system[system_index] != "RL") {
            continue;
        }
        const double q_before = output.value[system_index][row][cue_index];
        return output.utility[row] < q_before
            ? parameter_or_zero(task.params, "alphaN")
            : parameter_or_zero(task.params, "alphaP");
    }
    return 0.0;
}

void update_hidden_state(
    HiddenState& state,
    const RunTask& task,
    const Process3Loop& output,
    std::size_t row,
    std::size_t cue_index
) {
    const int previous_choice = state.prev_choice_index;
    const bool had_previous = previous_choice >= 0;
    const bool repeated =
        had_previous && previous_choice == static_cast<int>(cue_index);

    state.prev_switch = had_previous && !repeated ? 1.0 : 0.0;
    state.prev_repeat_sign = had_previous
        ? (repeated ? 1.0 : -1.0)
        : 0.0;
    state.choice_streak = repeated ? state.choice_streak + 1.0 : 1.0;
    state.prev_choice_index = static_cast<int>(cue_index);
    state.prev_reward = output.reward[row];

    const double q_before = output.value[0][row][cue_index];
    state.pe_prev = output.utility[row] - q_before;
    state.alpha_prev = effective_alpha(task, output, row, cue_index);
    state.beta_prev = parameter_or_zero(task.params, "beta");
    state.gamma_prev = parameter_or_zero(task.params, "gamma");
    state.delta_prev = parameter_or_zero(task.params, "delta");
    state.epsilon_prev = parameter_or_zero(task.params, "epsilon");
    state.zeta_prev = parameter_or_zero(task.params, "zeta");
    state.q0_prev = parameter_or_zero(task.params, "Q0");
    state.lapse_prev = parameter_or_zero(task.params, "lapse");
    state.sticky_prev = parameter_or_zero(task.params, "sticky");
    state.weight_prev = parameter_or_zero(task.params, "weight");
    state.threshold_prev = parameter_or_zero(task.params, "threshold");
    state.bonus_prev = parameter_or_zero(task.params, "bonus");
    state.reset_prev = parameter_or_zero(task.params, "reset");
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

    const int max_block =
        *std::max_element(input.block.begin(), input.block.end());
    const int max_trial =
        *std::max_element(input.trial.begin(), input.trial.end());
    HiddenState hidden;

    for (std::size_t row = 0; row < input.n_rows; ++row) {
        if (input.trial[row] == 1) {
            reset_hidden_state(hidden);
        }

        const auto& state_row = input.state[row];
        output.shown[row] = record_shown(state_row, behrule.cue);
        HiddenFeatures features = base_hidden_features(
            task,
            output,
            hidden,
            row,
            max_block,
            max_trial
        );
        TrialContext context = modify_context(task, output, row, features);

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

        set_before_qvalue_features(features, hidden, qvalue);
        modify_context_qvalue(
            context,
            qvalue,
            output.exploration[row],
            settings.system,
            features
        );

        output.prob[row] = funcs.prob_func(
            context,
            params
        );

        int option_row = -1;
        if (!settings.generate) {
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

        std::vector<double> after_count = output.count[row];
        const double reset = params.get("reset");
        if (is_nb && !std::isnan(reset)) {
            after_count.assign(behrule.cue.size(), 0.0);
        }
        after_count[cue_index] += 1.0;
        set_after_qvalue_features(
            context.features,
            qvalue,
            cue_index,
            after_count
        );

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

        update_hidden_state(
            hidden,
            task,
            output,
            row,
            cue_index
        );

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
