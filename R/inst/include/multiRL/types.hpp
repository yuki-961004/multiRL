#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace multiRL {

inline double missing_real() {
    return std::numeric_limits<double>::quiet_NaN();
}

using StringMatrix = std::vector<std::vector<std::string>>;
using DoubleMatrix = std::vector<std::vector<double>>;

struct Process1Input {
    std::vector<std::vector<std::vector<std::string>>> state;
    DoubleMatrix reward_table;
    std::vector<std::string> action;
    std::vector<int> block;
    std::vector<int> trial;
    StringMatrix idinfo;
    StringMatrix exinfo;
    std::size_t n_rows = 0;
    std::size_t n_options = 0;
};

struct Process2Behrule {
    std::vector<std::string> cue;
    std::vector<std::string> rsp;
    std::unordered_map<std::string, std::size_t> cue_index;
};

struct Process3Loop {
    std::vector<DoubleMatrix> value;
    DoubleMatrix bias;
    DoubleMatrix shown;
    DoubleMatrix prob;
    DoubleMatrix count;
    StringMatrix behave;
    std::vector<double> exploration;
    std::vector<std::string> latent;
    std::vector<double> reward;
    std::vector<double> utility;
    std::vector<std::string> simulation;
    std::vector<std::string> position;
};

struct HiddenFeatures {
    double progress = 0.0;
    double block = 0.0;
    double trial = 0.0;
    double block_progress = 0.0;
    double session_progress = 0.0;
    double log_trial_block = 0.0;
    DoubleMatrix before_qvalue;
    DoubleMatrix after_qvalue;
    std::vector<double> before_count;
    std::vector<double> after_count;
    std::vector<double> before_q_max;
    std::vector<double> before_q_mean;
    std::vector<double> before_q_range;
    std::vector<double> before_q_entropy;
    std::vector<double> before_q_chosen;
    std::vector<double> before_q_unchosen;
    std::vector<double> after_q_max;
    std::vector<double> after_q_mean;
    std::vector<double> after_q_range;
    std::vector<double> after_q_entropy;
    std::vector<double> after_q_chosen;
    std::vector<double> after_q_unchosen;
    double prev_reward = 0.0;
    double prev_repeat_sign = 0.0;
    double prev_switch = 0.0;
    double choice_streak = 0.0;
    double pe_prev = 0.0;
    double abs_pe_prev = 0.0;
    double valid_count_total = 0.0;
    double first_trial_in_block = 0.0;
    double reward = missing_real();
    double utility = missing_real();
    double pe = missing_real();
    double abs_pe = missing_real();
    double q_chosen = missing_real();
    double prev_alpha = 0.0;
    double prev_beta = 0.0;
    double prev_gamma = 0.0;
    double prev_delta = 0.0;
    double prev_epsilon = 0.0;
    double prev_zeta = 0.0;
    double prev_q0 = 0.0;
    double prev_lapse = 0.0;
    double prev_sticky = 0.0;
    double prev_weight = 0.0;
    double prev_threshold = 0.0;
    double prev_bonus = 0.0;
    double prev_reset = 0.0;
};

struct HiddenState {
    int prev_choice_index = -1;
    double prev_reward = 0.0;
    double prev_switch = 0.0;
    double prev_repeat_sign = 0.0;
    double choice_streak = 0.0;
    double pe_prev = 0.0;
    double alpha_prev = 0.0;
    double beta_prev = 0.0;
    double gamma_prev = 0.0;
    double delta_prev = 0.0;
    double epsilon_prev = 0.0;
    double zeta_prev = 0.0;
    double q0_prev = 0.0;
    double lapse_prev = 0.0;
    double sticky_prev = 0.0;
    double weight_prev = 0.0;
    double threshold_prev = 0.0;
    double bonus_prev = 0.0;
    double reset_prev = 0.0;
};

struct TrialContext {
    std::size_t row = 0;
    int rownum = 0;
    HiddenFeatures features;
    std::vector<double> shown;
    std::vector<double> count;
    std::vector<std::string> idinfo;
    std::vector<std::string> exinfo;
    std::vector<std::string> behave;
    std::vector<std::string> cue;
    std::vector<std::string> rsp;
    std::vector<std::vector<std::string>> state;
    std::vector<std::string> systems;
    std::vector<std::vector<double>> qvalue;
    std::vector<double> value0;
    std::vector<double> values;
    double exploration = missing_real();
    double reward = missing_real();
    double utility = missing_real();
    double qi = missing_real();
    bool is_nb = false;
    bool is_fp = false;
    std::string system;
};

struct Params {
    std::unordered_map<std::string, double> values;
    std::vector<std::string> free_names;

    bool has(const std::string& name) const;
    double get(const std::string& name) const;
};

struct FunctionConfig {
    std::string lrng_name = "func_alpha";
    std::string prob_name = "func_beta";
    std::string util_name = "func_gamma";
    std::string bias_name = "func_delta";
    std::string expl_name = "func_epsilon";
    std::string dcay_name = "func_zeta";
    bool has_custom = false;

    std::function<double(const TrialContext&, const Params&)> lrng_func;

    std::function<std::vector<double>(
        const TrialContext&,
        const Params&
    )> prob_func;

    std::function<double(const TrialContext&, const Params&)> util_func;

    std::function<std::vector<double>(
        const TrialContext&,
        const Params&
    )> bias_func;

    std::function<int(
        const TrialContext&,
        const Params&
    )> expl_func;

    std::function<std::vector<double>(
        const TrialContext&,
        const Params&
    )> dcay_func;
};

struct Settings {
    std::string name = "unknown";
    std::string mode = "fitting";
    std::string estimate = "MLE";
    bool generate = true;
    std::vector<std::string> system = {"RL"};
};

enum class PriorType {
    NORMAL,
    UNIFORM,
    LOGNORMAL,
    CAUCHY,
    BETA,
    EXPONENTIAL,
    NONE
};

struct PriorSpec {
    PriorType type = PriorType::NONE;
    double param1 = missing_real();
    double param2 = missing_real();
};

struct PriorGroup {
    bool active = false;
    std::unordered_map<std::string, PriorSpec> specs;
};

struct RunTask {
    Process1Input input;
    Process2Behrule behrule;
    Params params;
    Settings settings;
    PriorGroup priors;
    FunctionConfig funcs;
};

template <typename T>
struct CriterionValue {
    T acc = static_cast<T>(missing_real());
    T log_likelihood = static_cast<T>(missing_real());
    T log_prior = static_cast<T>(missing_real());
    T log_posterior = static_cast<T>(missing_real());
    T nll = static_cast<T>(missing_real());
    T aic = static_cast<T>(missing_real());
    T bic = static_cast<T>(missing_real());
};

using CriterionResult = CriterionValue<double>;

struct RunResult {
    Process3Loop result;
    CriterionResult metric;
};

}  // namespace multiRL
