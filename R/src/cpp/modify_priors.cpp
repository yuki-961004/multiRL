#include <multiRL/modify_priors.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace multiRL {

namespace {

PriorType prior_type_from_string(const std::string& type) {
    if (type == "normal" || type == "norm") {
        return PriorType::NORMAL;
    }
    if (type == "uniform" || type == "unif") {
        return PriorType::UNIFORM;
    }
    if (type == "lognormal" || type == "lnorm") {
        return PriorType::LOGNORMAL;
    }
    if (type == "cauchy") {
        return PriorType::CAUCHY;
    }
    if (type == "beta") {
        return PriorType::BETA;
    }
    if (type == "exponential" || type == "exp") {
        return PriorType::EXPONENTIAL;
    }
    if (type == "none") {
        return PriorType::NONE;
    }

    throw std::invalid_argument("Unknown prior type.");
}

double mean_value(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
}

double variance_value(const std::vector<double>& values, const double mean) {
    if (values.size() < 2) {
        return 0.0;
    }

    double out = 0.0;
    for (const double value : values) {
        out += (value - mean) * (value - mean);
    }

    return out / static_cast<double>(values.size() - 1);
}

}  // namespace

/* ========================================================================== *
 *                         Prior Standardization                              *
 * ========================================================================== */

PriorGroup modify_priors(
    const std::vector<std::string>& names,
    const std::vector<std::string>& types,
    const std::vector<double>& param1,
    const std::vector<double>& param2,
    const std::vector<std::string>& free_names,
    bool active
) {
    PriorGroup out;
    out.active = active;

    if (!active) {
        return out;
    }

    if (names.size() != types.size() ||
        names.size() != param1.size() ||
        names.size() != param2.size()) {
        throw std::invalid_argument("Prior specification sizes do not match.");
    }

    std::unordered_set<std::string> free_set;
    for (const std::string& name : free_names) {
        free_set.insert(name);
    }

    for (std::size_t index = 0; index < names.size(); ++index) {
        if (free_set.find(names[index]) == free_set.end()) {
            throw std::invalid_argument(
                "Prior names must match free parameter names."
            );
        }

        PriorSpec spec;
        spec.type = prior_type_from_string(types[index]);
        spec.param1 = param1[index];
        spec.param2 = param2[index];
        out.specs[names[index]] = spec;
    }

    return out;
}

/* ========================================================================== *
 *                         Empirical Prior Update                             *
 * ========================================================================== */

void update_priors(
    PriorGroup& priors,
    const std::vector<std::unordered_map<std::string, double>>& free_params
) {
    if (!priors.active || free_params.size() < 2) {
        return;
    }

    const double min_var = 1e-6;

    for (auto& kv : priors.specs) {
        const std::string& name = kv.first;
        PriorSpec& spec = kv.second;

        if (spec.type == PriorType::NONE || spec.type == PriorType::UNIFORM) {
            continue;
        }

        std::vector<double> values;
        values.reserve(free_params.size());
        for (const auto& subject_params : free_params) {
            auto it = subject_params.find(name);
            if (it != subject_params.end()) {
                values.push_back(it->second);
            }
        }

        if (values.size() < 2) {
            continue;
        }

        const double mean = mean_value(values);
        double variance = variance_value(values, mean);
        if (variance < min_var) {
            variance = min_var;
        }

        if (spec.type == PriorType::NORMAL) {
            spec.param1 = mean;
            spec.param2 = std::sqrt(variance);
        } else if (spec.type == PriorType::LOGNORMAL) {
            std::vector<double> log_values(values.size());
            for (std::size_t index = 0; index < values.size(); ++index) {
                log_values[index] = std::log(std::max(values[index], 1e-8));
            }
            const double log_mean = mean_value(log_values);
            const double log_var = std::max(
                variance_value(log_values, log_mean),
                min_var
            );
            spec.param1 = log_mean;
            spec.param2 = std::sqrt(log_var);
        } else if (spec.type == PriorType::BETA) {
            const double beta_mean = std::max(
                1e-4,
                std::min(mean, 1.0 - 1e-4)
            );
            const double max_var = beta_mean * (1.0 - beta_mean) - 1e-6;
            variance = std::min(variance, max_var);

            const double common =
                (beta_mean * (1.0 - beta_mean) / variance) - 1.0;
            spec.param1 = std::max(beta_mean * common, 1e-3);
            spec.param2 = std::max((1.0 - beta_mean) * common, 1e-3);
        } else if (spec.type == PriorType::EXPONENTIAL) {
            spec.param1 = 1.0 / std::max(mean, 1e-6);
            spec.param2 = missing_real();
        } else if (spec.type == PriorType::CAUCHY) {
            std::sort(values.begin(), values.end());
            const std::size_t n_values = values.size();
            spec.param1 = (n_values % 2 == 0) ?
                (values[n_values / 2 - 1] + values[n_values / 2]) / 2.0 :
                values[n_values / 2];
            const double iqr =
                values[n_values * 3 / 4] - values[n_values / 4];
            spec.param2 = std::max(iqr / 2.0, 1e-4);
        }
    }
}

}  // namespace multiRL
