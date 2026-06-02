#pragma once

#include <multiRL/types.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace multiRL {

template <typename T>
inline T criterion_prior_value(const Params& params, const PriorGroup& priors) {
    if (!priors.active) {
        return static_cast<T>(missing_real());
    }

    T log_prior = 0.0;
    const double pi = 3.14159265358979323846;

    using std::log;
    using std::pow;
    using std::sqrt;

    for (const std::string& name : params.free_names) {
        auto spec_it = priors.specs.find(name);
        if (spec_it == priors.specs.end()) {
            continue;
        }

        const PriorSpec& spec = spec_it->second;
        const T value = static_cast<T>(params.get(name));

        switch (spec.type) {
            case PriorType::NORMAL: {
                const double mean = spec.param1;
                const double sd = spec.param2;
                log_prior += -0.5 * log(2.0 * pi * sd * sd) -
                    0.5 * pow((value - mean) / sd, 2.0);
                break;
            }
            case PriorType::UNIFORM: {
                const double lower = spec.param1;
                const double upper = spec.param2;
                if (value >= lower && value <= upper) {
                    log_prior += -log(upper - lower);
                } else {
                    log_prior += -1e10;
                }
                break;
            }
            case PriorType::LOGNORMAL: {
                const double meanlog = spec.param1;
                const double sdlog = spec.param2;
                if (value > 0.0) {
                    log_prior += -log(value * sdlog * sqrt(2.0 * pi)) -
                        0.5 * pow((log(value) - meanlog) / sdlog, 2.0);
                } else {
                    log_prior += -1e10;
                }
                break;
            }
            case PriorType::CAUCHY: {
                const double location = spec.param1;
                const double scale = spec.param2;
                log_prior += -log(pi * scale) -
                    log(1.0 + pow((value - location) / scale, 2.0));
                break;
            }
            case PriorType::BETA: {
                const double shape1 = spec.param1;
                const double shape2 = spec.param2;
                if (value > 0.0 && value < 1.0) {
                    const double log_beta = std::lgamma(shape1) +
                        std::lgamma(shape2) -
                        std::lgamma(shape1 + shape2);
                    log_prior += (shape1 - 1.0) * log(value) +
                        (shape2 - 1.0) * log(1.0 - value) -
                        log_beta;
                } else {
                    log_prior += -1e10;
                }
                break;
            }
            case PriorType::EXPONENTIAL: {
                const double rate = spec.param1;
                if (value >= 0.0) {
                    log_prior += log(rate) - rate * value;
                } else {
                    log_prior += -1e10;
                }
                break;
            }
            case PriorType::NONE:
                break;
        }
    }

    return log_prior;
}

inline double prior_mean_value(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
}

inline double prior_variance_value(
    const std::vector<double>& values,
    const double mean
) {
    if (values.size() < 2) {
        return 0.0;
    }

    double out = 0.0;
    for (const double value : values) {
        out += (value - mean) * (value - mean);
    }

    return out / static_cast<double>(values.size() - 1);
}

inline void update_prior_group(
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

        const double mean = prior_mean_value(values);
        double variance = prior_variance_value(values, mean);
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
            const double log_mean = prior_mean_value(log_values);
            const double log_var = std::max(
                prior_variance_value(log_values, log_mean),
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

inline std::vector<std::unordered_map<std::string, double>>
prior_group_params_from_matrix(
    const std::vector<std::vector<double>>& group_free_params,
    const std::vector<std::string>& free_names
) {
    std::vector<std::unordered_map<std::string, double>> out;
    out.reserve(group_free_params.size());

    for (const std::vector<double>& row : group_free_params) {
        std::unordered_map<std::string, double> subject_params;
        const std::size_t n_values = std::min(row.size(), free_names.size());

        for (std::size_t index = 0; index < n_values; ++index) {
            subject_params[free_names[index]] = row[index];
        }

        out.push_back(subject_params);
    }

    return out;
}

class CriterionPrior {
public:
    CriterionPrior() = default;

    explicit CriterionPrior(const PriorGroup& priors) : priors_(priors) {}

    template <typename T>
    T evaluate(const Params& params) const {
        return criterion_prior_value<T>(params, priors_);
    }

    void update(
        const std::vector<std::unordered_map<std::string, double>>&
            free_params
    ) {
        update_prior_group(priors_, free_params);
    }

    void update(
        const std::vector<std::vector<double>>& group_free_params,
        const std::vector<std::string>& free_names
    ) {
        update(
            prior_group_params_from_matrix(group_free_params, free_names)
        );
    }

    const PriorGroup& group() const {
        return priors_;
    }

    PriorGroup& group() {
        return priors_;
    }

private:
    PriorGroup priors_;
};

inline double criterion_prior(const Params& params, const PriorGroup& priors) {
    CriterionPrior engine(priors);
    return engine.evaluate<double>(params);
}

}  // namespace multiRL
