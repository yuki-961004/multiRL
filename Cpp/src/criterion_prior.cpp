#include <multiRL/criterion_prior.hpp>

#include <cmath>

namespace multiRL {

template <typename T>
T criterion_prior_value(const Params& params, const PriorGroup& priors) {
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

double criterion_prior(const Params& params, const PriorGroup& priors) {
    return criterion_prior_value<double>(params, priors);
}

template double criterion_prior_value<double>(
    const Params& params,
    const PriorGroup& priors
);

}  // namespace multiRL
