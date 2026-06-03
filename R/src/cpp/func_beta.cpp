#include <multiRL/funcs.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace multiRL {

std::vector<double> func_beta(
    const TrialContext& context,
    const Params& params
) {
    const double beta = params.get("beta");
    const double lapse = params.get("lapse");
    const double weight_param = params.get("weight");
    const std::vector<std::vector<double>>& qvalue = context.qvalue;
    const double explor = context.exploration;
    const std::vector<std::string>& system = context.systems;
    const std::size_t n_system = qvalue.size();
    const std::size_t n_options = qvalue[0].size();

    std::vector<std::size_t> index;
    for (std::size_t j = 0; j < n_options; ++j) {
        if (!std::isnan(qvalue[0][j])) {
            index.push_back(j);
        }
    }

    const double n_shown = static_cast<double>(index.size());
    std::vector<double> weight;

    if (n_system == 1) {
        weight.push_back(weight_param);
    } else if (n_system == 2) {
        weight.push_back(weight_param);
        weight.push_back(1.0 - weight_param);
    } else {
        weight.assign(n_system, 1.0);
    }

    const double weight_sum =
        std::accumulate(weight.begin(), weight.end(), 0.0);
    for (double& value : weight) {
        value = value / weight_sum;
    }

    std::vector<std::vector<double>> prob_mat(
        n_system,
        std::vector<double>(n_options, missing_real())
    );

    if (explor == 1.0) {
        for (std::size_t s = 0; s < n_system; ++s) {
            for (std::size_t option_index : index) {
                prob_mat[s][option_index] = 1.0 / n_shown;
            }
        }
    } else {
        for (std::size_t s = 0; s < n_system; ++s) {
            double max_value = -std::numeric_limits<double>::infinity();

            for (double value : qvalue[s]) {
                if (!std::isnan(value)) {
                    max_value = std::max(max_value, value);
                }
            }

            double denom = 0.0;
            std::vector<double> exp_value(n_options, missing_real());
            for (std::size_t j = 0; j < n_options; ++j) {
                if (!std::isnan(qvalue[s][j])) {
                    exp_value[j] = std::exp(beta * (qvalue[s][j] - max_value));
                    denom += exp_value[j];
                }
            }

            for (std::size_t j = 0; j < n_options; ++j) {
                if (!std::isnan(exp_value[j])) {
                    prob_mat[s][j] = exp_value[j] / denom;
                }
            }
        }
    }

    std::vector<double> prob(n_options, missing_real());
    for (std::size_t j = 0; j < n_options; ++j) {
        double value = 0.0;
        bool any_missing = false;

        for (std::size_t s = 0; s < n_system; ++s) {
            if (std::isnan(prob_mat[s][j])) {
                any_missing = true;
            } else {
                value += prob_mat[s][j] * weight[s];
            }
        }

        if (!any_missing) {
            prob[j] = (1.0 - lapse * n_shown) * value + lapse;
        }
    }

    (void) system;
    return prob;
}

}  // namespace multiRL
