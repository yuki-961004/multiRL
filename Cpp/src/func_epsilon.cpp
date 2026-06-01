#include <multiRLcpp/funcs.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace multiRLcpp {

int func_epsilon(const TrialContext& context, const Params& params) {
    const double epsilon = params.get("epsilon");
    const double threshold = params.get("threshold");
    const int rownum = context.rownum;

    std::string model;
    if (std::isnan(epsilon) && threshold > 0.0) {
        model = "first";
    } else if (!std::isnan(epsilon) && threshold == 0.0) {
        model = "decreasing";
    } else if (!std::isnan(epsilon) && threshold == 1.0) {
        model = "greedy";
    } else {
        throw std::invalid_argument("Unknown model in func_epsilon.");
    }

    if (static_cast<double>(rownum) <= threshold) {
        return 1;
    }

    if (model == "first") {
        return 0;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(rownum));
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    if (model == "greedy") {
        return runif(rng) < epsilon ? 1 : 0;
    }

    const double prob_explore = 1.0 / (1.0 + epsilon * rownum);
    return runif(rng) < prob_explore ? 1 : 0;
}

}  // namespace multiRLcpp
