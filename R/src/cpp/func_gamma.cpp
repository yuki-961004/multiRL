#include <multiRL/funcs.hpp>

#include <cmath>

namespace multiRL {

double func_gamma(
    const TrialContext& context,
    const Params& params
) {
    const double gamma = params.get("gamma");
    const double reward = context.reward;
    const double sign = reward >= 0.0 ? 1.0 : -1.0;
    return sign * std::pow(std::abs(reward), gamma);
}

}  // namespace multiRL
