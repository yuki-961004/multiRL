#include <multiRLcpp/funcs.hpp>

#include <cmath>

namespace multiRLcpp {

double func_gamma(
    const TrialContext& context,
    double reward,
    const Params& params
) {
    (void) context;

    const double gamma = params.get("gamma");
    const double sign = reward >= 0.0 ? 1.0 : -1.0;
    return sign * std::pow(std::abs(reward), gamma);
}

}  // namespace multiRLcpp
