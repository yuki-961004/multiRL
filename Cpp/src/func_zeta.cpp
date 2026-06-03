#include <multiRL/funcs.hpp>

#include <cmath>

namespace multiRL {

std::vector<double> func_zeta(
    const TrialContext& context,
    const Params& params
) {
    (void) context.utility;
    (void) context.system;

    const double zeta = params.get("zeta");
    const double bonus = params.get("bonus");
    const double reset = params.get("reset");
    const bool is_nb = context.is_nb;
    const std::vector<double>& value0 = context.value0;
    const std::vector<double>& values = context.values;
    const double reward = context.reward;

    if (is_nb && !std::isnan(reset)) {
        return std::vector<double>(values.size(), reset);
    }

    std::vector<double> decay(values.size(), missing_real());
    for (std::size_t index = 0; index < values.size(); ++index) {
        decay[index] = values[index] + zeta * (value0[index] - values[index]);

        if (reward < 0.0) {
            decay[index] += bonus;
        } else if (reward > 0.0) {
            decay[index] -= bonus;
        }
    }

    return decay;
}

}  // namespace multiRL
