#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

double func_alpha(
    const TrialContext& context,
    const Params& params
);

std::vector<double> func_beta(
    const TrialContext& context,
    const Params& params
);

double func_gamma(
    const TrialContext& context,
    const Params& params
);

std::vector<double> func_delta(
    const TrialContext& context,
    const Params& params
);

int func_epsilon(const TrialContext& context, const Params& params);

std::vector<double> func_zeta(
    const TrialContext& context,
    const Params& params
);

}  // namespace multiRL
