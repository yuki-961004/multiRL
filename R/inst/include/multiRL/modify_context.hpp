#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

TrialContext modify_context(
    const RunTask& task,
    const Process3Loop& output,
    std::size_t row,
    const HiddenFeatures& features
);

void modify_context_choice(
    TrialContext& context,
    const Process3Loop& output,
    std::size_t row
);

void modify_context_qvalue(
    TrialContext& context,
    const std::vector<std::vector<double>>& qvalue,
    double exploration,
    const std::vector<std::string>& systems,
    const HiddenFeatures& features
);

void modify_context_outcome(
    TrialContext& context,
    double reward,
    double utility,
    bool is_nb,
    bool is_fp
);

void modify_context_system(
    TrialContext& context,
    const std::vector<double>& value0,
    const std::vector<double>& values,
    double qi,
    const std::string& system
);

}  // namespace multiRL
