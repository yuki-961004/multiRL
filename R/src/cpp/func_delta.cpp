#include <multiRLcpp/funcs.hpp>

#include <cmath>
#include <string>

namespace multiRLcpp {

namespace {

bool is_missing_string(const std::string& value) {
    return value.empty() || value == "NA";
}

bool row_contains(
    const std::vector<std::string>& row,
    const std::string& target
) {
    for (const std::string& value : row) {
        if (value == target) {
            return true;
        }
    }
    return false;
}

}  // namespace

std::vector<double> func_delta(
    const TrialContext& context,
    const Params& params
) {
    const double delta = params.get("delta");
    const double sticky = params.get("sticky");
    const std::vector<double>& shown = context.shown;
    const std::vector<double>& count = context.count;
    const std::vector<std::vector<std::string>>& state = context.state;
    const std::vector<std::string>& cue = context.cue;
    const std::string last_latent =
        context.behave.size() > 1 ? context.behave[1] : "";
    const std::string last_simulation =
        context.behave.size() > 2 ? context.behave[2] : "";
    const std::string last_position =
        context.behave.size() > 3 ? context.behave[3] : "";

    std::vector<double> bias(cue.size(), missing_real());

    for (std::size_t j = 0; j < cue.size(); ++j) {
        double last_latent_value = std::isnan(shown[j]) ? missing_real() : 0.0;
        double last_simulation_value =
            std::isnan(shown[j]) ? missing_real() : 0.0;
        double last_position_value =
            std::isnan(shown[j]) ? missing_real() : 0.0;

        if (!is_missing_string(last_latent) && !std::isnan(shown[j])) {
            last_latent_value = cue[j] == last_latent ? 1.0 : 0.0;
        }

        if (!is_missing_string(last_simulation) && !std::isnan(shown[j])) {
            for (const auto& row : state) {
                if (row_contains(row, cue[j]) &&
                    row_contains(row, last_simulation)) {
                    last_simulation_value = 1.0;
                }
            }
        }

        if (!is_missing_string(last_position) && !std::isnan(shown[j])) {
            const double position = std::stod(last_position);
            last_position_value = shown[j] == position ? 1.0 : 0.0;
        }

        const double ucb =
            delta *
            std::sqrt(std::log(count[j] + std::exp(1.0)) /
                      (count[j] + 1e-10));

        bias[j] = ucb +
            sticky * last_latent_value +
            sticky * last_simulation_value +
            sticky * last_position_value;
    }

    return bias;
}

}  // namespace multiRLcpp
