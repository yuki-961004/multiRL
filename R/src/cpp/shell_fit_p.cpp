#include <multiRL/shell_fit_p.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace multiRL {

namespace {

std::string normalize_estimator(std::string estimator) {
    std::transform(
        estimator.begin(),
        estimator.end(),
        estimator.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        }
    );
    return estimator;
}

}  // namespace

ShellFitPResult shell_fit_p(
    const std::vector<RunTask>& tasks,
    const ShellFitPControl& control
) {
    ShellFitPResult result;
    result.estimator = normalize_estimator(control.estimator);

    if (result.estimator == "mle") {
        result.mle = estimate_mle(tasks, control.mle);
        return result;
    }

    if (result.estimator == "map") {
        result.map = estimate_map(tasks, control.map);
        return result;
    }

    if (result.estimator == "mcmc") {
        result.mcmc = estimate_mcmc(tasks, control.mcmc);
        return result;
    }

    if (result.estimator == "abc") {
        result.abc = estimate_abc(tasks, control.abc);
        return result;
    }

    if (result.estimator == "rnn") {
        throw std::invalid_argument(
            "estimate_rnn is implemented in the R and Python frontends."
        );
    }

    throw std::invalid_argument(
        "Unknown fit_p estimator. Supported C++ estimators are mle, map, "
        "mcmc, and abc."
    );
}

std::vector<ShellFitPModelResult> shell_fit_p(
    const std::vector<ShellFitPModel>& models,
    const ShellFitPControl& control
) {
    std::vector<ShellFitPModelResult> out;
    out.reserve(models.size());

    for (const ShellFitPModel& model : models) {
        ShellFitPModelResult result;
        result.model = model.model;
        result.model_id = model.model_id;
        result.result = shell_fit_p(model.tasks, control);
        out.push_back(std::move(result));
    }

    return out;
}

}  // namespace multiRL
