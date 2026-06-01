#include <multiRLcpp/criterion_posterior.hpp>

#include <multiRLcpp/criterion_likelihood.hpp>
#include <multiRLcpp/criterion_prior.hpp>

#include <cmath>

namespace multiRLcpp {

CriterionResult criterion_posterior(
    const RunTask& task,
    const Process3Record& output
) {
    CriterionResult result = criterion_likelihood(task, output);
    result.log_prior = criterion_prior(task.params, task.priors);

    if (!std::isnan(result.log_prior)) {
        result.log_posterior = result.log_likelihood + result.log_prior;
    }

    return result;
}

}  // namespace multiRLcpp
