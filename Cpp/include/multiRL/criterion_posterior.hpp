#pragma once

#include <multiRL/types.hpp>
#include <multiRL/criterion_likelihood.hpp>
#include <multiRL/criterion_prior.hpp>

#include <cmath>

namespace multiRL {

class CriterionPosterior {
public:
    CriterionPosterior() = default;

    explicit CriterionPosterior(const PriorGroup& priors)
        : prior_(priors) {}

    CriterionResult evaluate(
        const RunTask& task,
        const Process3Record& output
    ) const {
        CriterionResult result = likelihood_.evaluate(task, output);
        result.log_prior = prior_.evaluate<double>(task.params);

        if (!std::isnan(result.log_prior)) {
            result.log_posterior = result.log_likelihood + result.log_prior;
        }

        return result;
    }

    CriterionResult operator()(
        const RunTask& task,
        const Process3Record& output
    ) const {
        return evaluate(task, output);
    }

    CriterionPrior& prior() {
        return prior_;
    }

    const CriterionPrior& prior() const {
        return prior_;
    }

private:
    CriterionLikelihood likelihood_;
    CriterionPrior prior_;
};

inline CriterionResult criterion_posterior(
    const RunTask& task,
    const Process3Record& output
) {
    CriterionPosterior engine(task.priors);
    return engine.evaluate(task, output);
}

}  // namespace multiRL
