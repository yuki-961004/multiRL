#pragma once

#include <multiRL/types.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace multiRL {

class CriterionLikelihood {
public:
    CriterionLikelihood() = default;

    template <typename T>
    CriterionValue<T> evaluate(
        const RunTask& task,
        const Process3Record& output
    ) const {
        return compute<T>(task, output);
    }

    CriterionResult evaluate(
        const RunTask& task,
        const Process3Record& output
    ) const {
        return evaluate<double>(task, output);
    }

    template <typename T>
    CriterionValue<T> operator()(
        const RunTask& task,
        const Process3Record& output
    ) const {
        return evaluate<T>(task, output);
    }

private:
    /* ---------------------------------------------------------------------- *
     * Compare two string vectors for equality
     * ---------------------------------------------------------------------- */

    static bool equal_strings(
        const std::vector<std::string>& lhs,
        const std::vector<std::string>& rhs
    ) {
        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (std::size_t index = 0; index < lhs.size(); ++index) {
            if (lhs[index] != rhs[index]) {
                return false;
            }
        }

        return true;
    }

    /* ---------------------------------------------------------------------- *
     * Core log-likelihood, ACC, AIC, BIC computation
     * ---------------------------------------------------------------------- */

    template <typename T>
    static CriterionValue<T> compute(
        const RunTask& task,
        const Process3Record& output
    ) {
        CriterionValue<T> metric;

        T correct = 0.0;
        for (std::size_t row = 0; row < task.input.n_rows; ++row) {
            if (task.input.action[row] == output.simulation[row]) {
                correct += 1.0;
            }
        }
        metric.acc = correct / static_cast<T>(task.input.n_rows);

        if (!equal_strings(task.behrule.cue, task.behrule.rsp)) {
            return metric;
        }

        T log_likelihood = 0.0;
        for (std::size_t row = 0; row < task.input.n_rows; ++row) {
            auto action_it =
                task.behrule.cue_index.find(task.input.action[row]);
            if (action_it == task.behrule.cue_index.end()) {
                throw std::runtime_error("Action is absent from cue.");
            }

            const T probability = static_cast<T>(
                output.prob[row][action_it->second]
            );
            log_likelihood += std::log(
                std::max(probability, static_cast<T>(0.0))
            );
        }

        const double l_norm = task.params.get("L");
        const double penalty = task.params.get("penalty");
        if (!std::isnan(l_norm)) {
            T regularizer = 0.0;
            for (const std::string& name : task.params.free_names) {
                const T value =
                    static_cast<T>(task.params.get(name));
                if (l_norm == 12.0) {
                    regularizer +=
                        std::abs(value) + value * value;
                } else {
                    regularizer +=
                        std::pow(std::abs(value), l_norm);
                }
            }
            log_likelihood -=
                static_cast<T>(penalty) * regularizer;
        }

        const T n_params =
            static_cast<T>(task.params.free_names.size());
        const T n_rows = static_cast<T>(task.input.n_rows);

        metric.log_likelihood = log_likelihood;
        metric.nll = -log_likelihood;
        metric.aic = 2.0 * n_params - 2.0 * log_likelihood;
        metric.bic = n_params * std::log(n_rows) -
            2.0 * log_likelihood;

        return metric;
    }
};

/* ========================================================================== *
 *                         Convenience Free Function
 * ========================================================================== */

inline CriterionResult criterion_likelihood(
    const RunTask& task,
    const Process3Record& output
) {
    CriterionLikelihood engine;
    return engine.evaluate<double>(task, output);
}

}  // namespace multiRL