#pragma once

#include <multiRL/estimate_mle.hpp>

#include <vector>

#ifdef MULTIRL_HAS_NLOPT
#include <nlopt.hpp>
#endif

namespace multiRL {

namespace FreeValues {
    void assign(
        Params& params,
        const std::vector<double>& free_values
    );

    std::vector<double> extract(const Params& params);
}

namespace Nlopt {
    CriterionResult evaluate(
        const RunTask& task,
        const std::vector<double>& free_values
    );

    double score(
        const RunTask& task,
        const std::vector<double>& free_values
    );

#ifdef MULTIRL_HAS_NLOPT
    double objective(
        const std::vector<double>& x,
        std::vector<double>& grad,
        void* data
    );

    nlopt::opt build(
        const NLoptControl& control,
        const std::size_t n_params
    );

    /* ================================================================== *
     * ?????? MLE                                                      *
     * ================================================================== *
     *                                                                       *
     * GN_MLSL / GN_CRS2_LM / GN_ISRES / GN_ESCH ????????? NLopt    *
     * ??? RNG. ??????????????????, ????????.      *
     *                                                                       *
     * ???? std::mt19937(seed) ??????????,                        *
     * ??????????? (LN_BOBYQA ??????), ???.                 *
     * ??? MLSL ???????, ?????????????.                   *
     * ================================================================== */

    EstimateMleResult deterministic_mle(
        const RunTask& task,
        const NLoptControl& control
    );
#endif
}

}  // namespace multiRL
