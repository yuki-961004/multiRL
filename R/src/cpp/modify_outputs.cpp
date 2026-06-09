#include <multiRL/modify_outputs.hpp>

namespace multiRL {
namespace modify_outputs {

/* ========================================================================== *
 *                         NLopt Estimator Builder
 * ========================================================================== */

OutputEstimator nlopt_estimator(
    const std::string& estimator_name,
    const NLoptControl& control
) {
    OutputEstimator out;
    out.name = estimator_name;
    out.backend = "nlopt";
    out.algorithm = control.algorithm;
    out.global_algorithm = control.algorithm;
    out.local_algorithm = control.local_algorithm;

    // Split control into basic type maps for R/Python wrappers.
    out.string_control["algorithm"] = control.algorithm;
    out.string_control["local_algorithm"] = control.local_algorithm;

    out.numeric_control["xtol_rel"] = control.xtol_rel;
    out.numeric_control["maxeval"] =
        static_cast<double>(control.maxeval);
    out.numeric_control["ftol_rel"] = control.ftol_rel;
    out.numeric_control["ftol_abs"] = control.ftol_abs;
    out.numeric_control["xtol_abs"] = control.xtol_abs;
    out.numeric_control["maxtime"] = control.maxtime;
    out.numeric_control["stopval"] = control.stopval;
    out.numeric_control["population"] =
        static_cast<double>(control.population);
    out.numeric_control["initial_step"] = control.initial_step;
    out.numeric_control["local_xtol_rel"] = control.local_xtol_rel;
    out.numeric_control["seed"] =
        static_cast<double>(control.seed);
    out.numeric_control["print_level"] =
        static_cast<double>(control.print_level);
    out.numeric_control["threads"] =
        static_cast<double>(control.threads);

    return out;
}

/* ========================================================================== *
 *                         Stan Estimator Builder
 * ========================================================================== */

OutputEstimator stan_estimator(
    const std::string& estimator_name,
    const MCMCControl& control
) {
    OutputEstimator out;
    const std::string algorithm =
        (control.algorithm == "nuts") ? "NUTS" : "Static HMC";

    out.name = estimator_name;
    out.backend = "stan";
    out.algorithm = algorithm;
    out.global_algorithm = algorithm;
    out.local_algorithm = "";

    // Split MCMC control into basic type maps.
    out.string_control["algorithm"] = control.algorithm;
    out.string_control["progress"] = control.progress;

    out.numeric_control["chains"] =
        static_cast<double>(control.chains);
    out.numeric_control["warmup"] =
        static_cast<double>(control.warmup);
    out.numeric_control["samples"] =
        static_cast<double>(control.samples);
    out.numeric_control["thin"] =
        static_cast<double>(control.thin);
    out.numeric_control["step_size"] = control.step_size;
    out.numeric_control["leapfrog_steps"] =
        static_cast<double>(control.leapfrog_steps);
    out.numeric_control["max_tree_depth"] =
        static_cast<double>(control.max_tree_depth);
    out.numeric_control["target_accept"] = control.target_accept;
    out.numeric_control["min_step_size"] = control.min_step_size;
    out.numeric_control["max_step_size"] = control.max_step_size;
    out.numeric_control["max_delta_energy"] =
        control.max_delta_energy;
    out.numeric_control["initial_jitter"] =
        control.initial_jitter;
    out.numeric_control["seed"] =
        static_cast<double>(control.seed);
    out.numeric_control["print_level"] =
        static_cast<double>(control.print_level);
    out.numeric_control["threads"] =
        static_cast<double>(control.threads);

    out.bool_control["adapt_step_size"] = control.adapt_step_size;

    return out;
}

}  // namespace modify_outputs
}  // namespace multiRL