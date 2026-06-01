#include <Rcpp.h>

#include <multiRL/modify_params.hpp>
#include <multiRL/modify_priors.hpp>
#include <multiRL/process_MDP_free.hpp>
#include <multiRL/shell_run_m.hpp>
#include <multiRL/task_builder.hpp>

#include "cpp/func_alpha.cpp"
#include "cpp/func_beta.cpp"
#include "cpp/func_gamma.cpp"
#include "cpp/func_delta.cpp"
#include "cpp/func_epsilon.cpp"
#include "cpp/func_zeta.cpp"
#include "cpp/process_MDP_free.cpp"
#include "cpp/criterion_likelihood.cpp"
#include "cpp/criterion_prior.cpp"
#include "cpp/criterion_posterior.cpp"
#include "cpp/modify_params.cpp"
#include "cpp/modify_priors.cpp"
#include "cpp/task_builder.cpp"
#include "cpp/shell_run_m.cpp"

namespace {

multiRL::StringMatrix as_string_matrix(
    const Rcpp::CharacterMatrix& matrix
) {
    multiRL::StringMatrix out(
        static_cast<std::size_t>(matrix.nrow()),
        std::vector<std::string>(static_cast<std::size_t>(matrix.ncol()))
    );

    for (R_xlen_t row = 0; row < matrix.nrow(); ++row) {
        for (R_xlen_t col = 0; col < matrix.ncol(); ++col) {
            out[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                Rcpp::as<std::string>(matrix(row, col));
        }
    }

    return out;
}

multiRL::DoubleMatrix as_double_matrix(
    const Rcpp::NumericMatrix& matrix
) {
    multiRL::DoubleMatrix out(
        static_cast<std::size_t>(matrix.nrow()),
        std::vector<double>(static_cast<std::size_t>(matrix.ncol()))
    );

    for (R_xlen_t row = 0; row < matrix.nrow(); ++row) {
        for (R_xlen_t col = 0; col < matrix.ncol(); ++col) {
            out[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                matrix(row, col);
        }
    }

    return out;
}

std::vector<std::string> as_string_vector(
    const Rcpp::CharacterVector& vector
) {
    std::vector<std::string> out(static_cast<std::size_t>(vector.size()));

    for (R_xlen_t index = 0; index < vector.size(); ++index) {
        out[static_cast<std::size_t>(index)] =
            Rcpp::as<std::string>(vector[index]);
    }

    return out;
}

std::vector<int> as_int_vector(const Rcpp::IntegerVector& vector) {
    std::vector<int> out(static_cast<std::size_t>(vector.size()));

    for (R_xlen_t index = 0; index < vector.size(); ++index) {
        out[static_cast<std::size_t>(index)] = vector[index];
    }

    return out;
}

std::vector<double> as_double_vector(const Rcpp::NumericVector& vector) {
    std::vector<double> out(static_cast<std::size_t>(vector.size()));

    for (R_xlen_t index = 0; index < vector.size(); ++index) {
        out[static_cast<std::size_t>(index)] = vector[index];
    }

    return out;
}

multiRL::Params as_params(
    const Rcpp::NumericVector& params,
    const Rcpp::CharacterVector& free_names
) {
    Rcpp::CharacterVector names = params.names();
    std::vector<std::string> cpp_names;
    std::vector<double> cpp_values;
    cpp_names.reserve(static_cast<std::size_t>(params.size()));
    cpp_values.reserve(static_cast<std::size_t>(params.size()));

    for (R_xlen_t index = 0; index < params.size(); ++index) {
        cpp_names.push_back(Rcpp::as<std::string>(names[index]));
        cpp_values.push_back(params[index]);
    }

    return multiRL::modify_params(
        cpp_names,
        cpp_values,
        as_string_vector(free_names)
    );
}

multiRL::PriorGroup as_priors(
    const Rcpp::CharacterVector& prior_names,
    const Rcpp::CharacterVector& prior_types,
    const Rcpp::NumericVector& prior_param1,
    const Rcpp::NumericVector& prior_param2,
    const Rcpp::CharacterVector& free_names,
    const bool prior_active
) {
    return multiRL::modify_priors(
        as_string_vector(prior_names),
        as_string_vector(prior_types),
        as_double_vector(prior_param1),
        as_double_vector(prior_param2),
        as_string_vector(free_names),
        prior_active
    );
}

Rcpp::NumericMatrix wrap_double_matrix(
    const multiRL::DoubleMatrix& matrix
) {
    if (matrix.empty()) {
        return Rcpp::NumericMatrix(0, 0);
    }

    Rcpp::NumericMatrix out(
        static_cast<int>(matrix.size()),
        static_cast<int>(matrix[0].size())
    );

    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t col = 0; col < matrix[row].size(); ++col) {
            out(static_cast<int>(row), static_cast<int>(col)) =
                matrix[row][col];
        }
    }

    return out;
}

Rcpp::CharacterMatrix wrap_string_matrix(
    const multiRL::StringMatrix& matrix
) {
    if (matrix.empty()) {
        return Rcpp::CharacterMatrix(0, 0);
    }

    Rcpp::CharacterMatrix out(
        static_cast<int>(matrix.size()),
        static_cast<int>(matrix[0].size())
    );

    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t col = 0; col < matrix[row].size(); ++col) {
            out(static_cast<int>(row), static_cast<int>(col)) =
                matrix[row][col];
        }
    }

    return out;
}

Rcpp::List wrap_result(
    const multiRL::RunResult& result,
    const Rcpp::CharacterVector& system
) {
    Rcpp::List value(system.size());
    for (R_xlen_t index = 0; index < system.size(); ++index) {
        value[index] = wrap_double_matrix(
            result.result.value[static_cast<std::size_t>(index)]
        );
    }
    value.names() = system;

    return Rcpp::List::create(
        Rcpp::_["metric"] = Rcpp::List::create(
            Rcpp::_["ACC"] = result.metric.acc,
            Rcpp::_["LogL"] = result.metric.log_likelihood,
            Rcpp::_["LogPr"] = result.metric.log_prior,
            Rcpp::_["LogPo"] = result.metric.log_posterior,
            Rcpp::_["NLL"] = result.metric.nll,
            Rcpp::_["AIC"] = result.metric.aic,
            Rcpp::_["BIC"] = result.metric.bic
        ),
        Rcpp::_["result"] = Rcpp::List::create(
            Rcpp::_["value"] = value,
            Rcpp::_["bias"] = wrap_double_matrix(result.result.bias),
            Rcpp::_["shown"] = wrap_double_matrix(result.result.shown),
            Rcpp::_["prob"] = wrap_double_matrix(result.result.prob),
            Rcpp::_["count"] = wrap_double_matrix(result.result.count),
            Rcpp::_["behave"] = wrap_string_matrix(result.result.behave),
            Rcpp::_["exploration"] = result.result.exploration,
            Rcpp::_["latent"] = result.result.latent,
            Rcpp::_["reward"] = result.result.reward,
            Rcpp::_["utility"] = result.result.utility,
            Rcpp::_["simulation"] = result.result.simulation,
            Rcpp::_["position"] = result.result.position
        )
    );
}

}  // namespace

// [[Rcpp::export(name = ".cpp_shell_run_m")]]
Rcpp::List r_shell_run_m(
    Rcpp::CharacterMatrix object,
    Rcpp::NumericMatrix reward,
    Rcpp::CharacterVector action,
    Rcpp::IntegerVector block,
    Rcpp::IntegerVector trial,
    Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo,
    Rcpp::CharacterVector cue,
    Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params,
    Rcpp::CharacterVector free_names,
    Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names,
    Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1,
    Rcpp::NumericVector prior_param2,
    bool prior_active,
    std::string policy,
    std::string name,
    std::string mode,
    std::string estimate
) {
    multiRL::Process1Input input = multiRL::process_1_input(
        as_string_matrix(object),
        as_double_matrix(reward),
        as_string_vector(action),
        as_int_vector(block),
        as_int_vector(trial),
        as_string_matrix(idinfo),
        as_string_matrix(exinfo)
    );

    multiRL::Process2Behrule behrule = multiRL::process_2_behrule(
        as_string_vector(cue),
        as_string_vector(rsp)
    );

    multiRL::Settings settings;
    settings.policy = policy;
    settings.name = name;
    settings.mode = mode;
    settings.estimate = estimate;
    settings.system = as_string_vector(system);

    multiRL::RunTask task = multiRL::task_builder(
        input,
        behrule,
        as_params(params, free_names),
        settings,
        as_priors(
            prior_names,
            prior_types,
            prior_param1,
            prior_param2,
            free_names,
            prior_active
        )
    );

    multiRL::RunResult result = multiRL::shell_run_m(task);
    return wrap_result(result, system);
}
