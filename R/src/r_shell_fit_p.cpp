#include <Rcpp.h>
#include <string>
#include <vector>

// Forward declarations of Rcpp wrapper functions:
Rcpp::List r_estimate_mle(
    Rcpp::CharacterMatrix object, Rcpp::NumericMatrix reward, Rcpp::CharacterVector action,
    Rcpp::IntegerVector block, Rcpp::IntegerVector trial, Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo, Rcpp::CharacterVector cue, Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params, Rcpp::CharacterVector free_names, Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names, Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1, Rcpp::NumericVector prior_param2,
    bool prior_active, bool generate, std::string name, std::string mode,
    int maxeval, std::string algorithm, std::string local_algorithm,
    double xtol_rel, double local_xtol_rel, int seed,
    Rcpp::NumericVector lower_bounds, Rcpp::NumericVector upper_bounds
);

Rcpp::List r_estimate_map(
    Rcpp::CharacterMatrix object, Rcpp::NumericMatrix reward, Rcpp::CharacterVector action,
    Rcpp::IntegerVector block, Rcpp::IntegerVector trial, Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo, Rcpp::CharacterVector cue, Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params, Rcpp::CharacterVector free_names, Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names, Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1, Rcpp::NumericVector prior_param2,
    bool prior_active, bool generate, std::string name, std::string mode,
    int mle_maxeval, int map_maxiter, double map_tol, int map_patience,
    std::string algorithm, std::string local_algorithm, double xtol_rel, double local_xtol_rel,
    int seed, Rcpp::NumericVector lower_bounds, Rcpp::NumericVector upper_bounds
);

Rcpp::List r_estimate_mcmc(
    Rcpp::CharacterMatrix object, Rcpp::NumericMatrix reward, Rcpp::CharacterVector action,
    Rcpp::IntegerVector block, Rcpp::IntegerVector trial, Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo, Rcpp::CharacterVector cue, Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params, Rcpp::CharacterVector free_names, Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names, Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1, Rcpp::NumericVector prior_param2,
    bool prior_active, bool generate, std::string name, std::string mode,
    int warmup, int samples, int chains, int thin, double step_size, double target_accept,
    int max_tree_depth, long seed, std::string algorithm,
    Rcpp::NumericVector lower_bounds, Rcpp::NumericVector upper_bounds
);

Rcpp::List r_estimate_abc(
    Rcpp::CharacterMatrix object, Rcpp::NumericMatrix reward, Rcpp::CharacterVector action,
    Rcpp::IntegerVector block, Rcpp::IntegerVector trial, Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo, Rcpp::CharacterVector cue, Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params, Rcpp::CharacterVector free_names, Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names, Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1, Rcpp::NumericVector prior_param2,
    bool prior_active, bool generate, std::string name, std::string mode,
    int samples, double tol, std::string method, std::string reduction, int n_comp,
    int fake_block, std::string scope, int seed, int threads, int print_level,
    Rcpp::NumericVector lower_bounds, Rcpp::NumericVector upper_bounds
);

Rcpp::List r_estimate_rnn_data(
    Rcpp::CharacterMatrix object, Rcpp::NumericMatrix reward, Rcpp::CharacterVector action,
    Rcpp::IntegerVector block, Rcpp::IntegerVector trial, Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo, Rcpp::CharacterVector cue, Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params, Rcpp::CharacterVector free_names, Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names, Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1, Rcpp::NumericVector prior_param2,
    bool prior_active, bool generate, std::string name, std::string mode,
    int n_draws, int seed, int threads, int interop_threads,
    int epochs, int batch_size, int units, int layers, double dropout, double learning_rate,
    std::string architecture, std::string loss, std::string regularization, double penalty,
    int verbose, std::string device, std::string scope, int subject_embedding_size,
    Rcpp::NumericVector lower_bounds, Rcpp::NumericVector upper_bounds
);

// [[Rcpp::export(name = ".shell_fit_p")]]
Rcpp::List r_shell_fit_p(
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
    bool generate,
    std::string name,
    std::string mode,
    std::string estimator,
    Rcpp::List control
) {
    if (estimator == "mle") {
        int maxeval = control["maxeval"];
        std::string algorithm = control["algorithm"];
        std::string local_algorithm = control["local_algorithm"];
        double xtol_rel = control["xtol_rel"];
        double local_xtol_rel = control["local_xtol_rel"];
        int seed = control["seed"];
        Rcpp::NumericVector lower_bounds = control["lower_bounds"];
        Rcpp::NumericVector upper_bounds = control["upper_bounds"];
        return r_estimate_mle(
            object, reward, action, block, trial, idinfo, exinfo, cue, rsp,
            params, free_names, system, prior_names, prior_types, prior_param1, prior_param2,
            prior_active, generate, name, mode,
            maxeval, algorithm, local_algorithm, xtol_rel, local_xtol_rel, seed, lower_bounds, upper_bounds
        );
    }
    if (estimator == "map") {
        int maxeval = control["mle_maxeval"];
        int map_maxiter = control["map_maxiter"];
        double map_tol = control["map_tol"];
        int map_patience = control["map_patience"];
        std::string algorithm = control["algorithm"];
        std::string local_algorithm = control["local_algorithm"];
        double xtol_rel = control["xtol_rel"];
        double local_xtol_rel = control["local_xtol_rel"];
        int seed = control["seed"];
        Rcpp::NumericVector lower_bounds = control["lower_bounds"];
        Rcpp::NumericVector upper_bounds = control["upper_bounds"];
        return r_estimate_map(
            object, reward, action, block, trial, idinfo, exinfo, cue, rsp,
            params, free_names, system, prior_names, prior_types, prior_param1, prior_param2,
            prior_active, generate, name, mode,
            maxeval, map_maxiter, map_tol, map_patience, algorithm, local_algorithm, xtol_rel, local_xtol_rel,
            seed, lower_bounds, upper_bounds
        );
    }
    if (estimator == "mcmc") {
        int warmup = control["warmup"];
        int samples = control["samples"];
        int chains = control["chains"];
        int thin = control["thin"];
        double step_size = control["step_size"];
        double target_accept = control["target_accept"];
        int max_tree_depth = control["max_tree_depth"];
        long seed = control["seed"];
        std::string algorithm = control["algorithm"];
        Rcpp::NumericVector lower_bounds = control["lower_bounds"];
        Rcpp::NumericVector upper_bounds = control["upper_bounds"];
        return r_estimate_mcmc(
            object, reward, action, block, trial, idinfo, exinfo, cue, rsp,
            params, free_names, system, prior_names, prior_types, prior_param1, prior_param2,
            prior_active, generate, name, mode,
            warmup, samples, chains, thin, step_size, target_accept, max_tree_depth, seed, algorithm,
            lower_bounds, upper_bounds
        );
    }
    if (estimator == "abc") {
        int samples = control["samples"];
        double tol = control["tol"];
        std::string method = control["method"];
        std::string reduction = control["reduction"];
        int n_comp = control["n_comp"];
        int fake_block = control["fake_block"];
        std::string scope = control["scope"];
        int seed = control["seed"];
        int threads = control["threads"];
        int print_level = control["print_level"];
        Rcpp::NumericVector lower_bounds = control["lower_bounds"];
        Rcpp::NumericVector upper_bounds = control["upper_bounds"];
        return r_estimate_abc(
            object, reward, action, block, trial, idinfo, exinfo, cue, rsp,
            params, free_names, system, prior_names, prior_types, prior_param1, prior_param2,
            prior_active, generate, name, mode,
            samples, tol, method, reduction, n_comp, fake_block, scope, seed, threads, print_level,
            lower_bounds, upper_bounds
        );
    }
    if (estimator == "rnn") {
        int n_draws = control["n_draws"];
        int seed = control["seed"];
        int threads = control["threads"];
        int interop_threads = control["interop_threads"];
        int epochs = control["epochs"];
        int batch_size = control["batch_size"];
        int units = control["units"];
        int layers = control["layers"];
        double dropout = control["dropout"];
        double learning_rate = control["learning_rate"];
        std::string architecture = control["architecture"];
        std::string loss = control["loss"];
        std::string regularization = control["regularization"];
        double penalty = control["penalty"];
        int verbose = control["verbose"];
        std::string device = control["device"];
        std::string scope = control["scope"];
        int subject_embedding_size = control.containsElementNamed(
            "subject_embedding_size"
        ) ? Rcpp::as<int>(control["subject_embedding_size"]) : 8;
        Rcpp::NumericVector lower_bounds = control["lower_bounds"];
        Rcpp::NumericVector upper_bounds = control["upper_bounds"];
        return r_estimate_rnn_data(
            object, reward, action, block, trial, idinfo, exinfo, cue, rsp,
            params, free_names, system, prior_names, prior_types, prior_param1, prior_param2,
            prior_active, generate, name, mode,
            n_draws, seed, threads, interop_threads, epochs, batch_size, units, layers, dropout, learning_rate,
            architecture, loss, regularization, penalty, verbose, device, scope,
            subject_embedding_size, lower_bounds, upper_bounds
        );
    }
    Rcpp::stop("Unknown estimator: " + estimator);
    return Rcpp::List::create();
}
