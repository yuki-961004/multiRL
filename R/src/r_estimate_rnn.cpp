#include <Rcpp.h>

#include <multiRL/estimate_rnn.hpp>

#include "r_wrapper_common.hpp"

#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>

#if defined(_WIN32) && !defined(MULTIRL_HAS_TORCH)
#define MULTIRL_USE_DLL_LOADING 1
#include <windows.h>
#include <memory>
#endif

namespace {

// ===========================================================================
// Result wrapping helpers
// ===========================================================================

Rcpp::DataFrame wrap_rnn_fit(
    const std::vector<multiRL::EstimateRnnSubjectResult>& results
) {
    std::size_t n_rows = results.size();
    std::vector<std::string> param_names;
    for (const auto& result : results) {
        if (!result.parameter_names.empty()) {
            param_names = result.parameter_names;
            break;
        }
    }

    Rcpp::CharacterVector subid(n_rows);
    Rcpp::IntegerVector status(n_rows);
    Rcpp::IntegerVector n_draws(n_rows);
    Rcpp::IntegerVector n_trials(n_rows);
    Rcpp::IntegerVector n_features(n_rows);
    Rcpp::NumericVector loss(n_rows);
    Rcpp::CharacterVector message(n_rows);
    Rcpp::List param_columns(param_names.size());

    for (std::size_t index = 0; index < param_names.size(); ++index) {
        param_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }

    for (std::size_t row = 0; row < n_rows; ++row) {
        const auto& result = results[row];
        subid[static_cast<R_xlen_t>(row)]     = result.subid;
        status[static_cast<R_xlen_t>(row)]    = result.status;
        n_draws[static_cast<R_xlen_t>(row)]   = result.n_draws;
        n_trials[static_cast<R_xlen_t>(row)]  = result.n_trials;
        n_features[static_cast<R_xlen_t>(row)]= result.n_features;
        loss[static_cast<R_xlen_t>(row)]      = result.loss;
        message[static_cast<R_xlen_t>(row)]   = result.message;

        for (std::size_t index = 0; index < param_names.size(); ++index) {
            Rcpp::NumericVector col = param_columns[index];
            if (index < result.estimates.size()) {
                col[static_cast<R_xlen_t>(row)] = result.estimates[index];
            }
        }
    }

    Rcpp::List columns = Rcpp::List::create(
        Rcpp::_["subid"] = subid
    );

    for (std::size_t index = 0; index < param_names.size(); ++index) {
        columns.push_back(param_columns[index], param_names[index]);
    }

    columns.push_back(status,     "status");
    columns.push_back(n_draws,    "n_draws");
    columns.push_back(n_trials,   "n_trials");
    columns.push_back(n_features, "n_features");
    columns.push_back(loss,       "loss");
    columns.push_back(message,    "message");

    return Rcpp::DataFrame(columns);
}

Rcpp::DataFrame wrap_rnn_diagnostics(
    const std::vector<multiRL::EstimateRnnSubjectResult>& results
) {
    std::size_t n_rows = results.size();
    Rcpp::CharacterVector subid(n_rows);
    Rcpp::IntegerVector status(n_rows);
    Rcpp::IntegerVector n_draws(n_rows);
    Rcpp::IntegerVector n_trials(n_rows);
    Rcpp::IntegerVector n_features(n_rows);
    Rcpp::IntegerVector epochs(n_rows);
    Rcpp::NumericVector loss(n_rows);
    Rcpp::CharacterVector architecture(n_rows);
    Rcpp::CharacterVector message(n_rows);

    for (std::size_t row = 0; row < n_rows; ++row) {
        const auto& result = results[row];
        subid[static_cast<R_xlen_t>(row)]        = result.subid;
        status[static_cast<R_xlen_t>(row)]        = result.status;
        n_draws[static_cast<R_xlen_t>(row)]       = result.n_draws;
        n_trials[static_cast<R_xlen_t>(row)]      = result.n_trials;
        n_features[static_cast<R_xlen_t>(row)]    = result.n_features;
        epochs[static_cast<R_xlen_t>(row)]        = result.epochs;
        loss[static_cast<R_xlen_t>(row)]          = result.loss;
        architecture[static_cast<R_xlen_t>(row)]  = result.architecture;
        message[static_cast<R_xlen_t>(row)]       = result.message;
    }

    return Rcpp::DataFrame::create(
        Rcpp::_["subid"]        = subid,
        Rcpp::_["status"]       = status,
        Rcpp::_["n_draws"]      = n_draws,
        Rcpp::_["n_trials"]     = n_trials,
        Rcpp::_["n_features"]   = n_features,
        Rcpp::_["epochs"]       = epochs,
        Rcpp::_["loss"]         = loss,
        Rcpp::_["architecture"] = architecture,
        Rcpp::_["message"]      = message
    );
}

// ===========================================================================
// DLL-based RNN runner (Windows, Rtools/GCC build without LibTorch)
// ===========================================================================
#ifdef MULTIRL_USE_DLL_LOADING

// Function pointer types matching c_backend.h
typedef int  (*fn_rnn_train)   (const double*, int, int, int,
                                 const double*, int,
                                 const int*, const int*, int, int,
                                 const char*, const char*, const char*, double,
                                 int, int, int, int, double, double,
                                 int, int, int, const char*,
                                 void**, char*, int);
typedef int  (*fn_rnn_predict) (void*, const double*, int, int, int,
                                 int, int,
                                 const char*, double*, char*, int);
typedef void (*fn_rnn_free)    (void*);

struct RnnDllHandle {
    HMODULE   hLib    = nullptr;
    fn_rnn_train   train   = nullptr;
    fn_rnn_predict predict = nullptr;
    fn_rnn_free    free_m  = nullptr;

    ~RnnDllHandle() {
        // We do not call FreeLibrary(hLib) here because LibTorch/MSVC runtime
        // may register atexit handlers or global destructors. Unloading the DLL
        // prematurely causes an access violation (0xC0000005) when R exits.
        // Keeping it loaded for the lifetime of the process is safe and avoids this.
    }
};

// Search for multiRL_torch_backend.dll in the usual locations:
//   1. The package libs/x64/ directory (inst/libs/x64/ after install)
//   2. The directory of the R package DLL (multiRLcpp.dll)
//   3. PATH
static std::string find_backend_dll() {
    // Try next to multiRLcpp.dll
    {
        HMODULE hSelf = GetModuleHandleA("multiRLcpp.dll");
        if (!hSelf) { hSelf = GetModuleHandleA(nullptr); }
        char path[MAX_PATH] = {};
        if (GetModuleFileNameA(hSelf, path, MAX_PATH)) {
            std::string dir(path);
            auto pos = dir.rfind('\\');
            if (pos != std::string::npos) { dir = dir.substr(0, pos + 1); }
            std::string candidate = dir + "multiRL_torch_backend.dll";
            if (GetFileAttributesA(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) {
                return candidate;
            }
        }
    }
    // Fall back: let Windows search PATH
    return "multiRL_torch_backend.dll";
}

static std::unique_ptr<RnnDllHandle> load_rnn_dll() {
    std::string dll_path = find_backend_dll();
    HMODULE hLib = LoadLibraryA(dll_path.c_str());
    if (!hLib) {
        DWORD err = GetLastError();
        throw std::runtime_error(
            std::string("Cannot load multiRL_torch_backend.dll (error ") +
            std::to_string(err) + ").\n"
            "Please place multiRL_torch_backend.dll in the same directory as "
            "multiRLcpp.dll or on your system PATH.\n"
            "Download from: https://github.com/yuki-961004/multiRL-remake/releases"
        );
    }

    auto h = std::make_unique<RnnDllHandle>();
    h->hLib    = hLib;
    h->train   = reinterpret_cast<fn_rnn_train>   (GetProcAddress(hLib, "rnn_backend_train"));
    h->predict = reinterpret_cast<fn_rnn_predict> (GetProcAddress(hLib, "rnn_backend_predict"));
    h->free_m  = reinterpret_cast<fn_rnn_free>    (GetProcAddress(hLib, "rnn_backend_free_model"));

    if (!h->train || !h->predict || !h->free_m) {
        throw std::runtime_error(
            "multiRL_torch_backend.dll loaded but required symbols not found. "
            "The DLL version may be incompatible."
        );
    }
    return h;
}

// Run the entire estimate_rnn pipeline through the DLL.
Rcpp::List run_via_dll(
    const std::vector<multiRL::RunTask>& tasks,
    const multiRL::RNNControl& ctrl_in
) {
    // Normalise control
    multiRL::RNNControl ctrl = multiRL::modify_control(ctrl_in, "rnn");

    auto dll = load_rnn_dll();

    const int n_params = static_cast<int>(
        tasks.empty() ? 0 : tasks.front().params.free_names.size()
    );
    const int n_features = 4;

    if (n_params == 0) {
        throw std::runtime_error(
            "estimate_rnn requires at least one free parameter."
        );
    }

    // -------------------------------------------------------------------------
    // Collect training data from the task sampler
    // (we call the core C++ sampler directly 閳?it doesn't need LibTorch)
    // -------------------------------------------------------------------------
    multiRL::TaskSamplerControl sctl;
    sctl.n_draws       = ctrl.n_draws > 0 ? ctrl.n_draws : ctrl.sample;
    sctl.seed          = static_cast<unsigned int>(ctrl.seed);
    sctl.threads       = ctrl.threads;
    sctl.lower_bounds  = ctrl.lower_bounds;
    sctl.upper_bounds  = ctrl.upper_bounds;

    // For individual scope: train one model per subject
    // For shared: train on first subject, predict all
    // For universal: concatenate all subjects, train once
    // This mirrors the logic in estimate_rnn.cpp exactly.

    std::vector<multiRL::EstimateRnnSubjectResult> results;
    results.reserve(tasks.size());

    // Helper: flatten sampler rows into training arrays
    struct FlatRnnData {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<int> lengths;
        std::vector<int> subjects;
    };

    auto flatten_xy = [&](const multiRL::TaskSamplerResult& sampler,
                          int nd, int nt, int subject_index) -> FlatRnnData {
        FlatRnnData result;
        result.x.assign(static_cast<std::size_t>(nd * nt * n_features), 0.0);
        result.y.assign(static_cast<std::size_t>(nd * n_params), 0.0);
        result.lengths.assign(static_cast<std::size_t>(nd), 0);
        result.subjects.assign(static_cast<std::size_t>(nd), subject_index);
        std::vector<bool> seen(static_cast<std::size_t>(nd), false);

        for (const auto& row : sampler.rows) {
            int draw  = row.draw - 1;
            int trial = row.sequence;
            if (draw < 0 || draw >= nd || trial < 0 || trial >= nt) { continue; }

            double ac = 0.0;
            for (std::size_t ci = 0; ci < sampler.cue_names.size(); ++ci) {
                if (sampler.cue_names[ci] == row.simulation) {
                    ac = static_cast<double>(ci + 1);
                    break;
                }
            }
            const std::size_t base = static_cast<std::size_t>(
                (draw * nt + trial) * n_features
            );
            result.x[base] = ac;
            result.x[base + 1] = std::isfinite(row.reward) ? row.reward : 0.0;
            result.x[base + 2] = static_cast<double>(row.block);
            result.x[base + 3] = static_cast<double>(row.trial);
            result.lengths[static_cast<std::size_t>(draw)] = std::max(
                result.lengths[static_cast<std::size_t>(draw)], trial + 1
            );

            if (!seen[static_cast<std::size_t>(draw)]) {
                seen[static_cast<std::size_t>(draw)] = true;
                for (int p = 0; p < n_params; ++p) {
                    if (static_cast<std::size_t>(p) < row.params.size()) {
                        result.y[static_cast<std::size_t>(
                            draw * n_params + p
                        )] = row.params[p];
                    }
                }
            }
        }
        return result;
    };

    // Helper: predict one subject
    auto predict_subject = [&](
        const multiRL::RunTask& task,
        void* model_handle,
        const std::string& subid,
        int nd,
        int subject_index
    ) -> multiRL::EstimateRnnSubjectResult
    {
        multiRL::EstimateRnnSubjectResult res;
        res.subid           = subid;
        res.parameter_names = task.params.free_names;
        res.n_draws         = nd;
        res.n_trials        = static_cast<int>(task.input.n_rows);
        res.n_features      = n_features;
        res.epochs          = ctrl.epochs;
        res.backend         = "torch";
        res.architecture    = ctrl.architecture;

        // Build observed x
        const int n_trials = static_cast<int>(task.input.n_rows);
        std::vector<double> x_obs(
            static_cast<std::size_t>(n_trials * n_features), 0.0
        );
        for (std::size_t row = 0; row < task.input.n_rows; ++row) {
            // action code: find cue index matching the chosen action
            double ac = 0.0;
            const std::string& act = task.input.action[row];
            for (std::size_t ci = 0; ci < task.behrule.cue.size(); ++ci) {
                if (task.behrule.cue[ci] == act) {
                    ac = static_cast<double>(ci + 1);
                    break;
                }
            }
            // reward (chosen): find the reward for the chosen action
            double rew = 0.0;
            for (std::size_t ci = 0; ci < task.input.state[row].size() && ci < task.input.reward_table[row].size(); ++ci) {
                for (const auto& sv : task.input.state[row][ci]) {
                    if (sv == act) { rew = task.input.reward_table[row][ci]; break; }
                }
            }

            const std::size_t base = row * static_cast<std::size_t>(n_features);
            x_obs[base + 0] = ac;
            x_obs[base + 1] = std::isfinite(rew) ? rew : 0.0;
            x_obs[base + 2] = static_cast<double>(task.input.block[row]);
            x_obs[base + 3] = static_cast<double>(task.input.trial[row]);
        }

        std::vector<double> preds(static_cast<std::size_t>(n_params), 0.0);
        char err_buf[512] = {};
        int rc = dll->predict(
            model_handle,
            x_obs.data(),
            n_trials,
            n_features,
            n_params,
            n_trials,
            subject_index,
            ctrl.loss.c_str(),
            preds.data(),
            err_buf, sizeof(err_buf)
        );

        if (rc != 0) {
            res.status  = -1;
            res.message = std::string("predict failed: ") + err_buf;
            return res;
        }

        res.estimates.resize(static_cast<std::size_t>(n_params));
        for (int p = 0; p < n_params; ++p) {
            double val = preds[static_cast<std::size_t>(p)];
            // clamp
            if (static_cast<std::size_t>(p) < ctrl.lower_bounds.size() &&
                std::isfinite(ctrl.lower_bounds[p]) && val < ctrl.lower_bounds[p]) {
                val = ctrl.lower_bounds[p];
            }
            if (static_cast<std::size_t>(p) < ctrl.upper_bounds.size() &&
                std::isfinite(ctrl.upper_bounds[p]) && val > ctrl.upper_bounds[p]) {
                val = ctrl.upper_bounds[p];
            }
            res.estimates[static_cast<std::size_t>(p)] = val;
        }
        res.loss    = 0.0;
        res.status  = 1;
        res.message = "ok";
        return res;
    };

    // Helper: train via DLL
    auto train_via_dll = [&](
        const FlatRnnData& data,
        int nd, int nt, int n_subjects, int embedding_size
    ) -> void*
    {
        void* handle = nullptr;
        char err_buf[512] = {};
        int rc = dll->train(
            data.x.data(), nd, nt, n_features,
            data.y.data(), n_params,
            data.lengths.data(), data.subjects.data(),
            n_subjects, embedding_size,
            ctrl.architecture.c_str(),
            ctrl.loss.c_str(),
            ctrl.regularization.c_str(),
            ctrl.penalty,
            ctrl.epochs, ctrl.batch_size,
            ctrl.units, ctrl.layers,
            ctrl.dropout, ctrl.learning_rate,
            ctrl.seed, ctrl.threads, ctrl.interop_threads,
            ctrl.device.c_str(),
            &handle,
            err_buf, sizeof(err_buf)
        );
        if (rc != 0 || !handle) {
            throw std::runtime_error(
                std::string("rnn_backend_train failed: ") + err_buf
            );
        }
        return handle;
    };

    // ---- scope = "shared" ---------------------------------------------------
    if (ctrl.scope == "shared") {
        const multiRL::RunTask& templ = tasks.front();
        multiRL::TaskSamplerControl sc = sctl;
        const multiRL::TaskSamplerResult sampler = multiRL::task_sampler(templ, sc);
        const int nd = sampler.control.n_draws;
        const int n_trials = static_cast<int>(templ.input.n_rows);

        FlatRnnData data = flatten_xy(sampler, nd, n_trials, 0);
        void* handle = train_via_dll(data, nd, n_trials, 1, 0);

        for (const auto& task : tasks) {
            std::string subid = (!task.input.idinfo.empty() && !task.input.idinfo[0].empty())
                ? task.input.idinfo[0][0] : "1";
            auto res = predict_subject(task, handle, subid, nd, 0);
            results.push_back(std::move(res));
        }
        if (dll->free_m) { dll->free_m(handle); }
    }
    // ---- scope = "universal" ------------------------------------------------
    else if (ctrl.scope == "universal") {
        // Concatenate sampler results across subjects
        int n_trials = 0;
        for (const auto& task : tasks) {
            n_trials = std::max(n_trials, static_cast<int>(task.input.n_rows));
        }
        FlatRnnData data;
        int total_draws = 0;

        for (std::size_t si = 0; si < tasks.size(); ++si) {
            multiRL::TaskSamplerControl sc = sctl;
            sc.seed = sctl.seed + static_cast<unsigned int>(si * sctl.n_draws);
            const multiRL::TaskSamplerResult sampler = multiRL::task_sampler(tasks[si], sc);
            const int nd = sampler.control.n_draws;
            FlatRnnData subject_data = flatten_xy(
                sampler, nd, n_trials, static_cast<int>(si)
            );
            data.x.insert(data.x.end(), subject_data.x.begin(), subject_data.x.end());
            data.y.insert(data.y.end(), subject_data.y.begin(), subject_data.y.end());
            data.lengths.insert(
                data.lengths.end(), subject_data.lengths.begin(),
                subject_data.lengths.end()
            );
            data.subjects.insert(
                data.subjects.end(), subject_data.subjects.begin(),
                subject_data.subjects.end()
            );
            total_draws += nd;
        }

        void* handle = train_via_dll(
            data, total_draws, n_trials, static_cast<int>(tasks.size()),
            ctrl.subject_embedding_size
        );

        for (std::size_t si = 0; si < tasks.size(); ++si) {
            const auto& task = tasks[si];
            std::string subid = (!task.input.idinfo.empty() && !task.input.idinfo[0].empty())
                ? task.input.idinfo[0][0] : "1";
            auto res = predict_subject(
                task, handle, subid, total_draws, static_cast<int>(si)
            );
            results.push_back(std::move(res));
        }
        if (dll->free_m) { dll->free_m(handle); }
    }
    // ---- scope = "individual" (default) ------------------------------------
    else {
        for (const auto& task : tasks) {
            std::string subid = (!task.input.idinfo.empty() && !task.input.idinfo[0].empty())
                ? task.input.idinfo[0][0] : "1";

            const multiRL::TaskSamplerResult sampler = multiRL::task_sampler(task, sctl);
            const int nd = sampler.control.n_draws;
            const int n_trials = static_cast<int>(task.input.n_rows);
            FlatRnnData data = flatten_xy(sampler, nd, n_trials, 0);

            void* handle = nullptr;

            multiRL::EstimateRnnSubjectResult res;
            res.subid           = subid;
            res.parameter_names = task.params.free_names;
            res.n_draws         = nd;
            res.n_trials        = n_trials;
            res.n_features      = n_features;
            res.epochs          = ctrl.epochs;
            res.backend         = "torch";
            res.architecture    = ctrl.architecture;

            try {
                handle = train_via_dll(data, nd, n_trials, 1, 0);
            } catch (const std::exception& ex) {
                res.status  = -1;
                res.message = ex.what();
                results.push_back(std::move(res));
                continue;
            }

            auto pred_res = predict_subject(task, handle, subid, nd, 0);
            pred_res.n_draws = nd;
            if (dll->free_m) { dll->free_m(handle); }
            results.push_back(std::move(pred_res));
        }
    }

    // Build output list
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["fit"] = wrap_rnn_fit(results),
        Rcpp::_["estimator"] = Rcpp::List::create(
            Rcpp::_["name"]          = "RNN",
            Rcpp::_["backend"]       = "torch (DLL)",
            Rcpp::_["architecture"]  = ctrl.architecture,
            Rcpp::_["loss_name"]     = ctrl.loss,
            Rcpp::_["regularization"]= ctrl.regularization,
            Rcpp::_["penalty"]       = ctrl.penalty,
            Rcpp::_["units"]         = ctrl.units,
            Rcpp::_["layers"]        = ctrl.layers,
            Rcpp::_["dropout"]       = ctrl.dropout,
            Rcpp::_["bidirectional"] = (ctrl.architecture.rfind("bi", 0) == 0),
            Rcpp::_["device"]        = ctrl.device,
            Rcpp::_["threads"]       = ctrl.threads,
            Rcpp::_["interop_threads"]= ctrl.interop_threads
        ),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["subjects"] = wrap_rnn_diagnostics(results)
        ),
        Rcpp::_["metadata"] = Rcpp::List::create(
            Rcpp::_["n_draws"]          = ctrl.n_draws,
            Rcpp::_["seed"]             = ctrl.seed,
            Rcpp::_["threads"]          = ctrl.threads,
            Rcpp::_["interop_threads"]  = ctrl.interop_threads,
            Rcpp::_["backend"]          = "torch (DLL)",
            Rcpp::_["parameter_names"]  = Rcpp::wrap(
                tasks.empty() ? std::vector<std::string>()
                              : tasks.front().params.free_names
            )
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_rnn_data", "list"
    );
    return out;
}

#endif  // MULTIRL_USE_DLL_LOADING

}  // namespace

// [[Rcpp::export(name = ".estimate_rnn_data")]]
Rcpp::List r_estimate_rnn_data(
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
    int n_draws,
    int seed,
    int threads,
    int interop_threads,
    int epochs,
    int batch_size,
    int units,
    int layers,
    double dropout,
    double learning_rate,
    std::string architecture,
    std::string loss,
    std::string regularization,
    double penalty,
    int verbose,
    std::string device,
    std::string scope,
    int subject_embedding_size,
    Rcpp::NumericVector lower_bounds,
    Rcpp::NumericVector upper_bounds
) {
    std::vector<multiRL::RunTask> tasks = multiRL_r::make_subject_tasks(
        object, reward, action, block, trial,
        idinfo, exinfo, cue, rsp,
        params, free_names, system,
        prior_names, prior_types, prior_param1, prior_param2,
        prior_active, generate, name, mode, "RNN"
    );

    multiRL::RNNControl control;
    control.n_draws          = n_draws;
    control.sample           = n_draws;
    control.seed             = seed;
    control.threads          = threads;
    control.interop_threads  = interop_threads;
    control.epoch            = epochs;
    control.epochs           = epochs;
    control.batch_size       = batch_size;
    control.units            = units;
    control.layers           = layers;
    control.dropout          = dropout;
    control.learning_rate    = learning_rate;
    control.architecture     = architecture;
    control.loss             = loss;
    control.regularization   = regularization;
    control.penalty          = penalty;
    control.verbose          = verbose;
    control.device           = device;
    control.scope            = scope;
    control.subject_embedding_size = subject_embedding_size;
    control.lower_bounds     = multiRL_r::as_double_vector(lower_bounds);
    control.upper_bounds     = multiRL_r::as_double_vector(upper_bounds);

#ifdef MULTIRL_USE_DLL_LOADING
    // Windows R (Rtools/GCC): delegate entirely to the MSVC backend DLL
    return run_via_dll(tasks, control);
#else
    // Native LibTorch build (MSVC Python build, or Linux/macOS)
    const std::vector<multiRL::EstimateRnnSubjectResult> results =
        multiRL::estimate_rnn(tasks, control);

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["fit"] = wrap_rnn_fit(results),
        Rcpp::_["estimator"] = Rcpp::List::create(
            Rcpp::_["name"]           = "RNN",
            Rcpp::_["backend"]        = "torch",
            Rcpp::_["architecture"]   = control.architecture,
            Rcpp::_["loss_name"]      = control.loss,
            Rcpp::_["regularization"] = control.regularization,
            Rcpp::_["penalty"]        = control.penalty,
            Rcpp::_["units"]          = control.units,
            Rcpp::_["layers"]         = control.layers,
            Rcpp::_["dropout"]        = control.dropout,
            Rcpp::_["bidirectional"]  = (control.architecture.rfind("bi", 0) == 0),
            Rcpp::_["device"]         = control.device,
            Rcpp::_["threads"]        = control.threads,
            Rcpp::_["interop_threads"]= control.interop_threads
        ),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["subjects"] = wrap_rnn_diagnostics(results)
        ),
        Rcpp::_["metadata"] = Rcpp::List::create(
            Rcpp::_["n_draws"]         = control.n_draws,
            Rcpp::_["seed"]            = control.seed,
            Rcpp::_["threads"]         = control.threads,
            Rcpp::_["interop_threads"] = control.interop_threads,
            Rcpp::_["backend"]         = "torch",
            Rcpp::_["parameter_names"] = free_names
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_rnn_data", "list"
    );
    return out;
#endif
}
